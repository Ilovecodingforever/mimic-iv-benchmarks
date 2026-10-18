from __future__ import absolute_import
from __future__ import print_function

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from mimic4benchmark.readers import FixedHorizonDecompensationReader


HEADER = 'Hours,Heart Rate\n'


def write_stay(root, name):
    with open(os.path.join(root, name), 'w') as f:
        f.write(HEADER)
        f.write('0,80\n')
        f.write('24,100\n')
        f.write('24.000001,110\n')


def write_data():
    root = tempfile.mkdtemp()
    for name in ('stay_12.csv', 'stay_12_boundary.csv', 'stay_24.csv',
                 'stay_alive.csv', 'stay_dead_before_24.csv'):
        write_stay(root, name)
    with open(os.path.join(root, 'listfile.csv'), 'w') as f:
        f.write('stay,period_length,y_true\n')
        f.write('stay_12.csv,23,0\n')
        f.write('stay_12.csv,24,1\n')
        f.write('stay_12_boundary.csv,24,1\n')
        f.write('stay_24.csv,24,1\n')
        f.write('stay_alive.csv,24,0\n')
    with open(os.path.join(root, 'mortality_metadata.csv'), 'w') as f:
        f.write('stay,mortality,lived_time,intime,deathtime,label_24h\n')
        f.write('stay_12.csv,1,35,0,,1\n')
        f.write('stay_12_boundary.csv,1,36,0,,1\n')
        f.write('stay_24.csv,1,47,0,,1\n')
        f.write('stay_alive.csv,0,1000000000000000000,0,,0\n')
        f.write('stay_dead_before_24.csv,1,20,0,,1\n')
    return root


def test_reader_labels_and_fixed_window():
    root = write_data()
    try:
        readers = [FixedHorizonDecompensationReader(root, horizon=h)
                   for h in FixedHorizonDecompensationReader.VALID_HORIZONS]
        names = readers[0].get_stay_names()
        assert names == ['stay_12.csv', 'stay_12_boundary.csv',
                         'stay_24.csv', 'stay_alive.csv']
        assert all(reader.get_stay_names() == names for reader in readers)

        by_horizon = [reader.get_labels() for reader in readers]
        assert by_horizon[0] == [1, 0, 0, 0]
        assert by_horizon[1] == [1, 1, 1, 0]
        for labels in zip(*by_horizon):
            assert list(labels) == sorted(labels)

        example = readers[0].read_example(0)
        assert example['t'] == 24.0
        assert max(float(row[0]) for row in example['X']) <= 24.0
    finally:
        shutil.rmtree(root)


def test_unsupported_horizon_fails():
    root = write_data()
    try:
        try:
            FixedHorizonDecompensationReader(root, horizon=13)
        except ValueError as exc:
            assert 'Unsupported horizon' in str(exc)
        else:
            raise AssertionError('unsupported horizon did not raise')
    finally:
        shutil.rmtree(root)


def test_reader_rejects_death_before_prediction_time():
    root = write_data()
    try:
        with open(os.path.join(root, 'listfile.csv'), 'a') as f:
            f.write('stay_dead_before_24.csv,24,1\n')
        try:
            FixedHorizonDecompensationReader(root, horizon=24)
        except ValueError as exc:
            assert 'not eligible at the 24h prediction time' in str(exc)
        else:
            raise AssertionError('ineligible stay did not fail')
    finally:
        shutil.rmtree(root)


def test_normalizer_path_is_task_specific():
    from mimic4models.fixed_horizon_decompensation.main import normalizer_path

    assert normalizer_path('/normalizers', 4.0, 'previous', 123) == (
        '/normalizers/fixed_horizon_decompensation_ts:4.00_impute:previous_start:zero_masks:True_n:123.normalizer')


if __name__ == '__main__':
    test_reader_labels_and_fixed_window()
    test_unsupported_horizon_fails()
    test_reader_rejects_death_before_prediction_time()
    test_normalizer_path_is_task_specific()
    print('ok')
