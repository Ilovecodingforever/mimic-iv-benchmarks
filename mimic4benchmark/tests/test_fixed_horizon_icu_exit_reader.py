from __future__ import absolute_import
from __future__ import print_function

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from mimic4benchmark.readers import FixedHorizonIcuExitReader
from mimic4models.fixed_horizon_icu_exit.main import default_normalizer_state_path


HEADER = 'Hours,Heart Rate\n'


def write_stay(root, name):
    with open(os.path.join(root, name), 'w') as f:
        f.write(HEADER)
        f.write('0,80\n')
        f.write('12,90\n')
        f.write('24,100\n')
        f.write('25,110\n')


ROWS = [
    ('stay_a.csv', 24, 10),
    ('stay_b.csv', 23, 1),
    ('stay_c.csv', 24.0, 30),
    ('stay_d.csv', 24, 200),
    ('stay_12.csv', 24, 12),
    ('stay_12_plus.csv', 24, 12.000001),
    ('stay_24.csv', 24, 24),
    ('stay_24_plus.csv', 24, 24.000001),
    ('stay_48.csv', 24, 48),
    ('stay_48_plus.csv', 24, 48.000001),
    ('stay_96.csv', 24, 96),
    ('stay_96_plus.csv', 24, 96.000001),
    ('stay_168.csv', 24, 168),
    ('stay_168_plus.csv', 24, 168.000001),
]


def write_listfile(root):
    path = os.path.join(root, 'listfile.csv')
    with open(path, 'w') as f:
        f.write('stay,period_length,y_true\n')
        for name, period_length, y_true in ROWS:
            f.write('{},{},{}\n'.format(name, period_length, y_true))
    return path


def make_data():
    root = tempfile.mkdtemp()
    for name, _, _ in ROWS:
        write_stay(root, name)
    write_listfile(root)
    return root


def labels_for(root, horizon):
    return FixedHorizonIcuExitReader(root, horizon=horizon).get_labels()


def label_map(root, horizon):
    reader = FixedHorizonIcuExitReader(root, horizon=horizon)
    return dict(zip(reader.get_stay_names(), reader.get_labels()))


def test_reader():
    root = make_data()
    try:
        reader = FixedHorizonIcuExitReader(root, horizon=24)
        assert 'stay_b.csv' not in reader.get_stay_names()
        assert reader.get_number_of_examples() == len(ROWS) - 1
        assert label_map(root, 24)['stay_a.csv'] == 1
        assert label_map(root, 24)['stay_c.csv'] == 0
        assert label_map(root, 24)['stay_d.csv'] == 0

        for horizon in FixedHorizonIcuExitReader.VALID_HORIZONS:
            labels = label_map(root, horizon)
            assert labels['stay_{}.csv'.format(horizon)] == 1
            assert labels['stay_{}_plus.csv'.format(horizon)] == 0

        by_horizon = [labels_for(root, h) for h in FixedHorizonIcuExitReader.VALID_HORIZONS]
        for labels in zip(*by_horizon):
            assert list(labels) == sorted(labels)

        example = reader.read_example(0)
        assert example['t'] == 24.0
        assert max(float(row[0]) for row in example['X']) <= 24.0

        assert default_normalizer_state_path('/normalizers', 8.0, 'previous', 123) == (
            '/normalizers/fixed_horizon_icu_exit_ts:8.00_impute:previous_start:zero_masks:True_n:123.normalizer')

        try:
            FixedHorizonIcuExitReader(root, horizon=13)
        except ValueError as exc:
            assert 'Unsupported horizon' in str(exc)
        else:
            raise AssertionError('unsupported horizon did not raise')
    finally:
        shutil.rmtree(root)


if __name__ == '__main__':
    test_reader()
    print('ok')
