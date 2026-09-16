from __future__ import absolute_import
from __future__ import print_function

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from mimic4benchmark.readers import FixedHorizonIcuExitReader


HEADER = 'Hours,Heart Rate\n'


def write_stay(root, name):
    with open(os.path.join(root, name), 'w') as f:
        f.write(HEADER)
        f.write('0,80\n')
        f.write('12,90\n')
        f.write('24,100\n')
        f.write('25,110\n')


def write_listfile(root):
    path = os.path.join(root, 'listfile.csv')
    with open(path, 'w') as f:
        f.write('stay,period_length,y_true\n')
        f.write('stay_a.csv,24,10\n')
        f.write('stay_b.csv,23,1\n')
        f.write('stay_c.csv,24.0,30\n')
        f.write('stay_d.csv,24,200\n')
    return path


def make_data():
    root = tempfile.mkdtemp()
    for name in ('stay_a.csv', 'stay_b.csv', 'stay_c.csv', 'stay_d.csv'):
        write_stay(root, name)
    write_listfile(root)
    return root


def labels_for(root, horizon):
    return FixedHorizonIcuExitReader(root, horizon=horizon).get_labels()


def test_reader():
    root = make_data()
    try:
        reader = FixedHorizonIcuExitReader(root, horizon=24)
        assert reader.get_stay_names() == ['stay_a.csv', 'stay_c.csv', 'stay_d.csv']
        assert reader.get_number_of_examples() == 3
        assert reader.get_labels() == [1, 0, 0]

        assert labels_for(root, 12) == [1, 0, 0]
        assert labels_for(root, 48) == [1, 1, 0]

        by_horizon = [labels_for(root, h) for h in FixedHorizonIcuExitReader.VALID_HORIZONS]
        for labels in zip(*by_horizon):
            assert list(labels) == sorted(labels)

        example = reader.read_example(0)
        assert example['t'] == 24.0
        assert max(float(row[0]) for row in example['X']) <= 24.0

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
