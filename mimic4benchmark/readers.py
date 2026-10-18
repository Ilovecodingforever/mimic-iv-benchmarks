from __future__ import absolute_import
from __future__ import print_function

import csv
import os
import numpy as np
import random

from mimic4benchmark.decompensation import (
    PREDICTION_TIME, VALID_HORIZONS, mortality_label)


class Reader(object):
    def __init__(self, dataset_dir, listfile=None):
        self._dataset_dir = dataset_dir
        self._current_index = 0
        if listfile is None:
            listfile_path = os.path.join(dataset_dir, "listfile.csv")
        else:
            listfile_path = listfile
        with open(listfile_path, "r") as lfile:
            self._data = lfile.readlines()
        self._listfile_header = self._data[0]
        self._data = self._data[1:]

    def get_number_of_examples(self):
        return len(self._data)

    def random_shuffle(self, seed=None):
        if seed is not None:
            random.seed(seed)
        random.shuffle(self._data)

    def read_example(self, index):
        raise NotImplementedError()

    def read_next(self):
        to_read_index = self._current_index
        self._current_index += 1
        if self._current_index == self.get_number_of_examples():
            self._current_index = 0
        return self.read_example(to_read_index)


class DecompensationReader(Reader):
    def __init__(self, dataset_dir, listfile=None):
        """ Reader for decompensation prediction task.
        :param dataset_dir: Directory where timeseries files are stored.
        :param listfile:    Path to a listfile. If this parameter is left `None` then
                            `dataset_dir/listfile.csv` will be used.
        """
        Reader.__init__(self, dataset_dir, listfile)
        self._data = [line.split(',') for line in self._data]
        self._data = [(x, float(t), int(y)) for (x, t, y) in self._data]

    def _read_timeseries(self, ts_filename, time_bound):
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                t = float(mas[0])
                if t > time_bound + 1e-6:
                    break
                ret.append(np.array(mas))
        return (np.stack(ret), header)

    def read_example(self, index):
        """ Read the example with given index.

        :param index: Index of the line of the listfile to read (counting starts from 0).
        :return: Directory with the following keys:
            X : np.array
                2D array containing all events. Each row corresponds to a moment.
                First column is the time and other columns correspond to different
                variables.
            t : float
                Length of the data in hours. Note, in general, it is not equal to the
                timestamp of last event.
            y : int (0 or 1)
                Mortality within next 24 hours.
            header : array of strings
                Names of the columns. The ordering of the columns is always the same.
            name: Name of the sample.
        """
        if index < 0 or index >= len(self._data):
            raise ValueError("Index must be from 0 (inclusive) to number of examples (exclusive).")

        name = self._data[index][0]
        t = self._data[index][1]
        y = self._data[index][2]
        (X, header) = self._read_timeseries(name, t)

        return {"X": X,
                "t": t,
                "y": y,
                "header": header,
                "name": name}


class InHospitalMortalityReader(Reader):
    def __init__(self, dataset_dir, listfile=None, period_length=48.0):
        """ Reader for in-hospital moratality prediction task.

        :param dataset_dir:   Directory where timeseries files are stored.
        :param listfile:      Path to a listfile. If this parameter is left `None` then
                              `dataset_dir/listfile.csv` will be used.
        :param period_length: Length of the period (in hours) from which the prediction is done.
        """
        Reader.__init__(self, dataset_dir, listfile)
        self._data = [line.split(',') for line in self._data]
        self._data = [(x, int(y)) for (x, y) in self._data]
        self._period_length = period_length

    def _read_timeseries(self, ts_filename):
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                ret.append(np.array(mas))
        return (np.stack(ret), header)

    def read_example(self, index):
        """ Reads the example with given index.

        :param index: Index of the line of the listfile to read (counting starts from 0).
        :return: Dictionary with the following keys:
            X : np.array
                2D array containing all events. Each row corresponds to a moment.
                First column is the time and other columns correspond to different
                variables.
            t : float
                Length of the data in hours. Note, in general, it is not equal to the
                timestamp of last event.
            y : int (0 or 1)
                In-hospital mortality.
            header : array of strings
                Names of the columns. The ordering of the columns is always the same.
            name: Name of the sample.
        """
        if index < 0 or index >= len(self._data):
            raise ValueError("Index must be from 0 (inclusive) to number of lines (exclusive).")

        name = self._data[index][0]
        t = self._period_length
        y = self._data[index][1]
        (X, header) = self._read_timeseries(name)

        return {"X": X,
                "t": t,
                "y": y,
                "header": header,
                "name": name}


class FixedHorizonIcuExitReader(Reader):
    VALID_HORIZONS = (12, 24, 48, 96, 168)
    PERIOD_LENGTH = 24.0

    def __init__(self, dataset_dir, listfile=None, horizon=24):
        """Reader for fixed-24h ICU-exit prediction from LOS listfiles.

        Uses existing length-of-stay rows at period_length == 24h and converts
        remaining LOS hours into y = int(y_true <= horizon).
        """
        horizon = int(horizon)
        if horizon not in self.VALID_HORIZONS:
            raise ValueError("Unsupported horizon {}. Valid horizons are {}.".format(
                horizon, ", ".join(map(str, self.VALID_HORIZONS))))

        Reader.__init__(self, dataset_dir, listfile)
        self._horizon = horizon
        rows = []
        for line in self._data:
            if not line.strip():
                continue
            name, period_length, y_true = line.strip().split(',')
            period_length = float(period_length)
            if abs(period_length - self.PERIOD_LENGTH) >= 1e-6:
                continue
            assert abs(period_length - self.PERIOD_LENGTH) < 1e-6
            remaining_los = float(y_true)
            rows.append((name, remaining_los, int(remaining_los <= horizon)))

        names = [row[0] for row in rows]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate stay found after filtering to 24h in {}".format(
                listfile if listfile is not None else os.path.join(dataset_dir, "listfile.csv")))
        self._data = rows

    def get_stay_names(self):
        return [row[0] for row in self._data]

    def get_remaining_los(self):
        return [row[1] for row in self._data]

    def get_labels(self):
        return [row[2] for row in self._data]

    def _read_timeseries(self, ts_filename, time_bound):
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                t = float(mas[0])
                if t > time_bound + 1e-6:
                    break
                ret.append(np.array(mas))
        return (np.stack(ret), header)

    def read_example(self, index):
        if index < 0 or index >= len(self._data):
            raise ValueError("Index must be from 0 (inclusive) to number of lines (exclusive).")

        name = self._data[index][0]
        y = self._data[index][2]
        (X, header) = self._read_timeseries(name, self.PERIOD_LENGTH)

        return {"X": X,
                "t": self.PERIOD_LENGTH,
                "y": y,
                "header": header,
                "name": name}


class FixedHorizonDecompensationReader(Reader):
    VALID_HORIZONS = VALID_HORIZONS
    PERIOD_LENGTH = PREDICTION_TIME

    def __init__(self, dataset_dir, listfile=None, horizon=24,
                 metadata_path=None):
        """Reader for fixed-time mortality labels from decompensation data.

        Existing decompensation listfiles define the eligible examples and the
        metadata sidecar supplies the death timing needed for other horizons.
        """
        horizon = int(horizon)
        if horizon not in self.VALID_HORIZONS:
            raise ValueError("Unsupported horizon {}. Valid horizons are {}.".format(
                horizon, ", ".join(map(str, self.VALID_HORIZONS))))

        Reader.__init__(self, dataset_dir, listfile)
        if metadata_path is None:
            listfile_path = listfile if listfile is not None else os.path.join(
                dataset_dir, 'listfile.csv')
            metadata_path = os.path.join(os.path.dirname(listfile_path),
                                         'mortality_metadata.csv')
        metadata = self._read_metadata(metadata_path)

        rows = []
        for line in self._data:
            if not line.strip():
                continue
            fields = line.strip().split(',')
            if len(fields) != 3:
                raise ValueError('Expected stay,period_length,y_true in {}'.format(
                    listfile if listfile is not None else 'listfile.csv'))
            name, period_length, source_label = fields
            period_length = float(period_length)
            if abs(period_length - self.PERIOD_LENGTH) >= 1e-6:
                continue
            if name not in metadata:
                raise ValueError('Missing mortality metadata for {}'.format(name))
            row = metadata[name]
            if row['lived_time'] + 1e-6 < self.PERIOD_LENGTH:
                raise ValueError('{} is not eligible at the 24h prediction time'.format(name))
            label_24h = mortality_label(
                row['mortality'], row['lived_time'], self.PERIOD_LENGTH, 24)
            if int(source_label) != label_24h or int(row['label_24h']) != label_24h:
                raise ValueError('24h label mismatch for {}'.format(name))
            label = mortality_label(
                row['mortality'], row['lived_time'], self.PERIOD_LENGTH, horizon)
            rows.append((name, row['lived_time'], label))

        names = [row[0] for row in rows]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate stay found after filtering to 24h in {}'.format(
                listfile if listfile is not None else os.path.join(dataset_dir, 'listfile.csv')))
        self._horizon = horizon
        self._metadata = metadata
        self._data = rows

    @staticmethod
    def _read_metadata(metadata_path):
        required = {'stay', 'mortality', 'lived_time', 'label_24h'}
        metadata = {}
        with open(metadata_path, 'r') as metadata_file:
            for row in csv.DictReader(metadata_file):
                if not required.issubset(set(row)):
                    raise ValueError('Mortality metadata is missing required columns: {}'.format(
                        sorted(required)))
                name = row['stay']
                if name in metadata:
                    raise ValueError('Duplicate mortality metadata for {}'.format(name))
                metadata[name] = {
                    'mortality': int(row['mortality']),
                    'lived_time': float(row['lived_time']),
                    'label_24h': int(row['label_24h']),
                }
        return metadata

    def get_stay_names(self):
        return [row[0] for row in self._data]

    def get_lived_times(self):
        return [row[1] for row in self._data]

    def get_labels(self):
        return [row[2] for row in self._data]

    def _read_timeseries(self, ts_filename, time_bound):
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), 'r') as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == 'Hours'
            for line in tsfile:
                mas = line.strip().split(',')
                t = float(mas[0])
                if t > time_bound + 1e-9:
                    break
                ret.append(np.array(mas))
        if not ret:
            raise ValueError('No measurements at or before {}h for {}'.format(
                time_bound, ts_filename))
        return np.stack(ret), header

    def read_example(self, index):
        if index < 0 or index >= len(self._data):
            raise ValueError('Index must be from 0 (inclusive) to number of examples (exclusive).')
        name = self._data[index][0]
        y = self._data[index][2]
        X, header = self._read_timeseries(name, self.PERIOD_LENGTH)
        return {'X': X,
                't': self.PERIOD_LENGTH,
                'y': y,
                'header': header,
                'name': name}


class LengthOfStayReader(Reader):
    def __init__(self, dataset_dir, listfile=None):
        """ Reader for length of stay prediction task.

        :param dataset_dir: Directory where timeseries files are stored.
        :param listfile:    Path to a listfile. If this parameter is left `None` then
                            `dataset_dir/listfile.csv` will be used.
        """
        Reader.__init__(self, dataset_dir, listfile)
        self._data = [line.split(',') for line in self._data]
        self._data = [(x, float(t), float(y)) for (x, t, y) in self._data]

    def _read_timeseries(self, ts_filename, time_bound):
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                t = float(mas[0])
                if t > time_bound + 1e-6:
                    break
                ret.append(np.array(mas))
        return (np.stack(ret), header)

    def read_example(self, index):
        """ Reads the example with given index.

        :param index: Index of the line of the listfile to read (counting starts from 0).
        :return: Dictionary with the following keys:
            X : np.array
                2D array containing all events. Each row corresponds to a moment.
                First column is the time and other columns correspond to different
                variables.
            t : float
                Length of the data in hours. Note, in general, it is not equal to the
                timestamp of last event.
            y : float
                Remaining time in ICU.
            header : array of strings
                Names of the columns. The ordering of the columns is always the same.
            name: Name of the sample.
        """
        if index < 0 or index >= len(self._data):
            raise ValueError("Index must be from 0 (inclusive) to number of lines (exclusive).")

        name = self._data[index][0]
        t = self._data[index][1]
        y = self._data[index][2]
        (X, header) = self._read_timeseries(name, t)

        return {"X": X,
                "t": t,
                "y": y,
                "header": header,
                "name": name}


class PhenotypingReader(Reader):
    def __init__(self, dataset_dir, listfile=None):
        """ Reader for phenotype classification task.

        :param dataset_dir: Directory where timeseries files are stored.
        :param listfile:    Path to a listfile. If this parameter is left `None` then
                            `dataset_dir/listfile.csv` will be used.
        """
        Reader.__init__(self, dataset_dir, listfile)
        self._data = [line.split(',') for line in self._data]
        self._data = [(mas[0], float(mas[1]), list(map(int, mas[2:]))) for mas in self._data]

    def _read_timeseries(self, ts_filename):
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                ret.append(np.array(mas))
        return (np.stack(ret), header)

    def read_example(self, index):
        """ Reads the example with given index.

        :param index: Index of the line of the listfile to read (counting starts from 0).
        :return: Dictionary with the following keys:
            X : np.array
                2D array containing all events. Each row corresponds to a moment.
                First column is the time and other columns correspond to different
                variables.
            t : float
                Length of the data in hours. Note, in general, it is not equal to the
                timestamp of last event.
            y : array of ints
                Phenotype labels.
            header : array of strings
                Names of the columns. The ordering of the columns is always the same.
            name: Name of the sample.
        """
        if index < 0 or index >= len(self._data):
            raise ValueError("Index must be from 0 (inclusive) to number of lines (exclusive).")

        name = self._data[index][0]
        t = self._data[index][1]
        y = self._data[index][2]
        (X, header) = self._read_timeseries(name)

        return {"X": X,
                "t": t,
                "y": y,
                "header": header,
                "name": name}


class MultitaskReader(Reader):
    def __init__(self, dataset_dir, listfile=None):
        """ Reader for multitask learning.

        :param dataset_dir: Directory where timeseries files are stored.
        :param listfile:    Path to a listfile. If this parameter is left `None` then
                            `dataset_dir/listfile.csv` will be used.
        """
        Reader.__init__(self, dataset_dir, listfile)
        self._data = [line.split(',') for line in self._data]

        def process_ihm(x):
            return list(map(int, x.split(';')))

        def process_los(x):
            x = x.split(';')
            if x[0] == '':
                return ([], [])
            return (list(map(int, x[:len(x)//2])), list(map(float, x[len(x)//2:])))

        def process_ph(x):
            return list(map(int, x.split(';')))

        def process_decomp(x):
            x = x.split(';')
            if x[0] == '':
                return ([], [])
            return (list(map(int, x[:len(x)//2])), list(map(int, x[len(x)//2:])))

        self._data = [(fname, float(t), process_ihm(ihm), process_los(los),
                       process_ph(pheno), process_decomp(decomp))
                      for fname, t, ihm, los, pheno, decomp in self._data]

    def _read_timeseries(self, ts_filename):
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                ret.append(np.array(mas))
        return (np.stack(ret), header)

    def read_example(self, index):
        """ Reads the example with given index.

        :param index: Index of the line of the listfile to read (counting starts from 0).
        :return: Return dictionary with the following keys:
            X : np.array
                2D array containing all events. Each row corresponds to a moment.
                First column is the time and other columns correspond to different
                variables.
            t : float
                Length of the data in hours. Note, in general, it is not equal to the
                timestamp of last event.
            ihm : array
                Array of 3 integers: [pos, mask, label].
            los : array
                Array of 2 arrays: [masks, labels].
            pheno : array
                Array of 25 binary integers (phenotype labels).
            decomp : array
                Array of 2 arrays: [masks, labels].
            header : array of strings
                Names of the columns. The ordering of the columns is always the same.
            name: Name of the sample.
        """
        if index < 0 or index >= len(self._data):
            raise ValueError("Index must be from 0 (inclusive) to number of lines (exclusive).")

        name = self._data[index][0]
        (X, header) = self._read_timeseries(name)

        return {"X": X,
                "t": self._data[index][1],
                "ihm": self._data[index][2],
                "los": self._data[index][3],
                "pheno": self._data[index][4],
                "decomp": self._data[index][5],
                "header": header,
                "name": name}
