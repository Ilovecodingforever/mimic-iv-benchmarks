from __future__ import absolute_import
from __future__ import print_function

import json
import os
import pickle
import platform

import numpy as np

from mimic4models import common_utils
from mimic4models.preprocessing import DISCRETIZER_EPS


class RawSequenceEncoder(object):
    """Encode original irregular rows as value features plus observation masks."""

    def __init__(self, config_path=os.path.join(os.path.dirname(__file__), '..', 'resources', 'discretizer_config.json')):
        with open(config_path) as f:
            config = json.load(f)
        self._id_to_channel = config['id_to_channel']
        self._channel_to_id = dict(zip(self._id_to_channel, range(len(self._id_to_channel))))
        self._is_categorical_channel = config['is_categorical_channel']
        self._possible_values = config['possible_values']
        self._begin_pos = []
        self._end_pos = []
        cur_len = 0
        for channel in self._id_to_channel:
            self._begin_pos.append(cur_len)
            if self._is_categorical_channel[channel]:
                cur_len += len(self._possible_values[channel])
            else:
                cur_len += 1
            self._end_pos.append(cur_len)
        self._value_dim = cur_len
        self._mask_dim = len(self._id_to_channel)
        self._feature_dim = self._value_dim + self._mask_dim
        self._header = self._make_header()

    @property
    def feature_dim(self):
        return self._feature_dim

    @property
    def value_dim(self):
        return self._value_dim

    def header(self):
        return list(self._header)

    def continuous_value_fields(self):
        fields = []
        for channel_id, channel in enumerate(self._id_to_channel):
            if not self._is_categorical_channel[channel]:
                fields.append(self._begin_pos[channel_id])
        return fields

    def continuous_mask_fields(self):
        fields = []
        for channel_id, channel in enumerate(self._id_to_channel):
            if not self._is_categorical_channel[channel]:
                fields.append(self._value_dim + channel_id)
        return fields

    def _make_header(self):
        names = []
        for channel in self._id_to_channel:
            if self._is_categorical_channel[channel]:
                for value in self._possible_values[channel]:
                    names.append(channel + '->' + value)
            else:
                names.append(channel)
        for channel in self._id_to_channel:
            names.append('mask->' + channel)
        return names

    def _write_value(self, data, row_id, channel, value):
        channel_id = self._channel_to_id[channel]
        begin = self._begin_pos[channel_id]
        if self._is_categorical_channel[channel]:
            category_id = self._possible_values[channel].index(value)
            data[row_id, begin + category_id] = 1.0
        else:
            data[row_id, begin] = float(value)
        data[row_id, self._value_dim + channel_id] = 1.0

    def transform(self, X, header=None, end=None):
        if header is None:
            header = ['Hours'] + self._id_to_channel
        assert header[0] == 'Hours'
        rows = []
        for row in X:
            if end is not None and float(row[0]) > end + DISCRETIZER_EPS:
                continue
            rows.append(row)
        data = np.zeros((len(rows), self._feature_dim), dtype=float)
        for row_id, row in enumerate(rows):
            for col_id in range(1, len(row)):
                value = row[col_id]
                if value == '':
                    continue
                channel = header[col_id]
                self._write_value(data, row_id, channel, value)
        return data, ','.join(self._header)


class RawObservedNormalizer(object):
    """Normalize continuous raw value columns using observed training values only."""

    def __init__(self, fields, mask_fields):
        self._fields = list(fields)
        self._mask_fields = list(mask_fields)
        if len(self._fields) != len(self._mask_fields):
            raise ValueError('fields and mask_fields must have the same length')
        self._sum_x = np.zeros((len(self._fields),), dtype=float)
        self._sum_sq_x = np.zeros((len(self._fields),), dtype=float)
        self._counts = np.zeros((len(self._fields),), dtype=float)
        self._means = None
        self._stds = None

    def _feed_data(self, x):
        for i, field in enumerate(self._fields):
            observed = x[:, self._mask_fields[i]] > 0.5
            values = x[observed, field]
            if values.shape[0] == 0:
                continue
            self._sum_x[i] += np.sum(values)
            self._sum_sq_x[i] += np.sum(values ** 2)
            self._counts[i] += values.shape[0]

    def _save_params(self, save_file_path):
        eps = 1e-7
        self._means = np.zeros((len(self._fields),), dtype=float)
        self._stds = np.ones((len(self._fields),), dtype=float)
        for i in range(len(self._fields)):
            n = self._counts[i]
            if n <= 0:
                self._means[i] = 0.0
                self._stds[i] = 1.0
            elif n == 1:
                self._means[i] = self._sum_x[i] / n
                self._stds[i] = 1.0
            else:
                mean = self._sum_x[i] / n
                var = (self._sum_sq_x[i] - 2.0 * self._sum_x[i] * mean + n * mean ** 2) / (n - 1)
                self._means[i] = mean
                self._stds[i] = max(np.sqrt(max(var, 0.0)), eps)
        dirname = os.path.dirname(save_file_path)
        if dirname:
            common_utils.create_directory(dirname)
        with open(save_file_path, 'wb') as save_file:
            pickle.dump(obj={'means': self._means,
                             'stds': self._stds,
                             'fields': self._fields,
                             'mask_fields': self._mask_fields,
                             'counts': self._counts},
                        file=save_file,
                        protocol=2)

    def load_params(self, load_file_path):
        with open(load_file_path, 'rb') as load_file:
            if platform.python_version()[0] == '2':
                dct = pickle.load(load_file)
            else:
                dct = pickle.load(load_file, encoding='latin1')
        self._means = dct['means']
        self._stds = dct['stds']
        self._fields = list(dct.get('fields', self._fields))
        self._mask_fields = list(dct.get('mask_fields', self._mask_fields))
        self._counts = dct.get('counts')

    def transform(self, X):
        ret = 1.0 * X
        for i, field in enumerate(self._fields):
            observed = ret[:, self._mask_fields[i]] > 0.5
            ret[observed, field] = (ret[observed, field] - self._means[i]) / self._stds[i]
            ret[~observed, field] = 0.0
        return ret


def is_raw_timestep(timestep):
    return abs(float(timestep)) <= DISCRETIZER_EPS


def load_raw_data(reader, encoder, normalizer, small_part=False, return_names=False):
    n_examples = reader.get_number_of_examples()
    if small_part:
        n_examples = min(n_examples, 1000)
    ret = common_utils.read_chunk(reader, n_examples)
    header = ret['header']
    data = [encoder.transform(X, header=header, end=t)[0]
            for (X, t) in zip(ret['X'], ret['t'])]
    if normalizer is not None:
        data = [normalizer.transform(X) for X in data]
    data = common_utils.pad_zeros(data)
    whole_data = (data, np.array(ret['y']))
    if not return_names:
        return whole_data
    return {'data': whole_data, 'names': ret['name']}


def raw_sequence_length_stats(reader, encoder, small_part=False):
    n_examples = reader.get_number_of_examples()
    if small_part:
        n_examples = min(n_examples, 1000)
    lengths = []
    for i in range(n_examples):
        ex = reader.read_example(i)
        encoded, _ = encoder.transform(ex['X'], header=ex['header'], end=ex['t'])
        lengths.append(encoded.shape[0])
    lengths = np.asarray(lengths, dtype=float)
    if lengths.shape[0] == 0:
        return {'n': 0, 'min': np.nan, 'median': np.nan, 'p90': np.nan,
                'p95': np.nan, 'p99': np.nan, 'max': np.nan}
    return {'n': int(lengths.shape[0]),
            'min': int(np.min(lengths)),
            'median': float(np.percentile(lengths, 50)),
            'p90': float(np.percentile(lengths, 90)),
            'p95': float(np.percentile(lengths, 95)),
            'p99': float(np.percentile(lengths, 99)),
            'max': int(np.max(lengths))}


def print_raw_sequence_length_stats(reader, encoder, small_part=False):
    stats = raw_sequence_length_stats(reader, encoder, small_part=small_part)
    print('Raw sequence length statistics:')
    for key in ['n', 'min', 'median', 'p90', 'p95', 'p99', 'max']:
        print('  {}: {}'.format(key, stats[key]))
    return stats
