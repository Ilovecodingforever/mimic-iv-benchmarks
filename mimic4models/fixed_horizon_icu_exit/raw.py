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

    def __init__(self, config_path=os.path.join(os.path.dirname(__file__), '..', 'resources', 'discretizer_config.json'),
                 include_timestamps=False):
        with open(config_path) as f:
            config = json.load(f)
        self._include_timestamps = bool(include_timestamps)
        self._timestamp_name = 'Hours'
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
        self._timestamp_pos = self._value_dim + self._mask_dim
        self._feature_dim = self._timestamp_pos + (1 if self._include_timestamps else 0)
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
        if self._include_timestamps:
            names.append(self._timestamp_name)
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
        data = np.zeros((len(rows), self._feature_dim), dtype=np.float32)
        for row_id, row in enumerate(rows):
            if self._include_timestamps:
                data[row_id, self._timestamp_pos] = float(row[0])
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
        ret = np.asarray(X, dtype=np.float32).copy()
        for i, field in enumerate(self._fields):
            observed = ret[:, self._mask_fields[i]] > 0.5
            ret[observed, field] = (ret[observed, field] - self._means[i]) / self._stds[i]
            ret[~observed, field] = 0.0
        return ret


def is_raw_timestep(timestep):
    return abs(float(timestep)) <= DISCRETIZER_EPS


def _load_raw_examples(reader, encoder, normalizer, small_part=False):
    n_examples = reader.get_number_of_examples()
    if small_part:
        n_examples = min(n_examples, 1000)
    ret = common_utils.read_chunk(reader, n_examples)
    header = ret['header']
    data = [encoder.transform(X, header=header, end=t)[0]
            for (X, t) in zip(ret['X'], ret['t'])]
    if normalizer is not None:
        data = [normalizer.transform(X) for X in data]
    return ret, data


def load_raw_data_unpadded(reader, encoder, normalizer, small_part=False, return_names=False):
    ret, data = _load_raw_examples(reader, encoder, normalizer, small_part)
    whole_data = (data, np.array(ret['y']))
    if not return_names:
        return whole_data
    return {'data': whole_data, 'names': ret['name']}


def load_raw_data(reader, encoder, normalizer, small_part=False, return_names=False):
    ret, data = _load_raw_examples(reader, encoder, normalizer, small_part)
    data = common_utils.pad_zeros(data)
    whole_data = (data, np.array(ret['y']))
    if not return_names:
        return whole_data
    return {'data': whole_data, 'names': ret['name']}


def pad_raw_batch(sequences, dtype=np.float32):
    if len(sequences) == 0:
        raise ValueError('Cannot pad an empty raw batch.')
    if dtype is None:
        dtype = sequences[0].dtype
    feature_shape = sequences[0].shape[1:]
    max_len = max(x.shape[0] for x in sequences)
    batch = np.zeros((len(sequences), max_len) + feature_shape, dtype=dtype)
    for i, seq in enumerate(sequences):
        if seq.shape[1:] != feature_shape:
            raise ValueError('Raw batch feature shapes differ: {} vs {}'.format(
                seq.shape[1:], feature_shape))
        batch[i, :seq.shape[0]] = seq.astype(dtype, copy=False)
    return batch


class RawBatchSequence(object):
    def __init__(self, sequences, labels, batch_size, shuffle=True, bucket_size=None,
                 seed=None, target_repl=False, dtype=np.float32, prepare_input=None):
        if len(sequences) != len(labels):
            raise ValueError('sequences and labels must have the same length')
        if len(sequences) == 0:
            raise ValueError('RawBatchSequence requires at least one example')
        self.sequences = list(sequences)
        self.labels = np.asarray(labels)
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError('batch_size must be positive')
        self.shuffle = bool(shuffle)
        self.bucket_size = int(bucket_size or max(self.batch_size * 100, self.batch_size))
        self.bucket_size = max(self.bucket_size, self.batch_size)
        self.target_repl = bool(target_repl)
        self.dtype = dtype
        self.prepare_input = prepare_input
        self.rng = np.random.RandomState(seed)
        self.lengths = np.asarray([x.shape[0] for x in self.sequences], dtype=int)
        if np.any(self.lengths <= 0):
            raise ValueError('Raw sequences must contain at least one observed row')
        feature_shape = self.sequences[0].shape[1:]
        for seq in self.sequences:
            if seq.ndim != 2 or seq.shape[1:] != feature_shape:
                raise ValueError('All raw sequences must have shape (T, feature_dim)')
        self._batches = []
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(float(len(self.sequences)) / float(self.batch_size)))

    def _make_batches(self, ordered):
        return [ordered[i:i + self.batch_size]
                for i in range(0, len(ordered), self.batch_size)]

    def on_epoch_end(self):
        order = np.argsort(self.lengths)
        batches = self._make_batches(order)
        if self.shuffle:
            for batch in batches:
                self.rng.shuffle(batch)
            self.rng.shuffle(batches)
        self._batches = batches

    def batch_indices(self, batch_index):
        return self._batches[batch_index]

    def padding_efficiency(self):
        real_steps = 0.0
        allocated_steps = 0.0
        batch_max_lengths = []
        for indices in self._batches:
            lengths = self.lengths[indices]
            max_len = int(np.max(lengths))
            real_steps += float(np.sum(lengths))
            allocated_steps += float(len(indices) * max_len)
            batch_max_lengths.append(max_len)
        efficiency = real_steps / allocated_steps if allocated_steps else np.nan
        return {'real_steps': real_steps,
                'allocated_steps': allocated_steps,
                'padding_efficiency': efficiency,
                'padding_overhead': 1.0 - efficiency if not np.isnan(efficiency) else np.nan,
                'mean_real_length': float(np.mean(self.lengths)),
                'mean_batch_max_length': float(np.mean(batch_max_lengths))}

    def print_diagnostics(self):
        lengths = self.lengths.astype(float)
        eff = self.padding_efficiency()
        global_allocated_steps = float(len(self.sequences) * np.max(self.lengths))
        global_efficiency = eff['real_steps'] / global_allocated_steps if global_allocated_steps else np.nan
        global_overhead = 1.0 - global_efficiency if not np.isnan(global_efficiency) else np.nan
        print('Raw batching:')
        print('  n examples: {}'.format(len(self.sequences)))
        print('  batch size: {}'.format(self.batch_size))
        print('  min length: {}'.format(int(np.min(lengths))))
        print('  median length: {:.1f}'.format(float(np.percentile(lengths, 50))))
        print('  p90 length: {:.1f}'.format(float(np.percentile(lengths, 90))))
        print('  p95 length: {:.1f}'.format(float(np.percentile(lengths, 95))))
        print('  p99 length: {:.1f}'.format(float(np.percentile(lengths, 99))))
        print('  max length: {}'.format(int(np.max(lengths))))
        print('  mean real sequence length: {:.1f}'.format(eff['mean_real_length']))
        print('  mean batch max length: {:.1f}'.format(eff['mean_batch_max_length']))
        print('  global padding efficiency: {:.1%}'.format(global_efficiency))
        print('  global padding overhead: {:.1%}'.format(global_overhead))
        print('  batch-local padding efficiency: {:.1%}'.format(eff['padding_efficiency']))
        print('  batch-local padding overhead: {:.1%}'.format(eff['padding_overhead']))
        eff['global_padding_efficiency'] = global_efficiency
        eff['global_padding_overhead'] = global_overhead
        return eff


    def __getitem__(self, batch_index):
        indices = self.batch_indices(batch_index)
        x_batch = pad_raw_batch([self.sequences[i] for i in indices], dtype=self.dtype)
        y_batch = self.labels[indices]
        prepared_x_batch = self.prepare_input(x_batch) if self.prepare_input is not None else x_batch
        if not self.target_repl:
            return prepared_x_batch, y_batch
        time_steps = prepared_x_batch[0].shape[1] if isinstance(prepared_x_batch, list) else x_batch.shape[1]
        y_repl = np.expand_dims(y_batch, axis=-1).repeat(time_steps, axis=1)
        y_repl = np.expand_dims(y_repl, axis=-1)
        return prepared_x_batch, [y_batch, y_repl]

    def iter_batches(self):
        while True:
            for i in range(len(self)):
                yield self[i]
            self.on_epoch_end()


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
