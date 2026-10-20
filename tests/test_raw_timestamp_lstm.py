from __future__ import absolute_import

import argparse
import inspect
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mimic4models import common_utils
from mimic4models.fixed_horizon_icu_exit import main as fixed_main
from mimic4models.fixed_horizon_icu_exit.matched_count_control.sampling import (
    SamplingReader, apply_sampling_to_example, random_matched_sample, structured_sample)
from mimic4models.fixed_horizon_icu_exit.raw import (
    RawObservedNormalizer, RawSequenceEncoder, load_raw_data_unpadded, pad_raw_batch)


def _continuous_channel(encoder):
    for channel in encoder._id_to_channel:
        if not encoder._is_categorical_channel[channel]:
            return channel
    raise AssertionError('test fixture requires a continuous channel')


def _raw_rows(times, values=None, encoder=None):
    encoder = encoder or RawSequenceEncoder()
    header = ['Hours'] + encoder._id_to_channel
    channel = _continuous_channel(encoder)
    channel_i = header.index(channel)
    if values is None:
        values = list(range(len(times)))
    rows = []
    for time, value in zip(times, values):
        row = [''] * len(header)
        row[0] = str(time)
        if value is not None:
            row[channel_i] = str(value)
        rows.append(row)
    return np.asarray(rows, dtype=object), header


def _example(times, values=None, name='toy'):
    encoder = RawSequenceEncoder()
    X, header = _raw_rows(times, values, encoder)
    return {'X': X, 'header': header, 't': 24.0, 'y': 0, 'name': name}


def _time_gap_column(header):
    return header.split(',').index('TimeSincePreviousInputDays')


def test_time_gaps_preserve_measurements_and_masks():
    plain_encoder = RawSequenceEncoder()
    delta_encoder = RawSequenceEncoder(include_time_deltas=True)
    X, header = _raw_rows([0, 1, 4, 8], [10, 20, 30, 40], plain_encoder)

    plain, plain_header = plain_encoder.transform(X, header=header, end=24.0)
    delta, delta_header = delta_encoder.transform(X, header=header, end=24.0)
    delta_i = _time_gap_column(delta_header)

    assert delta.shape[1] == plain.shape[1] + 1
    assert delta_header.split(',')[:-1] == plain_header.split(',')
    assert np.array_equal(delta[:, :plain.shape[1]], plain)
    assert np.allclose(delta[:, delta_i], [0.0, 1.0 / 24.0, 3.0 / 24.0, 4.0 / 24.0])


def test_time_gaps_handle_irregular_repeated_and_single_rows():
    encoder = RawSequenceEncoder(include_time_deltas=True)
    X, header = _raw_rows([0, 0, 2.5, 2.5, 9.5], [0, 1, 2, 3, 4], encoder)
    encoded, encoded_header = encoder.transform(X, header=header, end=24.0)
    gap_i = _time_gap_column(encoded_header)

    assert np.allclose(encoded[:, gap_i], [0.0, 0.0, 2.5 / 24.0, 0.0, 7.0 / 24.0])

    one, _ = _raw_rows([17.0], [0.0], encoder)
    single, single_header = encoder.transform(one, header=header, end=24.0)
    assert single.shape[0] == 1
    assert single[0, _time_gap_column(single_header)] == 0.0


def test_sampling_precedes_time_gap_calculation_for_dense_and_sparse_inputs():
    dense = _example(
        [0.5, 1.0, 3.0, 3.5, 4.1, 5.0, 7.0, 8.0, 12.0],
        [501, 999, 503, 535, 541, 550, 570, 580, 512],
        name='dense')
    sparse = _example([3.5, 7.0, 8.0, 12.0], [535, 570, 580, 512], name='sparse')
    encoder = RawSequenceEncoder(include_time_deltas=True)

    dense_sampled, dense_cells = structured_sample(
        dense['X'], dense['header'], sampling_interval=4, end=24.0)
    sparse_sampled, sparse_cells = structured_sample(
        sparse['X'], sparse['header'], sampling_interval=4, end=24.0)
    assert np.array_equal(dense_sampled, sparse_sampled)
    assert [cell['value'] for cell in dense_cells] == [cell['value'] for cell in sparse_cells]

    encoded, encoded_header = encoder.transform(
        dense_sampled, header=dense['header'], end=24.0)
    gap_i = _time_gap_column(encoded_header)
    retained_times = np.asarray(dense_sampled[:, 0], dtype=float)
    expected = np.r_[0.0, np.diff(retained_times) / 24.0]
    assert np.allclose(encoded[:, gap_i], expected)
    assert 999.0 not in encoded[:, encoded_header.split(',').index('Heart Rate')]


def test_random_matched_sampling_uses_retained_timestamps_only():
    example = _example(
        [0.5, 1.0, 3.0, 3.5, 4.0, 4.1, 7.0, 8.0, 12.0],
        [501, 999, 503, 535, 540, 541, 570, 580, 512],
        name='random-toy')
    sampled, _ = random_matched_sample(
        example['X'], example['header'], sampling_interval=4,
        sampling_seed=100, stay_name=example['name'], end=24.0)
    encoder = RawSequenceEncoder(include_time_deltas=True)
    encoded, encoded_header = encoder.transform(
        sampled, header=example['header'], end=24.0)
    retained_times = np.asarray(sampled[:, 0], dtype=float)
    expected = np.r_[0.0, np.diff(retained_times) / 24.0]

    assert np.allclose(encoded[:, _time_gap_column(encoded_header)], expected)
    assert encoded.shape[1] == RawSequenceEncoder().feature_dim + 1


def test_sampling_reader_pipeline_encodes_after_selection():
    example = _example([3.0, 4.0, 4.1, 8.0], [30, 40, 41, 80])
    encoder = RawSequenceEncoder(include_time_deltas=True)

    class Reader(object):
        def __init__(self, item):
            self.item = item
            self.reads = 0

        def get_number_of_examples(self):
            return 1

        def read_next(self):
            self.reads += 1
            return self.item

    for strategy in ('structured', 'random_matched'):
        expected = apply_sampling_to_example(example, strategy, 4, 100)
        expected_data, _ = encoder.transform(
            expected['X'], header=expected['header'], end=expected['t'])
        loaded = load_raw_data_unpadded(
            SamplingReader(Reader(example), strategy, 4, 100), encoder, None)
        assert np.array_equal(loaded[0][0], expected_data)


def test_train_validation_test_raw_loaders_share_timestamp_representation():
    encoder = RawSequenceEncoder(include_time_deltas=True)

    class Reader(object):
        def __init__(self, item):
            self.item = item

        def get_number_of_examples(self):
            return 1

        def read_next(self):
            return self.item

    splits = [
        _example([0.0, 1.0], [10, 20], name='train'),
        _example([2.0, 2.0], [30, 40], name='validation'),
        _example([5.0], [50], name='test'),
    ]
    train, validation = fixed_main.load_train_val_raw(
        Reader(splits[0]), Reader(splits[1]), encoder, None, False, raw_mode=True)
    test = fixed_main.load_test_raw(Reader(splits[2]), encoder, None, False, raw_mode=True)

    loaded = [train[0][0], validation[0][0], test['data'][0][0]]
    assert all(sequence.shape[1] == encoder.feature_dim for sequence in loaded)
    assert np.allclose(loaded[0][:, -1], [0.0, 1.0 / 24.0])
    assert np.allclose(loaded[1][:, -1], [0.0, 0.0])
    assert np.allclose(loaded[2][:, -1], [0.0])


def test_valid_zero_measurements_remain_distinguishable_from_padding():
    encoder = RawSequenceEncoder(include_time_deltas=True)
    X, header = _raw_rows([0.0, 0.0, 2.0], [0.0, 0.0, 0.0], encoder)
    encoded, _ = encoder.transform(X, header=header, end=24.0)

    # A zero-valued measurement still has an observation-mask bit set.
    assert np.all(np.any(encoded != 0.0, axis=1))
    padded = pad_raw_batch([encoded, encoded[:1]])
    assert np.all(padded[1, 1:] == 0.0)
    assert np.all(np.any(padded[0, :3] != 0.0, axis=1))


def test_raw_normalizer_reuses_state_without_touching_masks_or_time_gaps():
    plain_encoder = RawSequenceEncoder()
    delta_encoder = RawSequenceEncoder(include_time_deltas=True)
    X, header = _raw_rows([0.0, 24.0], [70.0, 90.0], plain_encoder)
    plain, plain_header = plain_encoder.transform(X, header=header, end=24.0)
    delta, delta_header = delta_encoder.transform(X, header=header, end=24.0)

    normalizer = RawObservedNormalizer(
        delta_encoder.continuous_value_fields(),
        delta_encoder.continuous_mask_fields())
    normalizer._means = np.zeros(len(normalizer._fields), dtype=float)
    normalizer._stds = np.ones(len(normalizer._fields), dtype=float)
    hr = plain_header.split(',').index(_continuous_channel(plain_encoder))
    hr_position = delta_encoder.continuous_value_fields().index(hr)
    normalizer._means[hr_position] = 70.0
    normalizer._stds[hr_position] = 10.0

    plain_normalized = normalizer.transform(plain)
    delta_normalized = normalizer.transform(delta)
    mask_names = [name for name in plain_header.split(',') if name.startswith('mask->')]
    mask_indices = [plain_header.split(',').index(name) for name in mask_names]
    delta_i = _time_gap_column(delta_header)

    assert np.allclose(delta_normalized[:, :plain.shape[1]], plain_normalized)
    assert np.allclose(delta_normalized[:, delta_i], [0.0, 1.0])
    assert np.array_equal(delta_normalized[:, mask_indices], delta[:, mask_indices])
    assert plain_normalized[0, hr] == 0.0
    assert plain_normalized[1, hr] == 2.0


def test_dense_and_sparse_raw_inputs_share_dimensions_and_default_is_unchanged():
    dense, header = _raw_rows([0.0, 4.0, 8.0], [0.0, 4.0, 8.0])
    sparse, _ = _raw_rows([0.0, 8.0], [0.0, 8.0])
    plain = RawSequenceEncoder()
    delta = RawSequenceEncoder(include_time_deltas=True)
    plain_dense, _ = plain.transform(dense, header=header, end=24.0)
    plain_sparse, _ = plain.transform(sparse, header=header, end=24.0)
    delta_dense, _ = delta.transform(dense, header=header, end=24.0)
    delta_sparse, _ = delta.transform(sparse, header=header, end=24.0)

    assert plain.feature_dim == 76
    assert delta.feature_dim == plain.feature_dim + 1
    assert plain_dense.shape[1] == plain_sparse.shape[1] == 76
    assert delta_dense.shape[1] == delta_sparse.shape[1] == 77


def test_timestamp_suffix_and_raw_only_validation_are_wired_into_runner():
    source = inspect.getsource(fixed_main.main)
    assert 'timestamp_suffix = ".td" if raw_mode and args.use_raw_timestamps else ""' in source
    assert 'args_dict[\'input_dim\'] = len(feature_header)' in source
    assert source.count('timestamp_suffix') >= 2

    completed = subprocess.run(
        [sys.executable, '-m', 'mimic4models.fixed_horizon_icu_exit.main',
         '--print_stats', '--data', os.path.join(ROOT, 'missing-data'),
         '--timestep', '1', '--use_raw_timestamps'],
        cwd=ROOT, env=dict(os.environ, PYTHONPATH=ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert completed.returncode != 0
    assert b'--use_raw_timestamps requires --timestep 0' in completed.stderr


def test_shared_cli_flag_defaults_off():
    parser = argparse.ArgumentParser()
    common_utils.add_common_arguments(parser)
    default_args = parser.parse_args(['--network', 'raw_lstm.py'])
    enabled_args = parser.parse_args([
        '--network', 'raw_lstm.py', '--use_raw_timestamps'])
    assert default_args.use_raw_timestamps is False
    assert enabled_args.use_raw_timestamps is True


if __name__ == '__main__':
    for _name in sorted(name for name in globals() if name.startswith('test_')):
        globals()[_name]()
    print('raw timestamp LSTM preprocessing tests passed')
