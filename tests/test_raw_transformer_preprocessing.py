from __future__ import absolute_import

import argparse
import inspect
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mimic4models import common_utils
from mimic4models.fixed_horizon_icu_exit import main as fixed_main
from mimic4models.fixed_horizon_icu_exit.matched_count_control.sampling import (
    random_matched_sample, structured_sample)
from mimic4models.fixed_horizon_icu_exit.raw import RawSequenceEncoder
from mimic4models.keras_models.raw_transformer_math import sinusoidal_position_encoding


def _rows(times, values):
    encoder = RawSequenceEncoder()
    header = ['Hours'] + encoder._id_to_channel
    channel = next(name for name in encoder._id_to_channel
                   if not encoder._is_categorical_channel[name])
    channel_i = header.index(channel)
    rows = []
    for time, value in zip(times, values):
        row = [''] * len(header)
        row[0] = str(time)
        row[channel_i] = str(value)
        rows.append(row)
    return np.asarray(rows, dtype=object), header, channel


def test_timestamp_and_index_sinusoidal_values():
    encoded = sinusoidal_position_encoding([0.0, 1.0], d_model=4)
    assert np.allclose(encoded[0], [0.0, 1.0, 0.0, 1.0])
    assert np.allclose(encoded[1], [np.sin(1.0), np.cos(1.0),
                                    np.sin(1.0 / 100.0), np.cos(1.0 / 100.0)])

    timestamp = sinusoidal_position_encoding([0.0, 2.0], d_model=16)
    index = sinusoidal_position_encoding([0.0, 1.0], d_model=16)
    assert np.array_equal(timestamp[0], index[0])
    assert not np.array_equal(timestamp[1], index[1])


def test_sampling_preserves_original_timestamps_for_dense_and_sparse_inputs():
    dense, header, _ = _rows(
        [0.5, 1.0, 3.0, 3.5, 4.1, 5.0, 7.0, 8.0],
        [501, 999, 503, 535, 541, 550, 570, 580])
    sparse, _, _ = _rows([3.5, 7.0, 8.0], [535, 570, 580])

    dense_sampled, _ = structured_sample(dense, header, 4, end=24.0)
    sparse_sampled, _ = structured_sample(sparse, header, 4, end=24.0)
    assert np.array_equal(dense_sampled, sparse_sampled)
    assert list(dense_sampled[:, 0]) == ['3.5', '8.0']


def test_random_sampling_has_no_removed_measurement_or_timestamp_leakage():
    rows, header, channel = _rows(
        [0.5, 1.0, 3.0, 3.5, 4.1, 5.0, 7.0, 8.0, 12.0],
        [501, 999, 503, 535, 541, 550, 570, 580, 512])
    sampled, _ = random_matched_sample(
        rows, header, 4, sampling_seed=100, stay_name='synthetic', end=24.0)

    assert all(float(time) in set(float(x) for x in rows[:, 0])
               for time in sampled[:, 0])
    assert '541' not in set(sampled[:, header.index(channel)])


def test_raw_timestamp_and_index_representations_have_expected_dimensions():
    encoder = RawSequenceEncoder()
    timestamp_encoder = RawSequenceEncoder(include_timestamps=True)
    assert timestamp_encoder.feature_dim == encoder.feature_dim + 1
    assert timestamp_encoder.header()[:-1] == encoder.header()
    assert timestamp_encoder.header()[-1] == 'Hours'


def test_shared_cli_defaults_and_positional_encoding_choices():
    parser = argparse.ArgumentParser()
    common_utils.add_common_arguments(parser)
    default_args = parser.parse_args(['--network', 'raw_transformer.py'])
    index_args = parser.parse_args([
        '--network', 'raw_transformer.py',
        '--transformer_positional_encoding', 'index'])
    assert default_args.transformer_positional_encoding == 'timestamp'
    assert index_args.transformer_positional_encoding == 'index'


def test_fixed_horizon_runner_has_transformer_mode_validation():
    source = inspect.getsource(fixed_main.main)
    assert 'IS_RAW_TRANSFORMER' in source
    assert 'raw Transformer requires --timestep 0' in source
    assert '--use_raw_timestamps is for raw LSTM inputs' in source


if __name__ == '__main__':
    for name in sorted(name for name in globals() if name.startswith('test_')):
        globals()[name]()
    print('raw Transformer preprocessing tests passed')
