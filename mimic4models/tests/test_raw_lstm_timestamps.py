from __future__ import absolute_import

import argparse

import numpy as np

from mimic4models import common_utils
from mimic4models.fixed_horizon_icu_exit.raw import RawSequenceEncoder


def _raw_rows(times):
    return np.asarray([[str(t), ''] for t in times], dtype=object)


def test_raw_timestamp_flag_defaults_off_and_can_be_enabled():
    parser = argparse.ArgumentParser()
    common_utils.add_common_arguments(parser)

    default_args = parser.parse_args(['--network', 'raw_lstm.py'])
    enabled_args = parser.parse_args([
        '--network', 'raw_lstm.py', '--use_raw_timestamps'])

    assert default_args.use_raw_timestamps is False
    assert enabled_args.use_raw_timestamps is True


def test_raw_lstm_time_delta_feature_is_days_and_resets_per_sequence():
    rows = _raw_rows([0.25, 1.75, 24.0])
    default_encoder = RawSequenceEncoder()
    default_encoded, default_header = default_encoder.transform(
        rows, header=['Hours'] + default_encoder._id_to_channel)
    assert default_encoded.shape[1] == default_encoder.feature_dim
    assert 'TimeSincePreviousInputDays' not in default_header.split(',')

    encoder = RawSequenceEncoder(include_time_deltas=True)
    encoded, header = encoder.transform(
        rows, header=['Hours'] + encoder._id_to_channel)

    assert header.split(',')[-1] == 'TimeSincePreviousInputDays'
    assert encoded.shape[1] == RawSequenceEncoder().feature_dim + 1
    assert np.allclose(
        encoded[:, -1], [0.0, 1.5 / 24.0, 22.25 / 24.0])

    repeated, _ = encoder.transform(
        _raw_rows([3.0, 3.0]), header=['Hours'] + encoder._id_to_channel)
    restarted, _ = encoder.transform(
        _raw_rows([8.0]), header=['Hours'] + encoder._id_to_channel)
    assert np.allclose(repeated[:, -1], [0.0, 0.0])
    assert restarted[0, -1] == 0.0


def test_raw_hours_mode_remains_separate_from_time_delta_mode():
    rows = _raw_rows([0.25, 1.75])
    encoder = RawSequenceEncoder(include_timestamps=True)
    encoded, header = encoder.transform(
        rows, header=['Hours'] + encoder._id_to_channel)

    assert header.split(',')[-1] == 'Hours'
    assert np.allclose(encoded[:, -1], [0.25, 1.75])


def test_raw_encoder_rejects_both_time_modes():
    try:
        RawSequenceEncoder(include_timestamps=True, include_time_deltas=True)
    except ValueError:
        pass
    else:
        raise AssertionError('raw encoder accepted incompatible time modes')


if __name__ == '__main__':
    test_raw_timestamp_flag_defaults_off_and_can_be_enabled()
    test_raw_lstm_time_delta_feature_is_days_and_resets_per_sequence()
    test_raw_hours_mode_remains_separate_from_time_delta_mode()
    test_raw_encoder_rejects_both_time_modes()
    print('ok')
