from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from mimic4models.fixed_horizon_icu_exit import main as fixed_main
from mimic4models.fixed_horizon_icu_exit.raw import (
    RawBatchSequence, RawObservedNormalizer, RawSequenceEncoder, pad_raw_batch)


def _seq(length, value):
    return np.ones((length, 76), dtype=np.float32) * value


def test_pad_raw_batch_uses_batch_local_post_padding():
    batch = pad_raw_batch([_seq(7, 1.0), _seq(8, 2.0), _seq(10, 3.0), _seq(11, 4.0)])

    assert batch.shape == (4, 11, 76)
    assert np.all(batch[0, :7] == 1.0)
    assert np.all(batch[0, 7:] == 0.0)
    assert np.all(batch[3] == 4.0)


def test_raw_batch_sequence_sorts_lengths_without_global_padding():
    sequences = [_seq(10, 10.0), _seq(1, 1.0), _seq(8, 8.0), _seq(2, 2.0), _seq(7, 7.0)]
    labels = np.arange(len(sequences))

    batches = RawBatchSequence(sequences, labels, batch_size=2, shuffle=False)

    x0, y0 = batches[0]
    x1, y1 = batches[1]
    x2, y2 = batches[2]

    assert x0.shape == (2, 2, 76)
    assert list(y0) == [1, 3]
    assert x1.shape == (2, 8, 76)
    assert list(y1) == [4, 2]
    assert x2.shape == (1, 10, 76)
    assert list(y2) == [0]


def test_raw_batch_sequence_forms_batches_before_shuffle():
    sequences = [_seq(1, 1.0), _seq(100, 100.0), _seq(2, 2.0), _seq(99, 99.0)]
    labels = np.arange(len(sequences))

    batches = RawBatchSequence(sequences, labels, batch_size=2, shuffle=True, seed=7)

    batch_lengths = []
    for batch_i in range(len(batches)):
        lengths = sorted([sequences[i].shape[0] for i in batches.batch_indices(batch_i)])
        batch_lengths.append(lengths)
    assert sorted(batch_lengths) == [[1, 2], [99, 100]]


def test_raw_float32_from_encoder_and_normalizer():
    encoder = RawSequenceEncoder()
    header = ['Hours'] + encoder._id_to_channel
    channel_id = [
        i for i, channel in enumerate(encoder._id_to_channel)
        if not encoder._is_categorical_channel[channel]
    ][0]
    row = ['0.1'] + [''] * len(encoder._id_to_channel)
    row[channel_id + 1] = '70'
    encoded, _ = encoder.transform(np.array([row]), header=header, end=24.0)
    assert encoded.dtype == np.float32

    normalizer = RawObservedNormalizer(
        [encoder._begin_pos[channel_id]], [encoder.value_dim + channel_id])
    normalizer._means = np.array([10.0])
    normalizer._stds = np.array([2.0])
    normalized = normalizer.transform(encoded.astype(np.float64))
    assert normalized.dtype == np.float32


class _FakeModel(object):
    def predict(self, x, batch_size=None, verbose=0):
        return np.sum(x, axis=(1, 2)).reshape((-1, 1))


def test_raw_global_vs_local_prediction_check_and_ordering():
    sequences = [_seq(5, 1.0), _seq(1, 2.0), _seq(3, 3.0), _seq(2, 4.0)]
    labels = np.arange(len(sequences))

    predictions = fixed_main.predict_raw_batches(_FakeModel(), sequences, labels, batch_size=2)
    expected = np.array([np.sum(seq) for seq in sequences])
    assert np.allclose(predictions, expected)

    max_diff = fixed_main.check_raw_padding_prediction_equivalence(
        _FakeModel(), sequences, labels, batch_size=2, max_examples=4)
    assert max_diff == 0.0


def test_padding_efficiency_uses_batch_local_lengths():
    sequences = [_seq(1, 1.0), _seq(2, 2.0), _seq(9, 9.0), _seq(10, 10.0)]
    labels = np.arange(len(sequences))
    batches = RawBatchSequence(sequences, labels, batch_size=2, shuffle=False)

    eff = batches.padding_efficiency()

    assert eff['real_steps'] == 22.0
    assert eff['allocated_steps'] == 24.0
    assert abs(eff['padding_efficiency'] - (22.0 / 24.0)) < 1e-12
    assert eff['mean_batch_max_length'] == 6.0
