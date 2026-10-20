from __future__ import absolute_import

import numpy as np

from mimic4models.fixed_horizon_icu_exit.raw import (
    RawBatchSequence, RawSequenceEncoder, pad_raw_batch)
from mimic4models.length_of_stay.utils import RawBatchGen


def _raw_rows(encoder):
    header = ['Hours'] + encoder._id_to_channel
    channel_id = [
        i for i, channel in enumerate(encoder._id_to_channel)
        if not encoder._is_categorical_channel[channel]
    ][0]
    rows = []
    for hour, value in [(0.25, 70), (1.75, 71), (24.0, 72), (24.25, 73)]:
        row = [''] * len(header)
        row[0] = str(hour)
        row[channel_id + 1] = str(value)
        rows.append(row)
    return np.asarray(rows), header


def test_raw_los_encoder_truncates_at_prediction_time_and_preserves_irregular_time():
    encoder = RawSequenceEncoder(include_timestamps=True)
    rows, header = _raw_rows(encoder)
    encoded, _ = encoder.transform(rows, header=header, end=24.0)

    assert encoded.dtype == np.float32
    assert encoded.shape[0] == 3
    assert np.allclose(encoded[:, -1], [0.25, 1.75, 24.0])


def test_raw_los_batch_gen_preserves_original_labels_for_metrics():
    sequences = [
        np.ones((1, 76), dtype=np.float32),
        np.ones((3, 76), dtype=np.float32),
        np.ones((2, 76), dtype=np.float32),
    ]
    labels = np.asarray([24.0, 48.0, 72.0])
    batch_gen = RawBatchGen(
        sequences, labels, partition='custom', batch_size=2,
        steps=2, shuffle=False)

    x, processed, original = batch_gen.next(return_y_true=True)
    assert x.shape == (2, 2, 76)
    assert processed.dtype.kind in ('i', 'u')
    assert np.array_equal(original, np.asarray([24.0, 72.0]))

    _, processed_none, original_none = RawBatchGen(
        sequences, labels, partition='none', batch_size=2,
        steps=2, shuffle=False).next(return_y_true=True)
    assert processed_none.dtype.kind == 'f'
    assert np.array_equal(original_none, np.asarray([24.0, 72.0]))


def test_raw_los_test_batches_restore_original_order():
    sequences = [
        np.ones((5, 76), dtype=np.float32) * 1.0,
        np.ones((1, 76), dtype=np.float32) * 2.0,
        np.ones((3, 76), dtype=np.float32) * 3.0,
    ]
    labels = np.asarray([10.0, 20.0, 30.0])
    batches = RawBatchSequence(
        sequences, labels, batch_size=2, shuffle=False,
        bucket_size=200, target_repl=False)
    predictions = [None] * len(sequences)
    for batch_i in range(len(batches)):
        x_batch, _ = batches[batch_i]
        batch_predictions = np.sum(x_batch, axis=(1, 2))
        for index, prediction in zip(
                batches.batch_indices(batch_i), batch_predictions):
            predictions[index] = prediction

    assert len(predictions) == len(sequences)
    assert predictions[0] == np.sum(sequences[0])
    assert predictions[1] == np.sum(sequences[1])
    assert predictions[2] == np.sum(sequences[2])


def test_raw_los_padding_is_batch_local():
    sequences = [
        np.ones((2, 76), dtype=np.float32),
        np.ones((5, 76), dtype=np.float32),
    ]
    padded = pad_raw_batch(sequences)
    assert padded.shape == (2, 5, 76)
    assert np.all(padded[0, 2:] == 0.0)
    assert np.all(padded[1] == 1.0)


if __name__ == '__main__':
    test_raw_los_encoder_truncates_at_prediction_time_and_preserves_irregular_time()
    test_raw_los_batch_gen_preserves_original_labels_for_metrics()
    test_raw_los_test_batches_restore_original_order()
    test_raw_los_padding_is_batch_local()
    print('ok')
