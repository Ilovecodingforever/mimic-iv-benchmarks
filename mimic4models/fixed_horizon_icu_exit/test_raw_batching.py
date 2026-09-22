from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from mimic4models.fixed_horizon_icu_exit.raw import RawBatchSequence, pad_raw_batch


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
