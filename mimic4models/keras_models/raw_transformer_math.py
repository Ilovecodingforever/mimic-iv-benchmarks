from __future__ import absolute_import

import math

import numpy as np


def sinusoidal_position_encoding(positions, d_model=16):
    """Return fixed interleaved sin/cos encodings for scalar positions."""
    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim == 0:
        positions = positions.reshape((1,))
    inverse = np.exp(np.arange(0, d_model, 2, dtype=np.float32) *
                     (-math.log(10000.0) / float(d_model)))
    angles = positions[..., None] * inverse
    encoding = np.zeros(positions.shape + (d_model,), dtype=np.float32)
    encoding[..., 0::2] = np.sin(angles)
    encoding[..., 1::2] = np.cos(angles)
    return encoding
