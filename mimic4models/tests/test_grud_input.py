from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from mimic4models.preprocessing import Discretizer
from mimic4models.keras_models.grud import (
    elapsed_since_observed,
    split_grud_inputs,
    value_mask_mapping,
)


def discretizer_header():
    d = Discretizer(timestep=1.0, store_masks=True,
                    impute_strategy='previous', start_time='zero')
    return d.transform([['0'] + [''] * 17], end=24.0)[1].split(',')


def test_grud_input_shapes_and_mask_expansion():
    header = discretizer_header()
    value_indices, mask_indices, sources = value_mask_mapping(header)
    X = np.zeros((2, 24, len(header)), dtype='float32')

    hr_mask = header.index('mask->Heart Rate')
    gcs_eye_mask = header.index('mask->Glascow coma scale eye opening')
    X[:, ::2, hr_mask] = 1.0
    X[:, 1::3, gcs_eye_mask] = 1.0

    values, masks, timestamps = split_grud_inputs(X, header, timestep=1.0)
    assert values.shape == (2, 24, 59)
    assert masks.shape == values.shape
    assert timestamps.shape == (2, 24, 1)

    hr_value_pos = sources.index('Heart Rate')
    assert np.array_equal(masks[:, :, hr_value_pos], X[:, :, hr_mask])

    gcs_eye_positions = [i for i, source in enumerate(sources)
                         if source == 'Glascow coma scale eye opening']
    assert len(gcs_eye_positions) > 1
    for pos in gcs_eye_positions:
        assert np.array_equal(masks[:, :, pos], X[:, :, gcs_eye_mask])


def test_grud_elapsed_time_is_variable_specific():
    timestamps = np.arange(6, dtype='float32').reshape((1, 6, 1))
    masks = np.zeros((1, 6, 2), dtype='float32')
    masks[0, [0, 1, 5], 0] = 1.0
    masks[0, [0, 3], 1] = 1.0

    delta = elapsed_since_observed(timestamps, masks)
    assert delta[0, 5, 0] == 4.0
    assert delta[0, 5, 1] == 2.0
