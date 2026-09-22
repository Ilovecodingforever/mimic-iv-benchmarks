from __future__ import absolute_import
from __future__ import print_function

import os
import sys

import numpy as np

if __package__ is None or __package__ == '':
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from mimic4models.fixed_horizon_icu_exit.matched_count_control.sampling import apply_sampling_to_example
from mimic4models.preprocessing import Discretizer
from mimic4models.keras_models.grud import (
    Network,
    elapsed_since_observed,
    prepare_input,
    split_grud_inputs,
    value_mask_mapping,
)


def discretizer_header():
    d = Discretizer(timestep=1.0, store_masks=True,
                    impute_strategy='previous', start_time='zero')
    return d.transform([['0'] + [''] * 17], end=24.0)[1].split(',')


def assert_raises_value_error(fn, message_part):
    try:
        fn()
    except ValueError as exc:
        assert message_part in str(exc)
        return
    raise AssertionError('Expected ValueError containing {}'.format(message_part))


def test_grud_rejects_depth_greater_than_one():
    header = ['Heart Rate', 'mask->Heart Rate']
    assert_raises_value_error(
        lambda: Network(dim=2, batch_norm=False, dropout=0.0, rec_dropout=0.0,
                        task='ihm', depth=2, header=header),
        'Current GRU-D implementation supports only depth=1.')


def test_continuous_mask_mapping():
    header = ['Heart Rate', 'mask->Heart Rate']
    value_indices, mask_indices, sources = value_mask_mapping(header)
    assert value_indices == [0]
    assert mask_indices == [1]
    assert sources == ['Heart Rate']


def test_categorical_one_hot_mask_expansion():
    header = [
        'Glascow coma scale eye opening->To Pain',
        'Glascow coma scale eye opening->Spontaneously',
        'Glascow coma scale eye opening->No Response',
        'mask->Glascow coma scale eye opening',
    ]
    value_indices, mask_indices, sources = value_mask_mapping(header)
    assert value_indices == [0, 1, 2]
    assert mask_indices == [3, 3, 3]
    assert sources == ['Glascow coma scale eye opening'] * 3


def test_grud_input_shapes_and_mask_expansion():
    header = discretizer_header()
    _, _, sources = value_mask_mapping(header)
    X = np.zeros((2, 24, len(header)), dtype='float32')

    hr_mask = header.index('mask->Heart Rate')
    gcs_eye_mask = header.index('mask->Glascow coma scale eye opening')
    X[:, ::2, hr_mask] = 1.0
    X[:, 1::3, gcs_eye_mask] = 1.0

    values, masks, timestamps = split_grud_inputs(X, header, timestep=1.0)
    assert values.shape == (2, 24, 59)
    assert masks.shape == values.shape
    assert timestamps.shape == (2, 24, 1)
    assert np.all((masks == 0.0) | (masks == 1.0))

    hr_value_pos = sources.index('Heart Rate')
    assert np.array_equal(masks[:, :, hr_value_pos], X[:, :, hr_mask])

    gcs_eye_positions = [i for i, source in enumerate(sources)
                         if source == 'Glascow coma scale eye opening']
    assert len(gcs_eye_positions) > 1
    for pos in gcs_eye_positions:
        assert np.array_equal(masks[:, :, pos], X[:, :, gcs_eye_mask])


def test_grud_elapsed_time_sequences():
    timestamps = np.arange(6, dtype='float32').reshape((1, 6, 1))
    masks = np.zeros((1, 6, 2), dtype='float32')
    masks[0, :, 0] = np.asarray([1, 0, 0, 0, 0, 1], dtype='float32')
    masks[0, :, 1] = np.asarray([0, 0, 1, 0, 0, 1], dtype='float32')

    delta = elapsed_since_observed(timestamps, masks)
    expected = np.asarray([
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 1],
        [4, 2],
        [5, 3],
    ], dtype='float32')
    assert np.array_equal(delta[0], expected)
    return delta[0, :, 0], delta[0, :, 1]


def test_previous_imputed_values_are_ignored_when_mask_zero():
    np.random.seed(123)
    header = ['Heart Rate', 'mask->Heart Rate']
    X1 = np.zeros((2, 4, 2), dtype='float32')
    X1[:, :, 0] = np.asarray([1.0, 10.0, 20.0, 2.0], dtype='float32')
    X1[:, :, 1] = np.asarray([1.0, 0.0, 0.0, 1.0], dtype='float32')
    X2 = X1.copy()
    X2[:, 1, 0] = -100.0
    X2[:, 2, 0] = 500.0

    model = Network(dim=3, batch_norm=False, dropout=0.0, rec_dropout=0.0,
                    task='ihm', header=header)
    pred1 = model.predict(prepare_input(X1, header, timestep=1.0), batch_size=2, verbose=0)
    pred2 = model.predict(prepare_input(X2, header, timestep=1.0), batch_size=2, verbose=0)
    assert np.allclose(pred1, pred2, atol=0.0, rtol=0.0)
    return float(np.max(np.abs(pred1 - pred2)))


def structured_r8_diagnostic():
    raw_header = ['Hours', 'Heart Rate', 'Temperature']
    rows = []
    for hour in range(12):
        rows.append([str(float(hour)), str(80 + hour), str(36.0 + 0.1 * hour)])
    example = {
        'X': np.asarray(rows, dtype=object),
        'header': raw_header,
        't': 12.0,
        'name': 'synthetic_episode.csv',
        'y': 0,
    }
    thinned = apply_sampling_to_example(example, 'structured', 8)
    discretizer = Discretizer(timestep=1.0, store_masks=True,
                              impute_strategy='previous', start_time='zero')
    dense_grid, dense_header = discretizer.transform(example['X'], header=raw_header, end=example['t'])
    thin_grid, thin_header = discretizer.transform(thinned['X'], header=raw_header, end=example['t'])
    dense_header = dense_header.split(',')
    thin_header = thin_header.split(',')
    assert dense_header == thin_header
    assert dense_grid.shape == thin_grid.shape

    dense_values, dense_masks, dense_timestamps = split_grud_inputs(dense_grid[None, :, :], dense_header, 1.0)
    thin_values, thin_masks, thin_timestamps = split_grud_inputs(thin_grid[None, :, :], thin_header, 1.0)
    del dense_values, thin_values
    assert dense_timestamps.shape == thin_timestamps.shape == (1, dense_grid.shape[0], 1)
    assert np.array_equal(dense_timestamps, thin_timestamps)
    assert np.array_equal(dense_timestamps[0, :, 0], np.arange(dense_grid.shape[0], dtype='float32'))

    _, _, sources = value_mask_mapping(dense_header)
    hr_pos = sources.index('Heart Rate')
    dense_hr_mask = dense_masks[0, :, hr_pos]
    thin_hr_mask = thin_masks[0, :, hr_pos]
    assert dense_hr_mask.sum() > thin_hr_mask.sum()

    dense_delta = elapsed_since_observed(dense_timestamps, dense_masks)[0, :, hr_pos]
    thin_delta = elapsed_since_observed(thin_timestamps, thin_masks)[0, :, hr_pos]
    assert thin_delta.max() > dense_delta.max()
    return {
        'dense_shape': dense_grid.shape,
        'structured_shape': thin_grid.shape,
        'timestamps': thin_timestamps[0, :, 0],
        'dense_hr_mask': dense_hr_mask,
        'structured_hr_mask': thin_hr_mask,
        'dense_hr_elapsed': dense_delta,
        'structured_hr_elapsed': thin_delta,
    }


def run_all():
    test_grud_rejects_depth_greater_than_one()
    print('depth validation: ok')
    test_continuous_mask_mapping()
    print('continuous mask mapping: ok')
    test_categorical_one_hot_mask_expansion()
    print('categorical one-hot mask expansion: ok')
    test_grud_input_shapes_and_mask_expansion()
    print('shape and binary-mask check: ok')
    elapsed_a, elapsed_b = test_grud_elapsed_time_sequences()
    print('elapsed feature A:', elapsed_a.astype(int).tolist())
    print('elapsed feature B:', elapsed_b.astype(int).tolist())
    max_diff = test_previous_imputed_values_are_ignored_when_mask_zero()
    print('missing-position value perturbation max abs diff:', max_diff)
    diag = structured_r8_diagnostic()
    print('dense grid shape:', diag['dense_shape'])
    print('structured-r8 grid shape:', diag['structured_shape'])
    print('structured-r8 timestamps:', diag['timestamps'].astype(int).tolist())
    print('dense Heart Rate mask:', diag['dense_hr_mask'].astype(int).tolist())
    print('structured-r8 Heart Rate mask:', diag['structured_hr_mask'].astype(int).tolist())
    print('dense Heart Rate elapsed:', diag['dense_hr_elapsed'].astype(int).tolist())
    print('structured-r8 Heart Rate elapsed:', diag['structured_hr_elapsed'].astype(int).tolist())
    print('all GRU-D input sanity checks passed')


if __name__ == '__main__':
    run_all()
