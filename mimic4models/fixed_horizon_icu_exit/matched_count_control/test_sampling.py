from __future__ import absolute_import
from __future__ import print_function

import inspect
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mimic4models.preprocessing import Discretizer, discretizer_bin_id
from mimic4models.create_normalizer_state import validate_normalizer_args
from mimic4models.fixed_horizon_icu_exit import main as fixed_main
from mimic4models.fixed_horizon_icu_exit.raw import RawObservedNormalizer, RawSequenceEncoder
from mimic4models.fixed_horizon_icu_exit.matched_count_control.validate_sampling import (
    structured_vs_coarse_value_mismatches)
from mimic4models.fixed_horizon_icu_exit.matched_count_control.sampling import (
    apply_sampling_to_example, cell_counts_by_channel, choose_matched_candidates,
    mask_counts_by_channel, nonempty_cell_count, random_matched_cells, random_matched_sample,
    reconstruct_sparse_raw, select_last_raw_observation_per_bin, structured_cells,
    structured_sample)


HEADER = ['Hours', 'Heart Rate', 'Glucose', 'pH']
CAT_HEADER = ['Hours', 'Capillary refill rate']


def a(rows, header=HEADER):
    return np.array(rows, dtype=object)


def values(cells, channel):
    return [cell['value'] for cell in cells if cell['channel'] == channel]


def times(cells, channel):
    return [cell['time'] for cell in cells if cell['channel'] == channel]


def transformed(X, timestep=1.0, header=HEADER, impute='previous'):
    d = Discretizer(timestep=timestep, store_masks=True,
                    impute_strategy=impute, start_time='zero')
    return d.transform(X, header=header, end=24.0)


def col(header, name):
    return header.split(',').index(name)


def test_multiple_observations_inside_one_coarse_bin():
    X = a([['0.5', '70', '', ''], ['1.2', '71', '', ''], ['3.8', '72', '', '']])
    cells = structured_cells(X, HEADER, 4)
    assert values(cells, 'Heart Rate') == ['72']
    data, h = transformed(X, 4, HEADER, impute='zero')
    assert data[0, col(h, 'Heart Rate')] == 72.0


def test_boundaries_use_current_discretizer_semantics():
    rows = [[str(t), str(int(t)), '', ''] for t in [0, 4, 8, 12, 16, 20, 24]]
    X = a(rows)
    selected = select_last_raw_observation_per_bin(X, HEADER, 4, end=24.0)
    expected = {}
    for row_id, t in enumerate([0, 4, 8, 12, 16, 20, 24]):
        expected[discretizer_bin_id(float(t), 4.0)] = str(int(t))
    assert dict((key[0], cell['value']) for key, cell in selected.items()) == expected
    assert [discretizer_bin_id(float(t), 4.0) for t in [0, 1, 2, 4, 8, 12, 24]] == [0, 0, 0, 0, 1, 2, 5]


def test_equal_timestamps_same_variable_last_row_wins():
    X = a([['3.0', '70', '', ''], ['3.0', '75', '', '']])
    assert values(structured_cells(X, HEADER, 4), 'Heart Rate') == ['75']


def test_same_timestamp_different_variables_survive_independently():
    X = a([['3.0', '70', '', ''], ['3.0', '', '120', '']])
    sparse, cells = structured_sample(X, HEADER, 4)
    assert values(cells, 'Heart Rate') == ['70']
    assert values(cells, 'Glucose') == ['120']
    data, h = transformed(sparse, 1, HEADER)
    counts = mask_counts_by_channel(data, h)
    assert counts['Heart Rate'] == 1
    assert counts['Glucose'] == 1


def test_counts_are_per_variable_not_global():
    X = a([['1', '1', '', ''], ['2', '2', '20', ''], ['3', '3', '', ''],
           ['5', '5', '', ''], ['6', '', '', '7.1'], ['7', '7', '', ''],
           ['9', '9', '', ''], ['10', '', '100', '']])
    b, _ = structured_sample(X, HEADER, 4)
    c, _ = random_matched_sample(X, HEADER, 4, sampling_seed=100, stay_name='stay')
    assert cell_counts_by_channel(b, HEADER) == {'Heart Rate': 3, 'Glucose': 2, 'pH': 1}
    assert cell_counts_by_channel(c, HEADER) == cell_counts_by_channel(b, HEADER)


def test_empty_bins_and_never_observed_variables():
    X = a([['1.0', '70', '', ''], ['13.0', '90', '', '']])
    b, _ = structured_sample(X, HEADER, 4)
    c, _ = random_matched_sample(X, HEADER, 4, sampling_seed=100, stay_name='stay')
    assert cell_counts_by_channel(b, HEADER)['Heart Rate'] == 2
    assert cell_counts_by_channel(b, HEADER)['pH'] == 0
    assert cell_counts_by_channel(c, HEADER)['pH'] == 0


def test_observations_after_prediction_window_are_ignored():
    X = a([['23.9', '70', '', ''], ['24.0', '80', '', ''], ['24.1', '90', '', '']])
    cells = structured_cells(X, HEADER, 4, end=24.0)
    assert values(cells, 'Heart Rate') == ['80']
    assert times(cells, 'Heart Rate') == [24.0]


def test_random_candidates_are_occupied_1h_representatives():
    X = a([['2.1', '70', '', ''], ['2.4', '72', '', ''], ['2.9', '75', '', '']])
    fine = select_last_raw_observation_per_bin(X, HEADER, 1.0, end=24.0)
    assert list(fine.values())[0]['time'] == 2.9
    assert list(fine.values())[0]['value'] == '75'


def test_structured_selected_cells_remain_distinct_after_1h_discretization():
    X = a([['4.0', '64', '', ''], ['4.1', '65', '', ''], ['8.0', '68', '', '']])
    b, _ = structured_sample(X, HEADER, 4)
    data, h = transformed(b, 1, HEADER)
    assert mask_counts_by_channel(data, h)['Heart Rate'] == cell_counts_by_channel(b, HEADER)['Heart Rate']


def test_random_matched_count_after_1h_discretization():
    X = a([['1', '1', '', ''], ['2', '2', '', ''], ['5', '5', '', ''],
           ['6', '6', '', ''], ['9', '9', '', ''], ['10', '10', '', '']])
    b, _ = structured_sample(X, HEADER, 4)
    c, _ = random_matched_sample(X, HEADER, 4, sampling_seed=100, stay_name='stay')
    bd, bh = transformed(b, 1, HEADER)
    cd, ch = transformed(c, 1, HEADER)
    assert mask_counts_by_channel(bd, bh)['Heart Rate'] == 3
    assert mask_counts_by_channel(cd, ch)['Heart Rate'] == 3


def test_model_visible_1h_observed_cells_can_be_less_than_raw_obs():
    X = a([['2.1', '70', '', ''], ['2.4', '72', '', ''], ['2.9', '75', '', ''],
           ['3.5', '77', '', ''], ['5.1', '80', '', ''], ['9.1', '90', '', '']])
    b, _ = structured_sample(X, HEADER, 4)
    c, _ = random_matched_sample(X, HEADER, 4, sampling_seed=100, stay_name='stay')

    ax, ah = transformed(X, 1, HEADER)
    bx, bh = transformed(b, 1, HEADER)
    cx, ch = transformed(c, 1, HEADER)
    raw_obs = nonempty_cell_count(X)
    a_1h = sum(mask_counts_by_channel(ax, ah).values())
    b_1h = sum(mask_counts_by_channel(bx, bh).values())
    c_1h = sum(mask_counts_by_channel(cx, ch).values())

    assert raw_obs == 6
    assert raw_obs > a_1h
    assert a_1h >= b_1h
    assert b_1h == c_1h


def test_deterministic_sampling_and_seed_count_invariance():
    X = a([['1', '1', '', ''], ['2', '2', '', ''], ['5', '5', '', ''],
           ['6', '6', '', ''], ['9', '9', '', ''], ['10', '10', '', '']])
    c1, cells1 = random_matched_sample(X, HEADER, 4, sampling_seed=100, stay_name='stay')
    c2, cells2 = random_matched_sample(X, HEADER, 4, sampling_seed=100, stay_name='stay')
    c3, _ = random_matched_sample(X, HEADER, 4, sampling_seed=101, stay_name='stay')
    assert np.array_equal(c1, c2)
    assert [(x['row_id'], x['col_id']) for x in cells1] == [(x['row_id'], x['col_id']) for x in cells2]
    assert cell_counts_by_channel(c1, HEADER) == cell_counts_by_channel(c3, HEADER)


def test_k_equals_all_candidates_and_impossible_k_message():
    X = a([['1', '1', '', ''], ['5', '5', '', '']])
    cells = random_matched_cells(X, HEADER, 4, sampling_seed=100, stay_name='stay')
    assert values(cells, 'Heart Rate') == ['1', '5']
    try:
        choose_matched_candidates([{'row_id': 0}], 2, 100, 'stay_x', 4, 'Heart Rate')
    except ValueError as exc:
        msg = str(exc)
        assert 'stay_x' in msg and 'Heart Rate' in msg and 'K=2' in msg and 'candidates=1' in msg and 'interval=4' in msg
    else:
        raise AssertionError('impossible K did not fail')


def test_categorical_variable_selection():
    X = a([['1', '0.0'], ['2', '1.0']], CAT_HEADER)
    cells = structured_cells(X, CAT_HEADER, 4)
    assert values(cells, 'Capillary refill rate') == ['1.0']
    data, h = transformed(X, 4, CAT_HEADER, impute='zero')
    assert data[0, col(h, 'Capillary refill rate->1.0')] == 1.0


def test_structured_vs_coarse_value_equivalence_helper_continuous_and_categorical():
    continuous = {'X': a([['0.5', '70', '', ''], ['1.2', '71', '100', ''],
                          ['3.8', '72', '', ''], ['4.1', '73', '', '7.2']]),
                  'header': HEADER, 't': 24.0, 'y': 0, 'name': 'toy'}
    assert structured_vs_coarse_value_mismatches(continuous, 4, 'zero') == 0

    categorical = {'X': a([['1', '0.0'], ['2', '1.0']], CAT_HEADER),
                   'header': CAT_HEADER, 't': 24.0, 'y': 0, 'name': 'toy_cat'}
    assert structured_vs_coarse_value_mismatches(categorical, 4, 'zero') == 0


def test_mixed_measurements_select_individual_cells():
    X = a([['2.0', '80', '120', ''], ['3.0', '', '130', '']])
    sparse, _ = structured_sample(X, HEADER, 4)
    assert sparse.shape == (2, 4)
    assert list(sparse[0]) == ['2.0', '80', '', '']
    assert list(sparse[1]) == ['3.0', '', '130', '']


def test_structured_selector_matches_coarse_discretizer_values():
    X = a([['0.5', '70', '', ''], ['1.2', '71', '100', ''], ['3.8', '72', '', ''],
           ['4.1', '73', '', '7.2'], ['8.0', '', '110', '7.3'], ['12.0', '90', '', '']])
    for r in [2, 4, 8]:
        data, h = transformed(X, r, HEADER, impute='zero')
        for (_, col_id), cell in select_last_raw_observation_per_bin(X, HEADER, r, end=24.0).items():
            assert data[cell['bin_id'], col(h, HEADER[col_id])] == float(cell['value'])


def test_standard_discretizer_outputs_are_unchanged_by_helper_refactor():
    X = a([['0.0', '60', '', ''], ['1.1', '61', '', ''], ['4.0', '64', '', '']])
    d1, h1 = transformed(X, 1, HEADER, impute='zero')
    assert d1.shape[0] == 24
    assert d1[0, col(h1, 'Heart Rate')] == 60.0
    assert d1[1, col(h1, 'Heart Rate')] == 61.0
    d4, h4 = transformed(X, 4, HEADER, impute='zero')
    assert d4.shape[0] == 6
    assert d4[0, col(h4, 'Heart Rate')] == 64.0


def test_sequence_lengths_for_conditions():
    X = a([['1', '70', '', ''], ['5', '80', '', '']])
    b, _ = structured_sample(X, HEADER, 4)
    c, _ = random_matched_sample(X, HEADER, 4, sampling_seed=100, stay_name='stay')
    assert transformed(X, 1, HEADER)[0].shape[0] == 24
    assert transformed(b, 1, HEADER)[0].shape[0] == 24
    assert transformed(c, 1, HEADER)[0].shape[0] == 24
    assert transformed(X, 2, HEADER)[0].shape[0] == 12
    assert transformed(X, 4, HEADER)[0].shape[0] == 6
    assert transformed(X, 8, HEADER)[0].shape[0] == 3


def test_sampling_changes_only_X_metadata_unchanged():
    example = {'X': a([['1', '70', '', ''], ['5', '80', '', '']]),
               't': 24.0, 'y': 1, 'header': HEADER, 'name': 'stay'}
    sampled = apply_sampling_to_example(example, 'structured', 4, 100)
    assert sampled['name'] == example['name']
    assert sampled['t'] == example['t']
    assert sampled['y'] == example['y']
    assert sampled['header'] == example['header']
    assert sampled['X'] is not example['X']


def test_previous_imputation_happens_after_selection_and_masks_define_counts():
    X = a([['1', '70', '', ''], ['9', '90', '', '']])
    b, _ = structured_sample(X, HEADER, 4)
    assert cell_counts_by_channel(b, HEADER)['Heart Rate'] == 2
    data, h = transformed(b, 1, HEADER, impute='previous')
    assert data[5, col(h, 'Heart Rate')] == 70.0
    assert mask_counts_by_channel(data, h)['Heart Rate'] == 2

    one = a([['1', '70', '', '']])
    data, h = transformed(one, 1, HEADER, impute='previous')
    assert mask_counts_by_channel(data, h)['Heart Rate'] == 1
    assert np.sum(data[:, col(h, 'Heart Rate')] == 70.0) > 1


def test_fixed_horizon_normalizer_rejects_no_masks():
    class Args(object):
        pass
    args = Args()
    args.task = 'fixed_horizon_icu_exit'
    args.store_masks = False
    args.sampling_strategy = 'none'
    args.sampling_interval = 4.0
    args.timestep = 1.0
    args.start_time = 'zero'
    try:
        validate_normalizer_args(args)
    except ValueError as exc:
        assert 'require masks' in str(exc)
        assert 'masks:True' in str(exc)
    else:
        raise AssertionError('fixed_horizon_icu_exit --no-masks did not fail')



def make_sampling_args(sampling_strategy='none', sampling_interval=4.0, sampling_seed=100,
                       test_sampling_strategy=None, test_sampling_interval=None,
                       test_sampling_seed=None, timestep=1.0):
    class Args(object):
        pass
    args = Args()
    args.sampling_strategy = sampling_strategy
    args.sampling_interval = sampling_interval
    args.sampling_seed = sampling_seed
    args.test_sampling_strategy = test_sampling_strategy
    args.test_sampling_interval = test_sampling_interval
    args.test_sampling_seed = test_sampling_seed
    args.timestep = timestep
    return args


def test_test_sampling_args_omitted_inherit_training_regime():
    args = make_sampling_args('structured', 4.0, 123)
    assert fixed_main.resolve_test_sampling_args(args) == ('structured', 4.0, 123)
    assert fixed_main.validate_all_sampling_args(args) == ('structured', 4.0, 123)


def test_train_none_test_structured_r4_resolves_without_changing_training_regime():
    args = make_sampling_args('none', 4.0, 100,
                              test_sampling_strategy='structured',
                              test_sampling_interval=4.0)
    assert fixed_main.resolve_test_sampling_args(args) == ('structured', 4.0, 100)
    assert args.sampling_strategy == 'none'
    assert args.sampling_interval == 4.0
    assert args.sampling_seed == 100


def test_train_structured_r8_test_omitted_resolves_both_to_structured_r8():
    args = make_sampling_args('structured', 8.0, 100)
    assert fixed_main.resolve_test_sampling_args(args) == ('structured', 8.0, 100)
    assert fixed_main.sampling_regime_label(
        args.sampling_strategy, args.sampling_interval, args.sampling_seed) == 'structured-r8'


def test_normalizer_path_uses_training_sampling_args_only():
    args = make_sampling_args('none', 4.0, 100,
                              test_sampling_strategy='structured',
                              test_sampling_interval=8.0)
    assert fixed_main.resolve_test_sampling_args(args) == ('structured', 8.0, 100)
    path = fixed_main.default_normalizer_state_path(
        '/norm', 1.0, 'previous', 321, args.sampling_strategy,
        args.sampling_interval, args.sampling_seed)
    assert path == '/norm/fixed_horizon_icu_exit_ts:1.00_impute:previous_start:zero_masks:True_n:321.normalizer'
    assert 'sampling:structured' not in path


def test_test_prediction_filename_preserves_existing_name_for_matched_regimes():
    none = fixed_main.test_prediction_path('/out', '/states/model.epoch1.state',
                                           'none', 4.0, 100,
                                           'none', 8.0, 999)
    structured = fixed_main.test_prediction_path('/out', '/states/model.epoch1.state',
                                                 'structured', 4.0, 100,
                                                 'structured', 4.0, 999)
    random_matched = fixed_main.test_prediction_path('/out', '/states/model.epoch1.state',
                                                     'random_matched', 4.0, 777,
                                                     'random_matched', 4.0, 777)
    assert none.endswith('/test_predictions/model.epoch1.state.csv')
    assert structured.endswith('/test_predictions/model.epoch1.state.csv')
    assert random_matched.endswith('/test_predictions/model.epoch1.state.csv')
    assert fixed_main.sampling_regimes_differ('none', 4.0, 100, 'none', 8.0, 999) is False
    assert fixed_main.sampling_regimes_differ('structured', 4.0, 100, 'structured', 4.0, 999) is False


def test_shifted_test_prediction_filenames_include_test_regime_and_do_not_collide():
    matched = fixed_main.test_prediction_path('/out', '/states/model.epoch1.state',
                                              'none', 4.0, 100,
                                              'none', 4.0, 100)
    r4 = fixed_main.test_prediction_path('/out', '/states/model.epoch1.state',
                                         'none', 4.0, 100,
                                         'structured', 4.0, 100)
    r8 = fixed_main.test_prediction_path('/out', '/states/model.epoch1.state',
                                         'none', 4.0, 100,
                                         'structured', 8.0, 100)
    assert matched.endswith('/test_predictions/model.epoch1.state.csv')
    assert r4.endswith('/test_predictions/model.epoch1.state.testsample-structured-r4.csv')
    assert r8.endswith('/test_predictions/model.epoch1.state.testsample-structured-r8.csv')
    assert len(set([matched, r4, r8])) == 3


def test_random_matched_shifted_test_prediction_filename_includes_test_seed():
    path = fixed_main.test_prediction_path('/out', '/states/model.epoch1.state',
                                           'none', 4.0, 100,
                                           'random_matched', 4.0, 777)
    assert path.endswith('/test_predictions/model.epoch1.state.testsample-random_matched-r4-sseed777.csv')


def test_random_matched_seed_change_is_an_effective_shift():
    path = fixed_main.test_prediction_path('/out', '/states/model.epoch1.state',
                                           'random_matched', 4.0, 100,
                                           'random_matched', 4.0, 777)
    assert fixed_main.sampling_regimes_differ('random_matched', 4.0, 100,
                                              'random_matched', 4.0, 777) is True
    assert path.endswith('/test_predictions/model.epoch1.state.testsample-random_matched-r4-sseed777.csv')


def test_structured_and_random_sampling_allow_raw_or_1h_downstream_only():
    fixed_main.validate_all_sampling_args(make_sampling_args('structured', 4.0, timestep=1.0))
    fixed_main.validate_all_sampling_args(make_sampling_args('structured', 4.0, timestep=0.0))
    fixed_main.validate_all_sampling_args(make_sampling_args('random_matched', 4.0, timestep=0.0))

    args = make_sampling_args('none', 4.0, 100,
                              test_sampling_strategy='structured',
                              test_sampling_interval=4.0,
                              timestep=4.0)
    try:
        fixed_main.validate_all_sampling_args(args)
    except ValueError as exc:
        assert 'requires downstream timestep=0.0 or 1.0' in str(exc)
    else:
        raise AssertionError('invalid test sampling/timestep combination did not fail')



def test_raw_normalizer_path_is_distinct_from_gridded_path():
    dense = fixed_main.default_normalizer_state_path('/norm', 0.0, 'previous', 321,
                                                     'none', 4.0, 100)
    structured = fixed_main.default_normalizer_state_path('/norm', 0.0, 'previous', 321,
                                                          'structured', 4.0, 100)
    random_path = fixed_main.default_normalizer_state_path('/norm', 0.0, 'previous', 321,
                                                           'random_matched', 4.0, 777)
    assert dense == '/norm/fixed_horizon_icu_exit_raw_ts:0.00_observed_only_n:321.normalizer'
    assert structured == '/norm/fixed_horizon_icu_exit_raw_sampling:structured_r:4_ts:0.00_observed_only_n:321.normalizer'
    assert random_path == '/norm/fixed_horizon_icu_exit_raw_sampling:random_matched_r:4_seed:777_ts:0.00_observed_only_n:321.normalizer'


def test_raw_sequence_encoder_uses_rows_no_imputation_and_masks():
    encoder = RawSequenceEncoder()
    X = a([['0.13', '82', '', ''], ['0.27', '', '110', '']], HEADER)
    data, header = encoder.transform(X, header=HEADER, end=24.0)
    names = header.split(',')
    assert data.shape == (2, 76)
    assert data[0, names.index('Heart Rate')] == 82.0
    assert data[1, names.index('Heart Rate')] == 0.0
    assert data[0, names.index('Glucose')] == 0.0
    assert data[1, names.index('Glucose')] == 110.0
    assert data[0, names.index('mask->Heart Rate')] == 1.0
    assert data[1, names.index('mask->Heart Rate')] == 0.0
    assert data[0, names.index('mask->Glucose')] == 0.0
    assert data[1, names.index('mask->Glucose')] == 1.0


def test_raw_sequence_encoder_uses_discretizer_categorical_vocabulary():
    encoder = RawSequenceEncoder()
    data, header = encoder.transform(a([['1.0', '1.0']], CAT_HEADER), header=CAT_HEADER, end=24.0)
    names = header.split(',')
    assert data.shape == (1, 76)
    assert data[0, names.index('Capillary refill rate->1.0')] == 1.0
    assert data[0, names.index('Capillary refill rate->0.0')] == 0.0
    assert data[0, names.index('mask->Capillary refill rate')] == 1.0


def test_raw_observed_normalizer_uses_observed_continuous_values_only():
    encoder = RawSequenceEncoder()
    X = a([['0.1', '70', '', ''], ['0.2', '', '120', ''], ['0.3', '90', '', '']], HEADER)
    data, header = encoder.transform(X, header=HEADER, end=24.0)
    names = header.split(',')
    normalizer = RawObservedNormalizer(encoder.continuous_value_fields(),
                                       encoder.continuous_mask_fields())
    normalizer._feed_data(data)
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.close()
    try:
        normalizer._save_params(tmp.name)
        loaded = RawObservedNormalizer(encoder.continuous_value_fields(),
                                       encoder.continuous_mask_fields())
        loaded.load_params(tmp.name)
        transformed = loaded.transform(data)
    finally:
        os.unlink(tmp.name)
    hr = names.index('Heart Rate')
    hr_mask = names.index('mask->Heart Rate')
    assert abs(transformed[0, hr] + transformed[2, hr]) < 1e-12
    assert transformed[1, hr] == 0.0
    assert transformed[1, hr_mask] == 0.0

def test_existing_sampling_behavior_unchanged_when_test_args_omitted():
    args = make_sampling_args('random_matched', 4.0, 555)
    assert fixed_main.resolve_test_sampling_args(args) == ('random_matched', 4.0, 555)
    assert fixed_main.sampling_regime_label(
        args.sampling_strategy, args.sampling_interval, args.sampling_seed) == 'random_matched-r4-sseed555'

    main_source = inspect.getsource(fixed_main.main)
    train_block = main_source.split("train_reader = build_reader", 1)[1].split("discretizer = Discretizer", 1)[0]
    val_block = main_source.split("val_reader = build_reader", 1)[1].split("train_raw, val_raw", 1)[0]
    assert 'args.sampling_strategy' in train_block
    assert 'args.sampling_interval' in train_block
    assert 'args.sampling_seed' in train_block
    assert 'args.sampling_strategy' in val_block
    assert 'args.sampling_interval' in val_block
    assert 'args.sampling_seed' in val_block


def test_test_mode_loader_path_does_not_load_train_val_raw():
    calls = []
    old_load_data = fixed_main.utils.load_data

    def fake_load_data(reader, discretizer, normalizer, small_part, return_names=False):
        calls.append((reader, return_names))
        return {'data': (np.zeros((1, 24, 1)), np.array([0])), 'names': ['stay.csv']}

    fixed_main.utils.load_data = fake_load_data
    try:
        ret = fixed_main.load_test_raw('test_reader', 'discretizer', 'normalizer', False)
    finally:
        fixed_main.utils.load_data = old_load_data
    assert ret['names'] == ['stay.csv']
    assert calls == [('test_reader', True)]

    main_source = inspect.getsource(fixed_main.main)
    test_block = main_source.split("elif args.mode == 'test':", 1)[1]
    assert 'load_train_val_raw' not in test_block
    assert 'load_test_raw' in test_block


def run_all():
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()


if __name__ == '__main__':
    run_all()
    print('ok')
