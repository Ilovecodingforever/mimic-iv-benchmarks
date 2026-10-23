from __future__ import absolute_import

import os
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd

from mimic4models.fixed_horizon_icu_exit import main as fixed_main
from mimic4models.fixed_horizon_icu_exit import feature_frequency as runner
from mimic4models.fixed_horizon_icu_exit.feature_frequency import (
    EXP2A_LSTM_CONFIG, build_command, condition_dir, parse_args)
from mimic4models.fixed_horizon_icu_exit.matched_count_control.sampling import (
    feature_structured_sample, structured_sample)
from mimic4models.preprocessing import Discretizer


HEADER = ['Hours', 'Heart Rate', 'Glucose', 'pH']


def _a(rows):
    return np.asarray(rows, dtype=object)


def _discretize(X, end=24.0):
    return Discretizer(timestep=1.0, store_masks=True,
                       impute_strategy='previous', start_time='zero').transform(
                           X, header=HEADER, end=end)


def _output_columns(header, channel):
    names = header.split(',')
    return names.index(channel), names.index('mask->{}'.format(channel))


def test_feature_normalizer_paths_are_condition_specific():
    dense = fixed_main.default_normalizer_state_path(
        '/norm', 1.0, 'previous', 10)
    feature = fixed_main.default_normalizer_state_path(
        '/norm', 1.0, 'previous', 10,
        feature_name='Heart Rate', feature_sampling_interval=4)
    other = fixed_main.default_normalizer_state_path(
        '/norm', 1.0, 'previous', 10,
        feature_name='Glucose', feature_sampling_interval=4)
    shared = fixed_main.default_normalizer_state_path(
        '/norm', 1.0, 'previous', 10,
        feature_name='Heart Rate', feature_sampling_interval=1)
    assert dense.endswith('fixed_horizon_icu_exit_ts:1.00_impute:previous_start:zero_masks:True_n:10.normalizer')
    assert shared == dense
    assert feature != dense and other != feature
    assert 'feature:8_Heart_Rate_r:4' in feature
    assert 'feature:7_Glucose_r:4' in other


def test_feature_runner_paths_and_commands_are_distinct():
    baseline = condition_dir('/out', 12, 'none', 1)
    heart_rate = condition_dir('/out', 12, 'Heart Rate', 4)
    glucose = condition_dir('/out', 12, 'Glucose', 4)
    assert baseline == os.path.join('/out', '12h', 'baseline', 'r1')
    assert heart_rate != glucose and heart_rate.endswith('feature-8_Heart_Rate/r4')
    command = build_command('python', 'train', '/data', '/norm', heart_rate,
                             12, 0, 'Heart Rate', 4)
    assert '--feature_name' in command
    assert command[command.index('--feature_name') + 1] == 'Heart Rate'
    assert command[command.index('--timestep') + 1] == '1.0'


def test_feature_runner_matches_experiment_2a_configuration_and_labels():
    command = build_command('python', 'train', '/data', '/norm', '/out', 12, 3,
                             'none', 1, epochs=100)
    values = dict((name, command[command.index(name) + 1]) for name in (
        '--dim', '--depth', '--dropout', '--rec_dropout', '--batch_size',
        '--optimizer', '--lr', '--beta_1', '--l1', '--l2',
        '--target_repl_coef', '--imputation', '--timestep', '--horizon',
        '--seed', '--feature_name', '--feature_sampling_interval'))
    assert values['--dim'] == str(EXP2A_LSTM_CONFIG['dim'])
    assert values['--depth'] == str(EXP2A_LSTM_CONFIG['depth'])
    assert values['--dropout'] == str(EXP2A_LSTM_CONFIG['dropout'])
    assert values['--rec_dropout'] == str(EXP2A_LSTM_CONFIG['rec_dropout'])
    assert values['--batch_size'] == str(EXP2A_LSTM_CONFIG['batch_size'])
    assert values['--optimizer'] == 'adam'
    assert values['--lr'] == '0.001'
    assert values['--beta_1'] == '0.9'
    assert values['--l1'] == '0.0' and values['--l2'] == '0.0'
    assert values['--target_repl_coef'] == '0.0'
    assert values['--imputation'] == 'previous'
    assert values['--timestep'] == '1.0'
    assert values['--horizon'] == '12' and values['--seed'] == '3'
    assert values['--feature_name'] == 'none'
    assert values['--feature_sampling_interval'] == '1'


def test_feature_sampling_integration_changes_only_selected_feature():
    X = _a([
        ['0.2', '70', '100', '7.0'],
        ['1.2', '71', '101', ''],
        ['3.2', '72', '', '7.1'],
        ['5.2', '73', '103', ''],
        ['25.0', '999', '999', '9.9'],
    ])
    sampled, _ = feature_structured_sample(X, HEADER, 'Heart Rate', 4, end=8.0)
    original_data, original_header = _discretize(X, end=8.0)
    sampled_data, sampled_header = _discretize(sampled, end=8.0)
    assert original_data.shape == sampled_data.shape == (8, original_data.shape[1])
    for channel in ('Glucose', 'pH'):
        original_value, original_mask = _output_columns(original_header, channel)
        sampled_value, sampled_mask = _output_columns(sampled_header, channel)
        assert np.array_equal(original_data[:, original_value], sampled_data[:, sampled_value])
        assert np.array_equal(original_data[:, original_mask], sampled_data[:, sampled_mask])
    original_hr_value, original_hr_mask = _output_columns(original_header, 'Heart Rate')
    sampled_hr_value, sampled_hr_mask = _output_columns(sampled_header, 'Heart Rate')
    assert not np.array_equal(original_data[:, original_hr_mask], sampled_data[:, sampled_hr_mask])
    assert original_data[:, original_hr_value].tolist() != sampled_data[:, sampled_hr_value].tolist()
    # The row after the prediction window cannot affect the fixed 1h encoding.
    prefix = X[:4]
    prefix_data, _ = _discretize(prefix, end=8.0)
    assert np.array_equal(prefix_data, original_data)
    assert sampled_data.shape[0] == original_data.shape[0]


def test_feature_sampling_empty_feature_keeps_other_inputs_and_fixed_length():
    X = _a([['0.5', '', '100', ''], ['2.5', '', '101', '7.0']])
    sampled, _ = feature_structured_sample(X, HEADER, 'Heart Rate', 24, end=24.0)
    original_data, original_header = _discretize(X)
    sampled_data, sampled_header = _discretize(sampled)
    assert sampled.tolist() == X.tolist()
    assert original_data.shape == sampled_data.shape == (24, original_data.shape[1])
    for channel in ('Heart Rate', 'Glucose', 'pH'):
        original_value, original_mask = _output_columns(original_header, channel)
        sampled_value, sampled_mask = _output_columns(sampled_header, channel)
        assert np.array_equal(original_data[:, original_value], sampled_data[:, sampled_value])
        assert np.array_equal(original_data[:, original_mask], sampled_data[:, sampled_mask])


def test_all_feature_and_feature_specific_thinning_are_compared_on_the_same_1h_grid():
    X = _a([['0.2', '70', '100', '7.0'], ['1.2', '71', '101', ''],
           ['7.2', '72', '102', '7.1'], ['23.2', '73', '103', '']])
    for interval in (8, 24):
        all_features, _ = structured_sample(X, HEADER, interval)
        one_feature, _ = feature_structured_sample(X, HEADER, 'Heart Rate', interval)
        all_data, all_header = _discretize(all_features)
        one_data, one_header = _discretize(one_feature)
        assert all_data.shape == one_data.shape == (24, all_data.shape[1])
        _, all_hr_mask = _output_columns(all_header, 'Heart Rate')
        _, one_hr_mask = _output_columns(one_header, 'Heart Rate')
        assert np.array_equal(all_data[:, all_hr_mask], one_data[:, one_hr_mask])
        # Feature-specific thinning preserves the other channels that all-feature
        # thinning intentionally removes; neither result is a coarse-grid run.
        _, one_glucose_mask = _output_columns(one_header, 'Glucose')
        assert int(np.sum(one_data[:, one_glucose_mask])) >= int(np.sum(all_data[:, _output_columns(all_header, 'Glucose')[1]]))


def test_validation_examples_use_train_directory_and_validation_listfile():
    calls = []
    old_reader = runner.FixedHorizonIcuExitReader

    class FakeReader(object):
        def __init__(self, dataset_dir, listfile, horizon):
            calls.append((dataset_dir, listfile, horizon))

        def get_number_of_examples(self):
            return 1

        def read_example(self, index):
            return {'index': index}

    runner.FixedHorizonIcuExitReader = FakeReader
    try:
        assert list(runner._examples('/data', 'val', 12)) == [{'index': 0}]
    finally:
        runner.FixedHorizonIcuExitReader = old_reader
    assert calls == [('/data/train', '/data/val_listfile.csv', 12)]


def test_retention_stats_create_feature_condition_directory():
    root = tempfile.mkdtemp()
    old_statistics = runner.feature_sampling_statistics
    try:
        runner.feature_sampling_statistics = lambda examples, feature, interval: {
            'original_measurements': 1,
            'retained_measurements': 1,
        }
        args = SimpleNamespace(data='/data', output_root=root)
        path = runner.write_retention_stats(args, 12, 'Heart Rate', 1)
        assert os.path.isfile(path)
        assert path.endswith(os.path.join(
            '12h', 'feature-8_Heart_Rate', 'r1',
            'feature_measurement_retention.csv'))
    finally:
        runner.feature_sampling_statistics = old_statistics
        import shutil
        shutil.rmtree(root)


def test_runner_detects_complete_and_partial_artifacts_without_training():
    root = tempfile.mkdtemp()
    output = os.path.join(root, '12h', 'baseline', 'r1')
    logs = os.path.join(output, 'keras_logs')
    states = os.path.join(output, 'keras_states')
    os.makedirs(logs)
    os.makedirs(states)
    log_name = 'k_lstm.n16.d0.3.dep2.bs8.ts1.0.seed0.csv'
    log_path = os.path.join(logs, log_name)
    pd.DataFrame({'epoch': [0, 1, 2], 'val_auprc': [0.1, 0.3, 0.2],
                  'val_loss': [0.8, 0.5, 0.7]}).to_csv(
        log_path, sep=';', index=False)
    checkpoint = os.path.join(states, log_name[:-4] + '.epoch2.test0.5.state')
    open(checkpoint, 'w').close()
    duplicate = os.path.join(states, log_name[:-4] + '.epoch2.test0.6.state')
    open(duplicate, 'w').close()
    old_ensure_normalizer = runner.ensure_normalizer
    old_check_call = runner.subprocess.check_call
    try:
        assert runner._run_log(output, 0) == log_path
        assert runner._selected_checkpoint(log_path, expected_epochs=3) == checkpoint
        assert runner._selected_checkpoint(log_path, expected_epochs=4) is None

        partial = os.path.join(logs, 'k_lstm.n16.d0.3.dep2.bs8.ts1.0.seed1.csv')
        pd.DataFrame({'epoch': [0], 'val_auprc': [0.1]}).to_csv(
            partial, sep=';', index=False)
        assert runner._selected_checkpoint(partial, expected_epochs=3) is None

        malformed = os.path.join(logs, 'k_lstm.n16.d0.3.dep2.bs8.ts1.0.seed2.csv')
        pd.DataFrame({'epoch': ['not-an-epoch']}).to_csv(malformed, sep=';', index=False)
        assert runner._selected_checkpoint(malformed, expected_epochs=3) is None

        calls = []
        runner.ensure_normalizer = lambda *args: '/norm/shared.normalizer'

        def fake_check_call(command):
            calls.append(command)
            prediction = runner._prediction_path(output, checkpoint)
            prediction_dir = os.path.dirname(prediction)
            if not os.path.isdir(prediction_dir):
                os.makedirs(prediction_dir)
            pd.DataFrame({'stay': ['a.csv', 'b.csv'],
                          'prediction': [0.1, 0.9],
                          'y_true': [0, 1]}).to_csv(prediction, index=False)

        runner.subprocess.check_call = fake_check_call
        args = SimpleNamespace(output_root=root, data='/data', normalizer_dir='/norm',
                               python='python', mode='test', force=False, epochs=3)
        result = runner.run_condition(args, 12, 0, 'none', 1)
        assert result['status'] == 'complete'
        assert len(calls) == 1
        calls[:] = []
        assert runner.run_condition(args, 12, 0, 'none', 1)['status'] == 'complete'
        assert calls == []
    finally:
        runner.ensure_normalizer = old_ensure_normalizer
        runner.subprocess.check_call = old_check_call
        import shutil
        shutil.rmtree(root)


def test_feature_runner_defaults_to_initial_experiment_grid():
    args = parse_args(['--data', '/data', '--normalizer_dir', '/norm',
                       '--output_root', '/out'])
    assert args.mode == 'all'
    assert args.epochs == 100


def test_feature_runner_reuses_one_baseline_for_each_feature_at_interval_one():
    root = tempfile.mkdtemp()
    calls = []
    old_run_condition = runner.run_condition
    old_write_stats = runner.write_retention_stats
    try:
        def fake_run_condition(args, horizon, seed, feature_name, interval):
            calls.append((horizon, seed, feature_name, interval))
            return {'status': 'complete'}

        runner.run_condition = fake_run_condition
        runner.write_retention_stats = lambda *args: None
        runner.main([
            '--data', '/data', '--normalizer_dir', '/norm', '--output_root', root,
            '--horizon', '12', '--features', 'Heart Rate', 'Glucose',
            '--intervals', '1', '4', '--seeds', '0'])
    finally:
        runner.run_condition = old_run_condition
        runner.write_retention_stats = old_write_stats
        import shutil
        shutil.rmtree(root)
    assert calls == [
        (12, 0, 'none', 1),
        (12, 0, 'Heart Rate', 4),
        (12, 0, 'Glucose', 4),
    ]


def test_feature_sampling_rejects_non_one_hour_discretization_and_mixed_sampling():
    args = SimpleNamespace(
        sampling_strategy='none', sampling_interval=4.0, sampling_seed=100,
        timestep=2.0, feature_name='Heart Rate', feature_sampling_interval=4.0,
        test_sampling_strategy=None, test_sampling_interval=None,
        test_sampling_seed=None)
    try:
        fixed_main.validate_all_sampling_args(args)
    except ValueError as exc:
        assert 'timestep=1.0' in str(exc)
    else:
        raise AssertionError('non-1h feature sampling was accepted')

    args.timestep = 1.0
    args.sampling_strategy = 'structured'
    try:
        fixed_main.validate_all_sampling_args(args)
    except ValueError as exc:
        assert 'combined with whole-feature sampling' in str(exc)
    else:
        raise AssertionError('mixed sampling interventions were accepted')


if __name__ == '__main__':
    for name, value in sorted(globals().items()):
        if name.startswith('test_'):
            value()
    print('feature frequency tests passed')
