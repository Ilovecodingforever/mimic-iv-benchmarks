from __future__ import absolute_import

import os
import tempfile
from types import SimpleNamespace

from mimic4models.fixed_horizon_icu_exit import main as fixed_main
from mimic4models.fixed_horizon_icu_exit import feature_frequency as runner
from mimic4models.fixed_horizon_icu_exit.feature_frequency import (
    build_command, condition_dir, parse_args)


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
    assert command[command.index('--timestep') + 1] == '1'


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
