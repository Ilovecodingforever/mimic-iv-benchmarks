from __future__ import absolute_import
from __future__ import print_function

import argparse
import glob
import os
import re
import subprocess
import sys

import pandas as pd

from mimic4benchmark.readers import FixedHorizonIcuExitReader
from mimic4models.fixed_horizon_icu_exit import main as fixed_main
from mimic4models.fixed_horizon_icu_exit.matched_count_control.sampling import (
    feature_name_slug, feature_sampling_statistics, feature_sampling_channels)


FEATURES = ('Heart Rate', 'Mean blood pressure', 'Temperature', 'Glucose')
INTERVALS = (1, 2, 4, 8, 12, 24)
SEEDS = (0, 1, 2, 3, 4)
HORIZONS = tuple(FixedHorizonIcuExitReader.VALID_HORIZONS)

# Keep this explicit so the feature-frequency baseline cannot silently drift
# from the Experiment 2A LSTM configuration.
EXP2A_LSTM_CONFIG = {
    'dim': 16,
    'depth': 2,
    'dropout': 0.3,
    'rec_dropout': 0.0,
    'batch_size': 8,
    'optimizer': 'adam',
    'lr': 0.001,
    'beta_1': 0.9,
    'l1': 0.0,
    'l2': 0.0,
    'target_repl_coef': 0.0,
    'imputation': 'previous',
    'prefix': '',
    'timestep': 1.0,
    'save_every': 1,
}


def condition_dir(output_root, horizon, feature_name, interval):
    if feature_name in (None, '', 'none'):
        return os.path.join(output_root, '{}h'.format(horizon), 'baseline', 'r1')
    return os.path.join(output_root, '{}h'.format(horizon),
                       'feature-{}'.format(feature_name_slug(feature_name)),
                       'r{}'.format(int(interval)))


def _normalizer_path(normalizer_dir, data_dir, horizon, feature_name, interval):
    reader = FixedHorizonIcuExitReader(
        dataset_dir=os.path.join(data_dir, 'train'),
        listfile=os.path.join(data_dir, 'train_listfile.csv'),
        horizon=horizon)
    return fixed_main.default_normalizer_state_path(
        normalizer_dir, 1.0, 'previous', reader.get_number_of_examples(),
        feature_name=feature_name, feature_sampling_interval=interval)


def build_command(python, mode, data_dir, normalizer_dir, output_dir,
                  horizon, seed, feature_name, interval, epochs=100):
    command = [python, '-m', 'mimic4models.fixed_horizon_icu_exit.main',
               '--network', 'mimic4models/keras_models/lstm.py',
               '--dim', str(EXP2A_LSTM_CONFIG['dim']),
               '--depth', str(EXP2A_LSTM_CONFIG['depth']),
               '--dropout', str(EXP2A_LSTM_CONFIG['dropout']),
               '--rec_dropout', str(EXP2A_LSTM_CONFIG['rec_dropout']),
               '--batch_size', str(EXP2A_LSTM_CONFIG['batch_size']),
               '--epochs', str(epochs), '--mode', mode,
               '--timestep', str(EXP2A_LSTM_CONFIG['timestep']),
               '--save_every', str(EXP2A_LSTM_CONFIG['save_every']),
               '--horizon', str(horizon), '--seed', str(seed),
               '--data', data_dir, '--normalizer_dir', normalizer_dir,
               '--output_dir', output_dir, '--imputation', 'previous',
               '--optimizer', EXP2A_LSTM_CONFIG['optimizer'],
               '--lr', str(EXP2A_LSTM_CONFIG['lr']),
               '--beta_1', str(EXP2A_LSTM_CONFIG['beta_1']),
               '--l1', str(EXP2A_LSTM_CONFIG['l1']),
               '--l2', str(EXP2A_LSTM_CONFIG['l2']),
               '--target_repl_coef', str(EXP2A_LSTM_CONFIG['target_repl_coef']),
               '--prefix', EXP2A_LSTM_CONFIG['prefix'],
               '--sampling_strategy', 'none',
               '--feature_name', feature_name,
               '--feature_sampling_interval', str(interval)]
    return command


def ensure_normalizer(python, data_dir, normalizer_dir, horizon,
                      feature_name, interval):
    path = _normalizer_path(normalizer_dir, data_dir, horizon, feature_name, interval)
    if os.path.exists(path):
        return path
    command = [python, '-m', 'mimic4models.create_normalizer_state',
               '--task', 'fixed_horizon_icu_exit', '--data', data_dir,
               '--horizon', str(horizon), '--timestep', '1',
               '--impute_strategy', 'previous', '--start_time', 'zero',
               '--output_dir', normalizer_dir,
               '--feature_name', feature_name,
               '--feature_sampling_interval', str(interval)]
    print('creating normalizer:', ' '.join(command))
    subprocess.check_call(command)
    if not os.path.exists(path):
        raise RuntimeError('normalizer command completed without creating {}'.format(path))
    return path


def _run_log(output_dir, seed):
    log_dir = os.path.join(output_dir, 'keras_logs')
    if not os.path.isdir(log_dir):
        return None
    pattern = re.compile(r'^k_lstm.*\.seed{}\.csv$'.format(int(seed)))
    paths = sorted(os.path.join(log_dir, name)
                   for name in os.listdir(log_dir) if pattern.match(name))
    if len(paths) != 1:
        return None
    return paths[0]


def _selected_checkpoint(log_path, expected_epochs=100):
    try:
        frame = pd.read_csv(log_path, sep=';')
    except (IOError, OSError, ValueError, pd.errors.ParserError):
        return None
    required = {'epoch', 'val_auprc'}
    if not required.issubset(frame.columns):
        return None
    try:
        epochs = list(frame['epoch'].astype(int))
    except (TypeError, ValueError):
        return None
    if len(frame) != expected_epochs or epochs != list(range(expected_epochs)):
        return None
    valid = frame.dropna(subset=['val_auprc'])
    if len(valid) == 0:
        return None
    best_row = valid.loc[valid['val_auprc'].idxmax()]
    epoch = int(best_row['epoch']) + 1
    state_dir = os.path.join(os.path.dirname(os.path.dirname(log_path)), 'keras_states')
    stem = os.path.splitext(os.path.basename(log_path))[0]
    if os.path.isdir(state_dir) and 'val_loss' in best_row and pd.notna(best_row['val_loss']):
        exact = os.path.join(state_dir, '{}.epoch{}.test{}.state'.format(
            stem, epoch, best_row['val_loss']))
        if os.path.isfile(exact):
            return exact
    pattern = '{}.epoch{}.test*.state'.format(stem, epoch)
    states = glob.glob(os.path.join(state_dir, pattern)) if os.path.isdir(state_dir) else []
    if len(states) != 1:
        return None
    return states[0]


def _prediction_path(output_dir, checkpoint):
    return fixed_main.test_prediction_path(
        output_dir, checkpoint, 'none', 1, 100, 'none', 1, 100)


def run_condition(args, horizon, seed, feature_name, interval):
    output_dir = condition_dir(args.output_root, horizon, feature_name, interval)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    normalizer = ensure_normalizer(
        args.python, args.data, args.normalizer_dir, horizon, feature_name, interval)
    log_path = _run_log(output_dir, seed) if os.path.isdir(os.path.join(output_dir, 'keras_logs')) else None
    complete = log_path is not None and _selected_checkpoint(log_path, args.epochs) is not None
    if args.mode in ('train', 'all') and (args.force or not complete):
        command = build_command(args.python, 'train', args.data, args.normalizer_dir,
                                output_dir, horizon, seed, feature_name, interval, args.epochs)
        subprocess.check_call(command)
        log_path = _run_log(output_dir, seed)
    if args.mode == 'train':
        return {'status': 'trained', 'output_dir': output_dir, 'normalizer_path': normalizer}
    if log_path is None:
        return {'status': 'missing_log', 'output_dir': output_dir, 'normalizer_path': normalizer}
    checkpoint = _selected_checkpoint(log_path, args.epochs)
    if checkpoint is None:
        return {'status': 'missing_checkpoint', 'output_dir': output_dir, 'normalizer_path': normalizer}
    prediction = _prediction_path(output_dir, checkpoint)
    if args.mode in ('test', 'all') and (args.force or not os.path.exists(prediction)):
        command = build_command(args.python, 'test', args.data, args.normalizer_dir,
                                output_dir, horizon, seed, feature_name, interval, args.epochs)
        command += ['--normalizer_state', normalizer, '--load_state', checkpoint]
        subprocess.check_call(command)
    if not os.path.exists(prediction):
        return {'status': 'missing_prediction', 'output_dir': output_dir,
                'checkpoint': checkpoint, 'normalizer_path': normalizer}
    from visualizers.visualize_utils import load_prediction_csv, prediction_metrics
    metrics = prediction_metrics(load_prediction_csv(prediction))
    return {'status': 'complete', 'output_dir': output_dir,
            'checkpoint': checkpoint, 'prediction_path': prediction,
            'normalizer_path': normalizer, **metrics}


def _examples(data_dir, split, horizon):
    if split == 'test':
        dataset_dir = os.path.join(data_dir, 'test')
        listfile = os.path.join(data_dir, 'test_listfile.csv')
    else:
        dataset_dir = os.path.join(data_dir, 'train')
        listfile = os.path.join(data_dir, '{}_listfile.csv'.format(split))
    reader = FixedHorizonIcuExitReader(
        dataset_dir=dataset_dir, listfile=listfile, horizon=horizon)
    for index in range(reader.get_number_of_examples()):
        yield reader.read_example(index)


def write_retention_stats(args, horizon, feature_name, interval):
    if feature_name in (None, '', 'none'):
        return None
    rows = []
    for split in ('train', 'val', 'test'):
        stats = feature_sampling_statistics(
            _examples(args.data, split, horizon), feature_name, interval)
        stats.update({'split': 'validation' if split == 'val' else split,
                      'horizon': horizon})
        rows.append(stats)
    path = os.path.join(condition_dir(args.output_root, horizon, feature_name, interval),
                        'feature_measurement_retention.csv')
    # The shared r=1 baseline is stored separately, so no feature condition
    # directory may have been created before its retention table is written.
    if not os.path.isdir(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Feature-specific fixed-horizon frequency grid runner.')
    parser.add_argument('--data', required=True)
    parser.add_argument('--normalizer_dir', required=True)
    parser.add_argument('--output_root', required=True)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--mode', choices=('train', 'test', 'all'), default='all')
    parser.add_argument('--horizon', dest='horizon_values', action='append', type=int)
    parser.add_argument('--horizons', nargs='+', type=int)
    parser.add_argument('--feature_name', dest='feature_values', action='append')
    parser.add_argument('--features', nargs='+')
    parser.add_argument('--feature_sampling_interval', dest='interval_values', action='append', type=int)
    parser.add_argument('--intervals', nargs='+', type=int)
    parser.add_argument('--seed', dest='seed_values', action='append', type=int)
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--force', action='store_true')
    return parser.parse_args(argv)


def _values(single, multiple, default):
    values = []
    if single:
        values.extend(single)
    if multiple:
        values.extend(multiple)
    return values or list(default)


def main(argv=None):
    args = parse_args(argv)
    horizons = _values(args.horizon_values, args.horizons, (12,))
    features = _values(args.feature_values, args.features, FEATURES)
    intervals = _values(args.interval_values, args.intervals, INTERVALS)
    seeds = _values(args.seed_values, args.seeds, SEEDS)
    invalid = set(features) - set(feature_sampling_channels()) - {'none'}
    if invalid:
        raise ValueError('unsupported feature(s): {}'.format(sorted(invalid)))
    if 'none' in features and any(interval != 1 for interval in intervals):
        raise ValueError('feature_name none is only valid for the shared interval-1 baseline')
    for horizon in horizons:
        if horizon not in HORIZONS:
            raise ValueError('horizon must be one of {}'.format(HORIZONS))
    if any(interval not in INTERVALS for interval in intervals):
        raise ValueError('intervals must be one of {}'.format(INTERVALS))

    rows = []
    written_stats = set()
    for horizon in horizons:
        for seed in seeds:
            baseline = run_condition(args, horizon, seed, 'none', 1)
            for feature_name in features:
                for interval in intervals:
                    if interval == 1:
                        result = baseline
                    else:
                        result = run_condition(args, horizon, seed, feature_name, interval)
                    row = {'horizon': horizon, 'seed': seed,
                           'feature_name': feature_name,
                           'feature_sampling_interval': interval}
                    row.update(result)
                    rows.append(row)
                    stats_key = (horizon, feature_name, interval)
                    if stats_key not in written_stats:
                        write_retention_stats(args, horizon, feature_name, interval)
                        written_stats.add(stats_key)
    output_path = os.path.join(args.output_root, 'feature_frequency_runs.csv')
    if not os.path.isdir(args.output_root):
        os.makedirs(args.output_root)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print('wrote', output_path)


if __name__ == '__main__':
    main()
