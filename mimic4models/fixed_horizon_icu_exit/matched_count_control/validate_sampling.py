from __future__ import absolute_import
from __future__ import print_function

import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mimic4benchmark.readers import FixedHorizonIcuExitReader
from mimic4models.preprocessing import Discretizer
from mimic4models.fixed_horizon_icu_exit.matched_count_control.sampling import (
    DEFAULT_SAMPLING_SEED, apply_sampling_to_example, cell_counts_by_channel,
    mask_counts_by_channel, nonempty_cell_count, select_last_raw_observation_per_bin)


def build_reader(data_dir, split, horizon):
    if split == 'test':
        dataset_dir = os.path.join(data_dir, 'test')
        listfile = os.path.join(data_dir, 'test_listfile.csv')
    else:
        dataset_dir = os.path.join(data_dir, 'train')
        listfile = os.path.join(data_dir, '{}_listfile.csv'.format(split))
    return FixedHorizonIcuExitReader(dataset_dir=dataset_dir, listfile=listfile, horizon=horizon)


def _channel_value_matches(discretized, discretized_header, bin_id, channel, raw_value):
    if channel in discretized_header:
        return abs(discretized[bin_id, discretized_header.index(channel)] - float(raw_value)) <= 1e-12
    categorical_name = channel + '->' + raw_value
    if categorical_name not in discretized_header:
        return False
    return discretized[bin_id, discretized_header.index(categorical_name)] == 1.0


def structured_vs_coarse_value_mismatches(example, interval, imputation='previous'):
    """Count observed coarse cells where structured selection disagrees with D.

    D's masks identify genuinely observed cells; imputed values are ignored.
    """
    discretizer = Discretizer(timestep=float(interval), store_masks=True,
                              impute_strategy=imputation, start_time='zero')
    coarse, coarse_header = discretizer.transform(example['X'], header=example['header'],
                                                  end=example['t'])
    coarse_header = coarse_header.split(',')
    selected = select_last_raw_observation_per_bin(example['X'], example['header'],
                                                   interval, end=example['t'])
    mismatches = 0
    for col_id in range(1, len(example['header'])):
        channel = example['header'][col_id]
        mask_name = 'mask->' + channel
        if mask_name not in coarse_header:
            mismatches += coarse.shape[0]
            continue
        mask_col = coarse_header.index(mask_name)
        for bin_id in range(coarse.shape[0]):
            if coarse[bin_id, mask_col] != 1:
                continue
            cell = selected.get((bin_id, col_id))
            if cell is None:
                mismatches += 1
                continue
            if not _channel_value_matches(coarse, coarse_header, bin_id, channel, cell['value']):
                mismatches += 1
    return mismatches


def add_counts(total, counts):
    for key, value in counts.items():
        total[key] = total.get(key, 0) + value


def changed(before, after):
    if before.shape != after.shape:
        return True
    return bool(np.any(before != after))


def validate_interval(reader, interval, num_examples, sampling_seed, imputation):
    d1 = Discretizer(timestep=1.0, store_masks=True,
                     impute_strategy=imputation, start_time='zero')
    dr = Discretizer(timestep=float(interval), store_masks=True,
                     impute_strategy=imputation, start_time='zero')

    n = min(num_examples, reader.get_number_of_examples())
    stats = {'examples': n,
             'raw_total': 0,
             'b_total': 0,
             'c_total': 0,
             'affected': 0,
             'count_mismatches': 0,
             'mask_mismatches': 0,
             'after_24': 0,
             'structured_vs_coarse_value_mismatches': 0}
    b_by_channel = {}
    c_by_channel = {}

    for i in range(n):
        raw = reader.read_example(i)
        b = apply_sampling_to_example(raw, 'structured', interval, sampling_seed)
        c = apply_sampling_to_example(raw, 'random_matched', interval, sampling_seed)
        c_repeat = apply_sampling_to_example(raw, 'random_matched', interval, sampling_seed)

        for key in ('name', 't', 'y'):
            if raw[key] != b[key] or raw[key] != c[key]:
                raise AssertionError('{} changed for example {}'.format(key, i))
        if list(raw['header']) != list(b['header']) or list(raw['header']) != list(c['header']):
            raise AssertionError('header changed for example {}'.format(i))
        if not np.array_equal(c['X'], c_repeat['X']):
            raise AssertionError('random sampling is not deterministic for {}'.format(raw['name']))

        stats['raw_total'] += nonempty_cell_count(raw['X'])
        stats['b_total'] += nonempty_cell_count(b['X'])
        stats['c_total'] += nonempty_cell_count(c['X'])
        if changed(raw['X'], b['X']):
            stats['affected'] += 1
        if b['X'].shape[0] and max(float(row[0]) for row in b['X']) > 24.0 + 1e-6:
            stats['after_24'] += 1
        if c['X'].shape[0] and max(float(row[0]) for row in c['X']) > 24.0 + 1e-6:
            stats['after_24'] += 1

        b_counts = cell_counts_by_channel(b['X'], b['header'])
        c_counts = cell_counts_by_channel(c['X'], c['header'])
        add_counts(b_by_channel, b_counts)
        add_counts(c_by_channel, c_counts)
        for channel in b_counts:
            if b_counts[channel] != c_counts.get(channel, 0):
                stats['count_mismatches'] += 1

        bx, bh = d1.transform(b['X'], end=b['t'])
        cx, ch = d1.transform(c['X'], end=c['t'])
        if bx.shape[0] != 24 or cx.shape[0] != 24:
            raise AssertionError('B/C sequence length is not 24 for {}'.format(raw['name']))
        b_masks = mask_counts_by_channel(bx, bh)
        c_masks = mask_counts_by_channel(cx, ch)
        for channel in b_masks:
            if b_masks[channel] != c_masks.get(channel, 0):
                stats['mask_mismatches'] += 1

        stats['structured_vs_coarse_value_mismatches'] += structured_vs_coarse_value_mismatches(
            raw, interval, imputation)

        dx, _ = dr.transform(raw['X'], header=raw['header'], end=raw['t'])
        expected_d_steps = int(24 / interval)
        if dx.shape[0] != expected_d_steps:
            raise AssertionError('D length for r={} is {}, expected {}'.format(
                interval, dx.shape[0], expected_d_steps))

    if stats['count_mismatches']:
        raise AssertionError('raw B/C count mismatches: {}'.format(stats['count_mismatches']))
    if stats['mask_mismatches']:
        raise AssertionError('post-1h mask mismatches: {}'.format(stats['mask_mismatches']))
    if stats['after_24']:
        raise AssertionError('selected observations after 24h: {}'.format(stats['after_24']))
    if stats['structured_vs_coarse_value_mismatches']:
        raise AssertionError('structured vs coarse value mismatches: {}'.format(
            stats['structured_vs_coarse_value_mismatches']))

    print('r={} examples={} raw_obs={} B_obs={} C_obs={} retention={:.6f} affected={}'.format(
        interval, stats['examples'], stats['raw_total'], stats['b_total'], stats['c_total'],
        float(stats['b_total']) / stats['raw_total'] if stats['raw_total'] else 0.0,
        stats['affected']))
    print('  mismatched patient-variable counts=0 post_1h_mask_mismatches=0 after_24=0 '
          'structured_vs_coarse_value_mismatches=0')
    print('  per-variable B counts:', b_by_channel)
    print('  per-variable C counts:', c_by_channel)


def main():
    parser = argparse.ArgumentParser(description='Validate matched-count sampling invariants.')
    parser.add_argument('--data', required=True, help='Path to fixed-horizon ICU-exit task data.')
    parser.add_argument('--split', default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--horizon', type=int, default=24,
                        choices=FixedHorizonIcuExitReader.VALID_HORIZONS)
    parser.add_argument('--num_examples', type=int, default=100)
    parser.add_argument('--intervals', type=int, nargs='+', default=[2, 4, 8], choices=[2, 4, 8])
    parser.add_argument('--sampling_seed', type=int, default=DEFAULT_SAMPLING_SEED)
    parser.add_argument('--imputation', default='previous', choices=['zero', 'next', 'previous', 'normal_value'])
    args = parser.parse_args()

    reader = build_reader(args.data, args.split, args.horizon)
    print('split={} horizon={} available_examples={} checking={}'.format(
        args.split, args.horizon, reader.get_number_of_examples(),
        min(args.num_examples, reader.get_number_of_examples())))
    for interval in args.intervals:
        validate_interval(reader, interval, args.num_examples, args.sampling_seed, args.imputation)


if __name__ == '__main__':
    main()
