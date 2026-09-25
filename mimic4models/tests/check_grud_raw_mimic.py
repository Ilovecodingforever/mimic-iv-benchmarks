from __future__ import absolute_import
from __future__ import print_function

import argparse
import os
import sys

if __package__ is None or __package__ == '':
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from mimic4benchmark.readers import FixedHorizonIcuExitReader
from mimic4models.fixed_horizon_icu_exit.matched_count_control.sampling import (
    DEFAULT_SAMPLING_SEED, SamplingReader, validate_sampling_args)
from mimic4models.fixed_horizon_icu_exit.raw import RawSequenceEncoder


def build_reader(data_dir, split, horizon):
    if split == 'test':
        dataset_dir = os.path.join(data_dir, 'test')
        listfile = os.path.join(data_dir, 'test_listfile.csv')
    else:
        dataset_dir = os.path.join(data_dir, 'train')
        listfile = os.path.join(data_dir, '{}_listfile.csv'.format(split))
    return FixedHorizonIcuExitReader(dataset_dir=dataset_dir, listfile=listfile,
                                     horizon=horizon)


def maybe_wrap_reader(reader, sampling_strategy, sampling_interval, sampling_seed):
    if sampling_strategy == 'none':
        return reader
    return SamplingReader(reader, sampling_strategy, int(sampling_interval),
                          sampling_seed)


def load_raw_grud_sequences(reader, limit=None):
    encoder = RawSequenceEncoder(include_timestamps=True)
    n_examples = reader.get_number_of_examples()
    if limit is not None and limit >= 0:
        n_examples = min(n_examples, int(limit))
    sequences = []
    names = []
    for i in range(n_examples):
        if i % 1000 == 0:
            print('encoded {} / {} examples'.format(i, n_examples), end='\r')
        example = reader.read_example(i)
        encoded, _ = encoder.transform(example['X'], header=example['header'],
                                       end=example['t'])
        sequences.append(encoded)
        names.append(example.get('name', str(i)))
    print('encoded {} / {} examples'.format(n_examples, n_examples))
    return sequences, encoder.header(), names


def parse_args():
    parser = argparse.ArgumentParser(
        description='Check that real raw rows used by GRU-D have nonzero expanded masks.')
    parser.add_argument('--data', required=True,
                        help='Path to fixed-horizon ICU-exit task data.')
    parser.add_argument('--split', default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--horizon', type=int, default=24,
                        choices=FixedHorizonIcuExitReader.VALID_HORIZONS)
    parser.add_argument('--limit', type=int, default=-1,
                        help='Maximum examples to check. Default -1 checks the full split.')
    parser.add_argument('--sampling_strategy', default='none',
                        choices=['none', 'structured', 'random_matched'])
    parser.add_argument('--sampling_interval', type=float, default=4.0,
                        choices=[2.0, 4.0, 8.0])
    parser.add_argument('--sampling_seed', type=int, default=DEFAULT_SAMPLING_SEED)
    parser.add_argument('--max_violations', type=int, default=10,
                        help='Maximum violation details to print.')
    parser.add_argument('--fail_on_violation', action='store_true',
                        help='Exit with status 1 if any all-zero real rows are found.')
    return parser.parse_args()


def main():
    args = parse_args()
    validate_sampling_args(args.sampling_strategy, args.sampling_interval, 0.0)
    reader = build_reader(args.data, args.split, args.horizon)
    reader = maybe_wrap_reader(reader, args.sampling_strategy,
                               args.sampling_interval, args.sampling_seed)
    sequences, header, names = load_raw_grud_sequences(reader, limit=args.limit)
    from mimic4models.keras_models.grud import print_raw_grud_real_row_mask_sanity
    report = print_raw_grud_real_row_mask_sanity(
        sequences, header, max_violations=args.max_violations)
    for violation in report['violations']:
        example_i = violation['example_index']
        violation['name'] = names[example_i] if example_i < len(names) else str(example_i)
    if report['violations']:
        print('first violations with names: {}'.format(report['violations']))
    if args.fail_on_violation and report['all_zero_real_rows'] > 0:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
