from __future__ import absolute_import
from __future__ import print_function

import hashlib
import random

import numpy as np

from mimic4models.preprocessing import DISCRETIZER_EPS, discretizer_bin_id


VALID_STRATEGIES = ('none', 'structured', 'random_matched')
VALID_INTERVALS = (2, 4, 8)
DEFAULT_SAMPLING_SEED = 100


def _as_rows(X):
    return [list(row) for row in X]


def _num_bins(end, timestep, first_time=0.0, eps=DISCRETIZER_EPS):
    max_hours = end - first_time
    return int(max_hours / timestep + 1.0 - eps)


def _first_time(rows, start_time):
    if start_time == 'zero':
        return 0.0
    if start_time == 'relative':
        if not rows:
            raise ValueError('relative start_time requires at least one row')
        return float(rows[0][0])
    raise ValueError('start_time is invalid')


def make_cell(row_id, col_id, bin_id, time, value, channel):
    return {'row_id': row_id,
            'col_id': col_id,
            'bin_id': bin_id,
            'time': time,
            'value': value,
            'channel': channel}


def select_last_raw_observation_per_bin(X, header, timestep, end=24.0,
                                        start_time='zero', eps=DISCRETIZER_EPS):
    """Return the raw row/column cell that Discretizer would keep per bin.

    The key is (bin_id, column_id). Overwriting happens in input row order,
    matching Discretizer.transform exactly, including equal timestamps.
    """
    assert header[0] == 'Hours'
    rows = _as_rows(X)
    first_time = _first_time(rows, start_time)
    max_hours = end - first_time
    n_bins = _num_bins(end, timestep, first_time, eps)
    selected = {}
    ts = [float(row[0]) for row in rows]
    for i in range(len(ts) - 1):
        assert ts[i] < ts[i + 1] + eps
    for row_id, row in enumerate(rows):
        t = float(row[0]) - first_time
        if t > max_hours + eps:
            continue
        bin_id = discretizer_bin_id(t, timestep, eps)
        assert 0 <= bin_id < n_bins
        for col_id in range(1, len(row)):
            if row[col_id] == '':
                continue
            selected[(bin_id, col_id)] = make_cell(row_id, col_id, bin_id,
                                                   float(row[0]), row[col_id], header[col_id])
    return selected


def reconstruct_sparse_raw(X, header, cells):
    """Build raw rows containing only the selected row/column cells."""
    rows = _as_rows(X)
    by_row = {}
    for cell in cells:
        row_id = cell['row_id']
        if row_id not in by_row:
            new_row = ['' for _ in header]
            new_row[0] = rows[row_id][0]
            by_row[row_id] = new_row
        by_row[row_id][cell['col_id']] = cell['value']
    if not by_row:
        return np.empty((0, len(header)), dtype=object)
    out = [by_row[row_id] for row_id in sorted(by_row)]
    return np.array(out, dtype=object)


def structured_cells(X, header, sampling_interval, end=24.0):
    selected = select_last_raw_observation_per_bin(X, header, sampling_interval, end=end)
    return list(selected.values())


def structured_sample(X, header, sampling_interval, end=24.0):
    cells = structured_cells(X, header, sampling_interval, end=end)
    return reconstruct_sparse_raw(X, header, cells), cells


def _stable_seed(sampling_seed, stay_name, sampling_interval, channel):
    key = '{}|{}|{}|{}'.format(sampling_seed, stay_name, sampling_interval, channel)
    digest = hashlib.md5(key.encode('utf-8')).hexdigest()
    return int(digest[:16], 16)


def choose_matched_candidates(candidates, k, sampling_seed=DEFAULT_SAMPLING_SEED,
                              stay_name='', sampling_interval=4, channel=''):
    if k > len(candidates):
        raise ValueError('Impossible random matched sample for stay={} variable={} '
                         'K={} candidates={} interval={}'.format(
                             stay_name, channel, k, len(candidates), sampling_interval))
    if k == 0:
        return []
    if k == len(candidates):
        return list(candidates)
    rng = random.Random(_stable_seed(sampling_seed, stay_name, sampling_interval, channel))
    return [candidates[i] for i in sorted(rng.sample(range(len(candidates)), k))]


def random_matched_cells(X, header, sampling_interval, sampling_seed=DEFAULT_SAMPLING_SEED,
                         stay_name='', end=24.0):
    coarse = select_last_raw_observation_per_bin(X, header, sampling_interval, end=end)
    fine = select_last_raw_observation_per_bin(X, header, 1.0, end=end)
    selected = []
    for col_id in range(1, len(header)):
        coarse_cells = [cell for cell in coarse.values() if cell['col_id'] == col_id]
        k = len(coarse_cells)
        candidates = [cell for cell in fine.values() if cell['col_id'] == col_id]
        candidates = sorted(candidates, key=lambda cell: (cell['bin_id'], cell['row_id']))
        chosen = choose_matched_candidates(candidates, k, sampling_seed,
                                           stay_name, sampling_interval, header[col_id])
        selected.extend(chosen)
    return selected


def random_matched_sample(X, header, sampling_interval,
                          sampling_seed=DEFAULT_SAMPLING_SEED, stay_name='', end=24.0):
    cells = random_matched_cells(X, header, sampling_interval, sampling_seed,
                                 stay_name=stay_name, end=end)
    return reconstruct_sparse_raw(X, header, cells), cells


def validate_sampling_args(strategy, sampling_interval, downstream_timestep):
    if strategy not in VALID_STRATEGIES:
        raise ValueError('sampling_strategy must be one of {}'.format(', '.join(VALID_STRATEGIES)))
    if strategy == 'none':
        return
    if int(sampling_interval) != sampling_interval or int(sampling_interval) not in VALID_INTERVALS:
        raise ValueError('sampling_interval must be one of {}'.format(', '.join(map(str, VALID_INTERVALS))))
    downstream_timestep = float(downstream_timestep)
    valid_downstream = (abs(downstream_timestep - 1.0) <= DISCRETIZER_EPS or
                        abs(downstream_timestep - 0.0) <= DISCRETIZER_EPS)
    if not valid_downstream:
        raise ValueError('structured/random_matched sampling requires downstream timestep=0.0 or 1.0; '
                         'got {}'.format(downstream_timestep))


def apply_sampling_to_example(example, strategy, sampling_interval,
                              sampling_seed=DEFAULT_SAMPLING_SEED):
    if strategy == 'none':
        return example
    end = example.get('t', 24.0)
    stay_name = example.get('name', '')
    if strategy == 'structured':
        X, _ = structured_sample(example['X'], example['header'], sampling_interval, end=end)
    elif strategy == 'random_matched':
        X, _ = random_matched_sample(example['X'], example['header'], sampling_interval,
                                     sampling_seed=sampling_seed,
                                     stay_name=stay_name, end=end)
    else:
        raise ValueError('Unknown sampling_strategy {}'.format(strategy))
    ret = dict(example)
    ret['X'] = X
    return ret


class SamplingReader(object):
    def __init__(self, reader, strategy='none', sampling_interval=4,
                 sampling_seed=DEFAULT_SAMPLING_SEED):
        self._reader = reader
        self._strategy = strategy
        self._sampling_interval = sampling_interval
        self._sampling_seed = sampling_seed

    def read_example(self, index):
        ret = self._reader.read_example(index)
        return apply_sampling_to_example(ret, self._strategy, self._sampling_interval,
                                         self._sampling_seed)

    def read_next(self):
        ret = self._reader.read_next()
        return apply_sampling_to_example(ret, self._strategy, self._sampling_interval,
                                         self._sampling_seed)

    def __getattr__(self, name):
        return getattr(self._reader, name)


def nonempty_cell_count(X):
    total = 0
    for row in X:
        for value in row[1:]:
            if value != '':
                total += 1
    return total


def cell_counts_by_channel(X, header):
    counts = dict((channel, 0) for channel in header[1:])
    for row in X:
        for col_id in range(1, len(header)):
            if row[col_id] != '':
                counts[header[col_id]] += 1
    return counts


def mask_counts_by_channel(discretized, discretized_header):
    names = discretized_header.split(',') if isinstance(discretized_header, str) else discretized_header
    counts = {}
    for col_id, name in enumerate(names):
        if name.startswith('mask->'):
            counts[name[len('mask->'):]] = int(np.sum(discretized[:, col_id]))
    return counts
