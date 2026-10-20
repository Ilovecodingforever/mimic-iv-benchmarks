from __future__ import absolute_import

import json
import os

import numpy as np
import pandas as pd

from visualize_utils import (
    compute_los_prediction_metrics, compute_seed_paired_drop,
    read_validation_log)


def test_los_prediction_metrics_use_benchmark_columns_and_definitions(tmpdir):
    path = os.path.join(str(tmpdir), 'predictions.csv')
    pd.DataFrame([
        {'stay': '1_episode1_timeseries.csv', 'period_length': 4.0,
         'prediction': 24.0, 'y_true': 24.0},
        {'stay': '2_episode1_timeseries.csv', 'period_length': 8.0,
         'prediction': 48.0, 'y_true': 72.0},
    ]).to_csv(path, index=False)
    metrics = compute_los_prediction_metrics(path)
    assert metrics['test_kappa'] <= 1.0
    assert metrics['test_mad'] == 12.0


def test_los_degradation_direction_is_metric_specific():
    rows = pd.DataFrame([
        {'condition': '1h', 'seed': 0, 'test_kappa': 0.8, 'test_mad': 10.0},
        {'condition': '4h', 'seed': 0, 'test_kappa': 0.7, 'test_mad': 12.0},
    ])
    kappa = compute_seed_paired_drop(
        rows, [], 'condition', '1h', 'test_kappa',
        drop_col='drop', higher_is_better=True)
    mad = compute_seed_paired_drop(
        rows, [], 'condition', '1h', 'test_mad',
        drop_col='drop', higher_is_better=False)
    assert np.isclose(kappa.loc[kappa['condition'] == '4h', 'drop'].iloc[0], 0.1)
    assert np.isclose(mad.loc[mad['condition'] == '4h', 'drop'].iloc[0], 2.0)


def test_los_validation_selection_uses_kappa(tmpdir):
    path = os.path.join(str(tmpdir), 'los.csv')
    pd.DataFrame([
        {'epoch': 0, 'val_kappa': 0.4, 'val_mad': 20.0},
        {'epoch': 1, 'val_kappa': 0.6, 'val_mad': 22.0},
    ]).to_csv(path, sep=';', index=False)
    selected = read_validation_log(
        path, selection_metric='val_kappa',
        required_metrics=('val_kappa', 'val_mad'))
    assert selected['best_csv_epoch'] == 1
    assert selected['val_kappa'] == 0.6
    assert selected['val_mad'] == 22.0


def test_exp1_notebook_contains_raw_los_analysis():
    notebook_path = os.path.join(
        os.path.dirname(__file__), 'visualize_exp1_frequency.ipynb')
    with open(notebook_path) as notebook_file:
        notebook = json.load(notebook_file)
    source = '\n'.join(
        ''.join(cell.get('source', [])) for cell in notebook['cells'])
    assert 'mimic4models.length_of_stay.main' in source
    assert 'los_raw_ts:0.00_observed_only_n:' in source
    assert 'test_kappa' in source
    assert 'test_mad' in source
    assert 'higher_is_better=False' in source


if __name__ == '__main__':
    print('ok')
