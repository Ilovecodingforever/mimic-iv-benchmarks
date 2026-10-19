from __future__ import absolute_import

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualize_utils import (
    aggregate_exp2a_absolute_performance,
    plot_exp2a_absolute_heatmaps,
)


def test_exp2a_notebook_has_isolated_task_configurations():
    notebook_path = os.path.join(os.path.dirname(__file__), 'visualize_exp2a_horizon.ipynb')
    with open(notebook_path) as notebook_file:
        notebook = json.load(notebook_file)
    source = '\n'.join(''.join(cell.get('source', [])) for cell in notebook['cells'])

    assert 'fixed_horizon_icu_exit' in source
    assert 'fixed_horizon_decompensation' in source
    assert 'data/decompensation' in source
    assert 'results/fixed_horizon_decompensation' in source
    assert 'metadata_path' in source
    assert 'EXP2A_TASK_CONFIG["module"]' in source

    summary_path = os.path.join(os.path.dirname(__file__), 'visualize_summary.ipynb')
    with open(summary_path) as summary_file:
        summary = json.load(summary_file)
    summary_source = '\n'.join(
        ''.join(cell.get('source', [])) for cell in summary['cells']
    )
    for notebook_source in (source, summary_source):
        assert 'Absolute Performance Heatmaps: Measurement Frequency × Prediction Horizon' in notebook_source
        assert 'plot_exp2a_absolute_heatmaps' in notebook_source
        assert 'fixed_horizon_icu_exit' in notebook_source
        assert 'fixed_horizon_decompensation' in notebook_source


def test_exp2a_heatmap_aggregation_preserves_axes_and_marks_missing_seeds():
    test_runs = pd.DataFrame([
        {"horizon": 12, "timestep": 0.0, "seed": 0, "test_auroc": 0.70},
        {"horizon": 12, "timestep": 0.0, "seed": 1, "test_auroc": 0.80},
        {"horizon": 12, "timestep": 1.0, "seed": 0, "test_auroc": 0.90},
        {"horizon": 24, "timestep": 0.0, "seed": 0, "test_auroc": 0.60},
    ])
    table = aggregate_exp2a_absolute_performance(
        test_runs, "auroc", [12, 24], [0.0, 1.0, 2.0], expected_seeds=2
    )

    assert table[["horizon", "timestep"]].values.tolist() == [
        [12, 0.0], [12, 1.0], [12, 2.0],
        [24, 0.0], [24, 1.0], [24, 2.0],
    ]
    complete = table[(table.horizon == 12) & (table.timestep == 0.0)].iloc[0]
    assert complete["mean"] == 0.75
    assert complete["n_seeds"] == 2
    assert complete["status"] == "COMPLETE"
    partial = table[(table.horizon == 24) & (table.timestep == 0.0)].iloc[0]
    assert partial["n_seeds"] == 1
    assert partial["status"] == "PARTIAL"
    missing = table[(table.horizon == 24) & (table.timestep == 2.0)].iloc[0]
    assert np.isnan(missing["mean"])
    assert missing["status"] == "NOT AVAILABLE"


def test_exp2a_heatmaps_use_shared_scale_and_ordered_missing_cells():
    rows = [
        {"metric": "auroc", "horizon": 12, "timestep": 0.0,
         "timestep_label": "Raw/irregular", "mean": 0.50,
         "n_seeds": 5, "expected_seeds": 5, "status": "COMPLETE"},
        {"metric": "auroc", "horizon": 12, "timestep": 1.0,
         "timestep_label": "1h", "mean": 0.80,
         "n_seeds": 5, "expected_seeds": 5, "status": "COMPLETE"},
    ]
    table = pd.DataFrame(rows)
    second = table.copy()
    second.loc[0, "mean"] = 0.60
    fig, scale = plot_exp2a_absolute_heatmaps(
        {"icu": table, "decomp": second}, "auroc",
        [12, 24], [0.0, 1.0],
        task_labels={"icu": "ICU exit", "decomp": "Decompensation"},
        show=False,
    )
    try:
        assert scale == {"vmin": 0.50, "vmax": 0.80}
        assert fig.axes[0].images[0].get_clim() == fig.axes[1].images[0].get_clim()
        assert fig.axes[2].get_position().x0 > fig.axes[1].get_position().x1
        assert [label.get_text() for label in fig.axes[0].get_xticklabels()] == [
            "Raw/irregular", "1h"
        ]
        assert [label.get_text() for label in fig.axes[0].get_yticklabels()] == [
            "12h", "24h"
        ]
        assert any(text.get_text() == "N/A" for text in fig.axes[0].texts)
    finally:
        plt.close(fig)


if __name__ == '__main__':
    test_exp2a_notebook_has_isolated_task_configurations()
    print('ok')
