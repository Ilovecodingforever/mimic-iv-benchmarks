from __future__ import absolute_import

import json
import os
import sys
import ast
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "visualizers"))

from experiment3_results import (  # noqa: E402
    CANONICAL_OUTPUTS,
    REQUIRED_RESULT_COLUMNS,
    STATUS_OUTPUTS,
    expected_prediction_path,
    build_inference_command,
    generate_experiment3_results,
    parse_log_name,
)


def _write_log(path, epochs=2):
    path = Path(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"epoch": epoch, "val_auprc": 0.1 + epoch * 0.7,
         "val_auroc": 0.2 + epoch * 0.5}
        for epoch in range(epochs)
    ]).to_csv(str(path), sep=";", index=False)


def _write_prediction(path, predictions=(0.1, 0.9)):
    path = Path(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "stay": ["1_episode1_timeseries.csv", "2_episode1_timeseries.csv"],
        "prediction": list(predictions),
        "y_true": [0, 1],
    }).to_csv(str(path), index=False)


def _write_complete_dense_run(root):
    root = Path(str(root))
    log = root / "12h" / "keras_logs" / \
        "k_lstm.n16.d0.3.dep2.h12.bs8.ts1.0.seed0.csv"
    _write_log(log)
    checkpoint = root / "12h" / "keras_states" / \
        "k_lstm.n16.d0.3.dep2.h12.bs8.ts1.0.seed0.epoch2.test0.2.state"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("synthetic checkpoint")
    return checkpoint


def test_model_name_parsing_distinguishes_raw_variants():
    assert parse_log_name(
        "raw_lstm.n16.d0.3.dep2.h12.bs8.ts0.0.td.seed0.csv")["model_key"] == \
        "lstm_raw_timestamp"
    assert parse_log_name(
        "raw_transformer.pe_timestamp.n16h4l2ff64.d0.3.h12.bs8.ts0.0.seed0.csv")["model_key"] == \
        "transformer_timestamp"
    assert parse_log_name(
        "raw_transformer.pe_index.n16h4l2ff64.d0.3.h12.bs8.ts0.0.seed0.csv")["model_key"] == \
        "transformer_index"


def test_generator_selects_checkpoint_prediction_and_metrics(tmpdir):
    root = os.path.join(str(tmpdir), "results")
    output = os.path.join(root, "plots")
    checkpoint = _write_complete_dense_run(tmpdir / "results")

    dense_prediction = expected_prediction_path(
        checkpoint, "none", None, None, "none", None, None)
    _write_prediction(dense_prediction)
    shifted_prediction = expected_prediction_path(
        checkpoint, "none", None, None, "structured", 4, None)
    _write_prediction(shifted_prediction, predictions=(0.2, 0.8))

    generated = generate_experiment3_results(
        root, output, horizons=[12], seeds=[0], expected_epochs=2)
    results = generated["lstm_grid"]["results"]
    assert set(REQUIRED_RESULT_COLUMNS).issubset(results.columns)
    dense = results[results["condition"] == "dense->dense"].iloc[0]
    assert dense["checkpoint_path"] == str(checkpoint)
    assert dense["prediction_path"] == str(dense_prediction)
    assert dense["test_auroc"] == 1.0
    shifted = results[results["condition"] == "dense->structured-r4"].iloc[0]
    assert shifted["prediction_path"] == str(shifted_prediction)
    assert shifted["test_auroc"] == 1.0

    status = generated["lstm_grid"]["status"]
    missing_r8 = status[status["condition"] == "dense->structured-r8"].iloc[0]
    assert missing_r8["status"] == "missing_prediction"
    assert pd.isna(missing_r8.get("test_auroc", np.nan))


def test_partial_runs_are_status_only_and_reruns_are_idempotent(tmpdir):
    root = tmpdir / "results"
    output = root / "plots"
    partial = root / "12h" / "keras_logs" / \
        "raw_transformer.pe_timestamp.n16h4l2ff64.d0.3.h12.bs8.ts0.0.seed0.csv"
    _write_log(partial, epochs=1)

    first = generate_experiment3_results(
        str(root), str(output), horizons=[12], seeds=[0], expected_epochs=2)
    status = first["transformer_timestamp"]["status"]
    assert (status["status"] == "incomplete_log").any()
    assert len(first["transformer_timestamp"]["results"]) == 0

    result_path = Path(str(output)) / CANONICAL_OUTPUTS["transformer_timestamp"]
    status_path = Path(str(output)) / STATUS_OUTPUTS["transformer_timestamp"]
    before = (result_path.read_bytes(), status_path.read_bytes())
    second = generate_experiment3_results(
        str(root), str(output), horizons=[12], seeds=[0], expected_epochs=2)
    after = (result_path.read_bytes(), status_path.read_bytes())
    assert before == after
    assert not second["transformer_timestamp"]["results"].duplicated().any()


def test_generator_writes_all_canonical_outputs_and_notebook_calls(tmpdir):
    root = os.path.join(str(tmpdir), "results")
    output = os.path.join(root, "plots")
    generated = generate_experiment3_results(
        root, output, horizons=[12], seeds=[0], expected_epochs=2)
    for spec, filename in CANONICAL_OUTPUTS.items():
        assert os.path.exists(os.path.join(output, filename))
        assert os.path.exists(os.path.join(output, STATUS_OUTPUTS[spec]))
        assert "model" in generated[spec]["results"].columns

    for notebook in [
        "visualize_exp3_frequency_shift.ipynb",
        "visualize_exp3_grud_frequency_shift.ipynb",
        "visualize_summary.ipynb",
    ]:
        path = os.path.join(os.path.dirname(__file__), "..", "visualizers", notebook)
        with open(path) as handle:
            source = "\n".join(
                "".join(cell.get("source", []))
                for cell in json.load(handle)["cells"]
            )
        assert "generate_experiment3_results" in source


def test_checkpoint_only_inference_command_never_uses_train_mode():
    command = build_inference_command({
        "model_key": "transformer_timestamp",
        "dim": 16, "depth": 2, "dropout": 0.3, "batch_size": 8,
        "timestep": 0.0, "horizon": 12, "model_seed": 0,
        "log_path": "/tmp/results/12h/keras_logs/run.csv",
        "checkpoint_path": "/tmp/results/12h/keras_states/run.state",
        "train_sampling_strategy": "none", "train_sampling_interval": None,
        "train_sampling_seed": None, "test_sampling_strategy": "structured",
        "test_sampling_interval": 4, "test_sampling_seed": None,
    }, "/data", "/normalizers", python_executable="python")
    assert "--mode" in command and command[command.index("--mode") + 1] == "test"
    assert "--transformer_positional_encoding" in command
    assert "train" not in command


def test_notebook_code_cells_compile_ignoring_ipython_magics():
    for notebook in [
        "visualize_exp3_frequency_shift.ipynb",
        "visualize_exp3_grud_frequency_shift.ipynb",
        "visualize_summary.ipynb",
    ]:
        path = os.path.join(os.path.dirname(__file__), "..", "visualizers", notebook)
        with open(path) as handle:
            cells = json.load(handle)["cells"]
        for index, cell in enumerate(cells):
            if cell.get("cell_type") != "code":
                continue
            source = "\n".join(
                line for line in "".join(cell.get("source", [])).splitlines()
                if not line.lstrip().startswith(("!", "%"))
            )
            ast.parse(source, filename="{}#{}".format(notebook, index))

