"""Artifact-only Experiment 3 run discovery and CSV export.

This module deliberately stops at the selected checkpoint and an existing test
prediction.  It never treats an incomplete training run as a completed result.
"""
from __future__ import absolute_import

import argparse
import os
import re
import subprocess
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .visualize_utils import (
        find_selected_checkpoint,
        load_prediction_csv,
        prediction_metrics,
        read_validation_log,
    )
except ImportError:
    from visualize_utils import (
        find_selected_checkpoint,
        load_prediction_csv,
        prediction_metrics,
        read_validation_log,
    )


CANONICAL_OUTPUTS = OrderedDict([
    ("lstm_grid", "experiment3_lstm_condition_run_results.csv"),
    ("lstm_raw", "experiment3e_lstm_raw_condition_run_results.csv"),
    ("grud_grid", "experiment3_grud_condition_run_results.csv"),
    ("grud_raw", "experiment3e_grud_raw_run_results.csv"),
    ("transformer_timestamp", "experiment3e_transformer_timestamp_raw_condition_run_results.csv"),
    ("transformer_index", "experiment3e_transformer_index_raw_condition_run_results.csv"),
    ("lstm_raw_timestamp", "experiment3e_lstm_timestamp_raw_condition_run_results.csv"),
])

STATUS_OUTPUTS = OrderedDict(
    (key, value.replace("_run_results.csv", "_run_status.csv"))
    for key, value in CANONICAL_OUTPUTS.items()
)

REQUIRED_RESULT_COLUMNS = [
    "model", "representation", "experiment", "horizon", "target_r",
    "condition", "model_seed", "train_sampling_strategy",
    "train_sampling_interval", "train_sampling_seed",
    "test_sampling_strategy", "test_sampling_interval", "test_sampling_seed",
    "normalizer_regime", "timestep", "log_path", "selected_val_epoch",
    "selected_val_auprc", "selected_val_auroc", "checkpoint_path",
    "checkpoint_name", "expected_prediction_path", "prediction_path",
    "prediction_status", "prediction_exists", "prediction_name_mode",
    "network", "run_kind", "n_epochs", "complete", "validation_auprc",
    "validation_auroc", "missing_reason",
    "test_auroc", "test_auprc", "test_brier", "number_of_stays",
    "number_of_patients",
]

_MODEL_METADATA = {
    "lstm_grid": {
        "model": "LSTM", "representation": "fixed_1h_grid",
        "raw": False, "prefix": "k_lstm", "timestamp_mode": None,
    },
    "lstm_raw": {
        "model": "LSTM", "representation": "raw_irregular",
        "raw": True, "prefix": "raw_lstm", "timestamp_mode": "none",
    },
    "lstm_raw_timestamp": {
        "model": "LSTM + timestamp", "representation": "raw_irregular_timestamp_delta",
        "raw": True, "prefix": "raw_lstm", "timestamp_mode": "delta",
    },
    "grud_grid": {
        "model": "GRU-D", "representation": "fixed_1h_grid",
        "raw": False, "prefix": "k_grud", "timestamp_mode": None,
    },
    "grud_raw": {
        "model": "GRU-D", "representation": "raw_irregular",
        "raw": True, "prefix": "k_grud", "timestamp_mode": None,
    },
    "transformer_timestamp": {
        "model": "Transformer (timestamp)", "representation": "raw_irregular_timestamp_pe",
        "raw": True, "prefix": "raw_transformer", "timestamp_mode": "timestamp",
    },
    "transformer_index": {
        "model": "Transformer (index)", "representation": "raw_irregular_index_pe",
        "raw": True, "prefix": "raw_transformer", "timestamp_mode": "index",
    },
}

_NETWORKS = {
    "lstm_grid": "mimic4models/keras_models/lstm.py",
    "lstm_raw": "mimic4models/keras_models/raw_lstm.py",
    "lstm_raw_timestamp": "mimic4models/keras_models/raw_lstm.py",
    "grud_grid": "mimic4models/keras_models/grud.py",
    "grud_raw": "mimic4models/keras_models/grud.py",
    "transformer_timestamp": "mimic4models/keras_models/raw_transformer.py",
    "transformer_index": "mimic4models/keras_models/raw_transformer.py",
}


def parse_sampling_from_name(name):
    match = re.search(
        r"\.sample(?P<strategy>structured|random_matched)"
        r"\.r(?P<interval>[0-9]+)(?:\.sseed(?P<seed>[0-9]+))?",
        str(name),
    )
    if match is None:
        return "none", None, None
    seed = match.group("seed")
    return (
        match.group("strategy"),
        int(match.group("interval")),
        int(seed) if seed is not None else None,
    )


def _number(name, token, cast=float, default=None):
    match = re.search(r"\.{}([0-9]+(?:\.[0-9]+)?)(?=\.|$)".format(token), name)
    if match is None:
        return default
    return cast(match.group(1))


def parse_log_name(log_path):
    """Parse the stable experiment fields encoded in a Keras log filename."""
    path = Path(log_path)
    name = path.name
    if name.startswith("raw_transformer"):
        mode_match = re.search(r"\.pe_(timestamp|index)\.", name)
        if mode_match is None:
            return None
        model_key = "transformer_{}".format(mode_match.group(1))
    elif name.startswith("raw_lstm"):
        model_key = "lstm_raw_timestamp" if ".td." in name else "lstm_raw"
    elif name.startswith("k_lstm"):
        model_key = "lstm_grid"
    elif name.startswith("k_grud"):
        model_key = "grud_raw" if _number(name, "ts", float) == 0.0 else "grud_grid"
    else:
        return None

    strategy, interval, sampling_seed = parse_sampling_from_name(name)
    mode = _MODEL_METADATA[model_key]
    return {
        "model_key": model_key,
        "model": mode["model"],
        "representation": mode["representation"],
        "filename": name,
        "log_path": str(path),
        "horizon": _number(name, "h", int),
        "timestep": _number(name, "ts", float),
        "model_seed": _number(name, "seed", int),
        "dim": _number(name, "n", int),
        "depth": _number(name, "dep", int),
        "dropout": _number(name, "d", float),
        "batch_size": _number(name, "bs", int),
        "train_sampling_strategy": strategy,
        "train_sampling_interval": interval,
        "train_sampling_seed": sampling_seed,
        "timestamp_mode": mode["timestamp_mode"],
        "raw": mode["raw"],
    }


def _effective_regime(strategy, interval, seed):
    if strategy == "none":
        return ("none",)
    if strategy == "structured":
        return ("structured", int(interval))
    return ("random_matched", int(interval), int(seed))


def sampling_suffix(strategy, interval, seed):
    if strategy == "none":
        return "none"
    value = "{}-r{}".format(strategy, int(interval))
    if strategy == "random_matched":
        value += "-sseed{}".format(int(seed))
    return value


def expected_prediction_path(checkpoint, train_strategy, train_interval, train_seed,
                             test_strategy, test_interval, test_seed):
    """Return the exact filename written by fixed_horizon_icu_exit.main."""
    checkpoint = Path(checkpoint)
    output_dir = checkpoint.parent.parent
    base = output_dir / "test_predictions" / checkpoint.name
    if _effective_regime(train_strategy, train_interval, train_seed) == _effective_regime(
            test_strategy, test_interval, test_seed):
        return Path(str(base) + ".csv")
    return Path(str(base) + ".testsample-{}.csv".format(
        sampling_suffix(test_strategy, test_interval, test_seed)))


def _normalizer_regime(log_path, raw=False, override=None):
    if override is not None:
        return override
    if "structured_r4_dense_norm_control" in str(log_path):
        return "dense"
    return "raw_train_matched" if raw else "train_matched"


def _experiment_name(horizon, condition, raw):
    if raw:
        return "Exp3E Raw-representation frequency shift"
    if horizon == 96:
        return "Exp3D Horizon replication"
    if "normalizer" in condition:
        return "Exp3B Normalizer control"
    if "random" in condition:
        return "Exp3C Random-matched frequency shift"
    return "Exp3A Structured frequency shift"


def _conditions(horizon, raw):
    if raw:
        if horizon != 12:
            return []
        return [
            (4, "none", None, None, "structured", 4, None,
             "raw-dense->raw-structured-r4", "raw_dense"),
            (4, "structured", 4, None, "structured", 4, None,
             "raw-structured-r4->raw-structured-r4", "raw_structured"),
            (8, "none", None, None, "structured", 8, None,
             "raw-dense->raw-structured-r8", "raw_dense"),
            (8, "structured", 8, None, "structured", 8, None,
             "raw-structured-r8->raw-structured-r8", "raw_structured"),
        ]

    rows = []
    if horizon == 12:
        rows.append((1, "none", None, None, "none", None, None,
                     "dense->dense", "dense"))
        for r in (4, 8):
            rows.extend([
                (r, "none", None, None, "structured", r, None,
                 "dense->structured-r{}".format(r), "dense"),
                (r, "structured", r, None, "structured", r, None,
                 "structured-r{}->structured-r{}".format(r, r), "structured"),
            ])
        rows.extend([
            (4, "structured", 4, None, "structured", 4, None,
             "structured-r4 train + dense norm", "structured_r4_dense_norm_control"),
            (4, "structured", 4, None, "structured", 4, None,
             "structured-r4 train + train-matched norm", "structured"),
            (4, "none", None, None, "random_matched", 4, 100,
             "dense->random-r4", "dense"),
            (4, "random_matched", 4, 100, "random_matched", 4, 100,
             "random-r4->random-r4", "random_matched"),
        ])
    elif horizon == 96:
        rows.extend([
            (4, "none", None, None, "structured", 4, None,
             "dense->structured-r4", "dense"),
            (4, "structured", 4, None, "structured", 4, None,
             "structured-r4->structured-r4", "structured"),
        ])
    return rows


def _log_matches(parsed, model_key, horizon, seed, strategy, interval,
                 sampling_seed, normalizer_kind):
    if parsed is None or parsed["model_key"] != model_key:
        return False
    if parsed.get("horizon") != int(horizon) or parsed.get("model_seed") != int(seed):
        return False
    expected_timestep = 0.0 if _MODEL_METADATA[model_key]["raw"] else 1.0
    if parsed.get("timestep") != expected_timestep:
        return False
    if parsed.get("train_sampling_strategy") != strategy:
        return False
    if strategy == "none":
        if parsed.get("train_sampling_interval") is not None:
            return False
    elif parsed.get("train_sampling_interval") != int(interval):
        return False
    if strategy == "random_matched" and parsed.get("train_sampling_seed") != int(sampling_seed):
        return False
    path = parsed["log_path"].replace(os.sep, "/")
    horizon_dir = "/{}h/keras_logs/".format(int(horizon))
    if not _MODEL_METADATA[model_key]["raw"]:
        if normalizer_kind == "structured_r4_dense_norm_control":
            if "structured_r4_dense_norm_control{}".format(horizon_dir) not in path:
                return False
        elif strategy == "none":
            if horizon_dir not in path:
                return False
        elif strategy == "structured":
            expected = "/structured/{}/{}h/keras_logs/".format(
                "{}h".format(int(horizon)), int(interval))
            if expected not in path:
                return False
        elif strategy == "random_matched":
            expected = "/random_matched/{}/{}h/keras_logs/".format(
                "{}h".format(int(horizon)), int(interval))
            if expected not in path:
                return False
    if normalizer_kind == "structured_r4_dense_norm_control":
        return "structured_r4_dense_norm_control" in parsed["log_path"]
    if normalizer_kind == "structured":
        return "structured_r4_dense_norm_control" not in parsed["log_path"]
    return True


def discover_logs(results_dir):
    results_dir = Path(results_dir)
    rows = []
    # ponytail: one recursive scan keeps legacy result layouts supported; strict
    # model, timestep, horizon, and directory filters reject stale artifacts.
    for path in sorted(results_dir.rglob("keras_logs/*.csv")):
        parsed = parse_log_name(path)
        if parsed is not None:
            rows.append(parsed)
    return rows


def _empty_row(spec, horizon, seed, condition, target_r):
    metadata = _MODEL_METADATA[spec]
    return {
        "model": metadata["model"],
        "representation": metadata["representation"],
        "experiment": _experiment_name(horizon, condition, metadata["raw"]),
        "horizon": horizon,
        "target_r": target_r,
        "condition": condition,
        "model_seed": seed,
        "status": "missing_log",
        "prediction_status": "missing",
        "missing_reason": None,
        "model_key": spec,
        "network": _NETWORKS[spec],
        "prediction_exists": False,
    }


def _select_log(logs, spec, horizon, seed, train_strategy, train_interval,
                train_seed, normalizer_kind):
    candidates = [
        row for row in logs
        if _log_matches(row, spec, horizon, seed, train_strategy, train_interval,
                        train_seed, normalizer_kind)
    ]
    return candidates


def _status_for_plan(logs, spec, horizon, seed, plan, expected_epochs):
    (target_r, train_strategy, train_interval, train_seed, test_strategy,
     test_interval, test_seed, condition, normalizer_kind) = plan
    row = _empty_row(spec, horizon, seed, condition, target_r)
    metadata = _MODEL_METADATA[spec]
    row.update({
        "train_sampling_strategy": train_strategy,
        "train_sampling_interval": train_interval,
        "train_sampling_seed": train_seed,
        "test_sampling_strategy": test_strategy,
        "test_sampling_interval": test_interval,
        "test_sampling_seed": test_seed,
        "normalizer_regime": _normalizer_regime("", metadata["raw"],
                                                 "dense" if normalizer_kind == "structured_r4_dense_norm_control" else None),
        "timestep": 0.0 if metadata["raw"] else 1.0,
    })
    candidates = _select_log(logs, spec, horizon, seed, train_strategy,
                             train_interval, train_seed, normalizer_kind)
    if len(candidates) == 0:
        row["missing_reason"] = "missing training log"
        return row, None
    if len(candidates) > 1:
        row["status"] = "ambiguous_log"
        row["missing_reason"] = "multiple matching training logs"
        return row, None

    parsed = candidates[0]
    row.update(parsed)
    row["normalizer_regime"] = _normalizer_regime(
        parsed["log_path"], metadata["raw"],
        "dense" if normalizer_kind == "structured_r4_dense_norm_control" else None)
    log_path = Path(parsed["log_path"])
    selected = read_validation_log(log_path, expected_epochs=expected_epochs)
    if selected is None:
        row["status"] = "incomplete_log"
        row["missing_reason"] = "validation log is empty or lacks required metrics"
        return row, None
    row.update({
        "n_epochs": selected["n_epochs"],
        "complete": selected["complete"],
        "selected_val_epoch": selected["selected_validation_epoch"],
        "selected_val_auprc": selected["validation_auprc"],
        "selected_val_auroc": selected["validation_auroc"],
        "validation_auprc": selected["validation_auprc"],
        "validation_auroc": selected["validation_auroc"],
    })
    if not selected["complete"]:
        row["status"] = "incomplete_log"
        row["missing_reason"] = "validation log did not contain the complete epoch set"
        return row, None

    try:
        checkpoint = find_selected_checkpoint(log_path, selected["selected_validation_epoch"])
    except RuntimeError as exc:
        row["status"] = "ambiguous_checkpoint"
        row["missing_reason"] = str(exc)
        return row, None
    if checkpoint is None:
        row["status"] = "missing_checkpoint"
        row["missing_reason"] = "selected checkpoint does not exist"
        return row, None

    expected = expected_prediction_path(
        checkpoint, train_strategy, train_interval, train_seed,
        test_strategy, test_interval, test_seed)
    row["checkpoint_path"] = str(checkpoint)
    row["checkpoint_name"] = checkpoint.name
    row["expected_prediction_path"] = str(expected)
    if not expected.exists():
        row["status"] = "missing_prediction"
        row["missing_reason"] = "selected-checkpoint test prediction does not exist"
        return row, None
    try:
        pred = load_prediction_csv(expected)
        metrics = prediction_metrics(pred)
    except Exception as exc:
        row["status"] = "invalid_prediction"
        row["missing_reason"] = str(exc)
        return row, None
    row.update(metrics)
    row["prediction_path"] = str(expected)
    row["prediction_status"] = "found"
    row["prediction_exists"] = True
    row["prediction_name_mode"] = (
        "matched-original-filename"
        if _effective_regime(train_strategy, train_interval, train_seed) ==
        _effective_regime(test_strategy, test_interval, test_seed)
        else "explicit-testsample-suffix"
    )
    row["status"] = "found"
    row["missing_reason"] = None
    return row, row


def _write_csv_atomic(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".{}.".format(path.name), dir=str(path.parent))
    os.close(fd)
    try:
        df.to_csv(temporary, index=False)
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _deduplicate(df):
    if len(df) == 0:
        return df
    keys = ["model_key", "horizon", "target_r", "condition", "model_seed"]
    keys = [key for key in keys if key in df.columns]
    return df.sort_values(keys).drop_duplicates(keys, keep="last").reset_index(drop=True)


def generate_experiment3_results(results_dir, output_dir=None, horizons=(12, 96),
                                 seeds=(0, 1, 2, 3, 4), expected_epochs=100,
                                 run_missing_inference=False, inference_args=None):
    """Generate all canonical Experiment 3 run/status CSVs.

    ``run_missing_inference`` is intentionally opt-in.  It invokes the existing
    fixed-horizon test command only when callers provide a complete command
    builder in ``inference_args``; this function never trains a model.
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir is not None else results_dir / "plots"
    logs = discover_logs(results_dir)
    generated = {}
    for spec, filename in CANONICAL_OUTPUTS.items():
        status_rows = []
        result_rows = []
        for horizon in sorted(set(int(value) for value in horizons)):
            for seed in sorted(set(int(value) for value in seeds)):
                for plan in _conditions(horizon, _MODEL_METADATA[spec]["raw"]):
                    status, result = _status_for_plan(
                        logs, spec, horizon, seed, plan, expected_epochs)
                    status_rows.append(status)
                    if result is not None:
                        result_rows.append(result)

        status_df = _deduplicate(pd.DataFrame(status_rows))
        result_df = _deduplicate(pd.DataFrame(result_rows))
        for column in REQUIRED_RESULT_COLUMNS:
            if column not in result_df.columns:
                result_df[column] = np.nan
        ordered = REQUIRED_RESULT_COLUMNS + [
            col for col in result_df.columns if col not in REQUIRED_RESULT_COLUMNS
        ]
        result_df = result_df[ordered]
        _write_csv_atomic(result_df, output_dir / filename)
        _write_csv_atomic(status_df, output_dir / STATUS_OUTPUTS[spec])
        if len(status_df) and (status_df["status"] != "found").any():
            print(
                "Experiment 3 {}: {} of {} planned rows are unavailable; see {}".format(
                    spec,
                    int((status_df["status"] != "found").sum()),
                    len(status_df),
                    output_dir / STATUS_OUTPUTS[spec],
                )
            )
        generated[spec] = {
            "results": result_df,
            "status": status_df,
            "result_path": str(output_dir / filename),
            "status_path": str(output_dir / STATUS_OUTPUTS[spec]),
        }

    manifest = []
    for spec, values in generated.items():
        counts = values["status"].get("status", pd.Series(dtype=str)).value_counts()
        manifest.append({
            "model_key": spec,
            "result_path": values["result_path"],
            "status_path": values["status_path"],
            "result_rows": len(values["results"]),
            "planned_rows": len(values["status"]),
            "found_rows": int(counts.get("found", 0)),
            "missing_rows": int(len(values["status"]) - counts.get("found", 0)),
        })
    _write_csv_atomic(pd.DataFrame(manifest), output_dir / "experiment3_csv_generation_manifest.csv")

    if run_missing_inference:
        if not inference_args:
            raise ValueError("--run-missing-inference requires inference_args")
        _run_missing_inference(generated, inference_args)
        return generate_experiment3_results(
            results_dir, output_dir, horizons, seeds, expected_epochs,
            run_missing_inference=False)
    return generated


def _run_missing_inference(generated, inference_args):
    """Run explicitly supplied checkpoint-only inference commands.

    The caller supplies a callable so this module cannot accidentally invent a
    training command or silently select a different model configuration.
    """
    builder = inference_args.get("command_builder")
    if builder is None or not callable(builder):
        raise ValueError("inference_args.command_builder must be callable")
    for values in generated.values():
        missing = values["status"]
        missing = missing[missing["status"] == "missing_prediction"]
        for _, row in missing.iterrows():
            command = builder(row)
            if not command:
                continue
            subprocess.check_call(command)


def build_inference_command(row, data, normalizer_dir, python_executable=None,
                            normalizer_state=None):
    """Build a checkpoint-only fixed-horizon test command for one row."""
    model_key = row["model_key"]
    command = [
        python_executable or sys.executable,
        "-m", "mimic4models.fixed_horizon_icu_exit.main",
        "--network", _NETWORKS[model_key],
        "--dim", str(int(row.get("dim") or 16)),
        "--depth", str(int(row.get("depth") or (1 if model_key.startswith("grud") else 2))),
        "--dropout", str(float(row.get("dropout") or 0.3)),
        "--rec_dropout", "0.0",
        "--batch_size", str(int(row.get("batch_size") or 8)),
        "--mode", "test",
        "--timestep", str(float(row["timestep"])),
        "--horizon", str(int(row["horizon"])),
        "--seed", str(int(row["model_seed"])),
        "--data", str(data),
        "--normalizer_dir", str(normalizer_dir),
        "--output_dir", str(Path(row["log_path"]).parent.parent),
        "--load_state", str(row["checkpoint_path"]),
        "--sampling_strategy", str(row["train_sampling_strategy"]),
        "--test_sampling_strategy", str(row["test_sampling_strategy"]),
    ]
    if row.get("train_sampling_interval") is not None and not pd.isna(row.get("train_sampling_interval")):
        command += ["--sampling_interval", str(float(row["train_sampling_interval"]))]
    if row.get("train_sampling_seed") is not None and not pd.isna(row.get("train_sampling_seed")):
        command += ["--sampling_seed", str(int(row["train_sampling_seed"]))]
    if row.get("test_sampling_interval") is not None and not pd.isna(row.get("test_sampling_interval")):
        command += ["--test_sampling_interval", str(float(row["test_sampling_interval"]))]
    if row.get("test_sampling_seed") is not None and not pd.isna(row.get("test_sampling_seed")):
        command += ["--test_sampling_seed", str(int(row["test_sampling_seed"]))]
    if normalizer_state:
        command += ["--normalizer_state", str(normalizer_state)]
    if model_key == "lstm_raw_timestamp":
        command.append("--use_raw_timestamps")
    elif model_key.startswith("transformer_"):
        command += ["--transformer_positional_encoding", model_key.split("_", 1)[1]]
    return command


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--horizons", nargs="+", type=int, default=[12, 96])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--expected_epochs", type=int, default=100)
    parser.add_argument("--data", default=None)
    parser.add_argument("--normalizer_dir", default=None)
    parser.add_argument("--normalizer_state", default=None)
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument(
        "--run-missing-inference", action="store_true",
        help="Run existing-checkpoint test inference for missing predictions; never trains.",
    )
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.run_missing_inference:
        if not args.data or not args.normalizer_dir:
            raise SystemExit("--run-missing-inference requires --data and --normalizer_dir")
        inference_args = {
            "command_builder": lambda row: build_inference_command(
                row, args.data, args.normalizer_dir, args.python_executable,
                args.normalizer_state),
        }
    else:
        inference_args = None
    generated = generate_experiment3_results(
        args.results_dir, args.output_dir, args.horizons, args.seeds,
        args.expected_epochs, args.run_missing_inference, inference_args)
    for spec, values in generated.items():
        print("{}: {} found / {} planned -> {}".format(
            spec, len(values["results"]), len(values["status"]), values["result_path"]))


if __name__ == "__main__":
    main()
