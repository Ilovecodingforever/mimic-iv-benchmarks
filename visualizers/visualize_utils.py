from __future__ import absolute_import
from __future__ import print_function

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.metrics import auc, brier_score_loss, precision_recall_curve, roc_auc_score

try:
    from IPython.display import display
except Exception:
    display = None

DEFAULT_EXPECTED_EPOCHS = 100
DEFAULT_BOOTSTRAP_SEED = 12345


def float_token_pattern(token):
    return r"\.{}([0-9]+(?:\.[0-9]+)?)(?=\.|$)".format(token)


def clean_complete_log(df, expected_epochs=DEFAULT_EXPECTED_EPOCHS):
    return validate_epoch_set(df, expected_epochs=expected_epochs)


def show_table(title, df, max_rows=80):
    print("\n" + title)
    print("=" * len(title))
    if df is None or len(df) == 0:
        print("<none>")
        return
    view = df.head(max_rows)
    if display is not None:
        display(view)
        if len(df) > max_rows:
            print("... {} more rows".format(len(df) - max_rows))
    else:
        print(view.to_string(index=False))


def validate_epoch_set(df, expected_epochs=DEFAULT_EXPECTED_EPOCHS):
    """Require the exact CSVLogger epoch set 0..expected_epochs-1."""
    if "epoch" not in df.columns:
        return False
    epoch_values = pd.to_numeric(df["epoch"], errors="coerce")
    if epoch_values.isnull().any():
        return False
    epochs = epoch_values.astype(int).tolist()
    expected = list(range(expected_epochs))
    return len(epochs) == expected_epochs and epochs == expected


def read_validation_log(log_path, only_complete=False, expected_epochs=DEFAULT_EXPECTED_EPOCHS):
    try:
        df = pd.read_csv(str(log_path), sep=";")
    except pd.errors.EmptyDataError:
        return None

    if len(df) == 0:
        return None
    complete = validate_epoch_set(df, expected_epochs=expected_epochs)
    if only_complete and not complete:
        return None
    valid = df.copy()
    for col in ["epoch", "val_auprc", "val_auroc"]:
        valid[col] = pd.to_numeric(valid[col], errors="coerce")
    valid = valid.dropna(subset=["epoch", "val_auprc", "val_auroc"])
    if len(valid) == 0:
        return None
    best_idx = valid["val_auprc"].idxmax()
    best = valid.loc[best_idx]
    best_csv_epoch = int(best["epoch"])
    return {
        "n_epochs": int(len(df)),
        "complete": bool(complete),
        "best_csv_epoch": best_csv_epoch,
        "selected_validation_epoch": best_csv_epoch,
        "checkpoint_epoch": best_csv_epoch + 1,
        "validation_auprc": float(best["val_auprc"]),
        "validation_auroc": float(best["val_auroc"]),
        # Backward-compatible aliases used by old notebook code.
        "best_val_epoch": best_csv_epoch,
        "best_val_auprc": float(best["val_auprc"]),
        "val_auroc_at_best_auprc": float(best["val_auroc"]),
        "val_auprc": float(best["val_auprc"]),
        "val_auroc": float(best["val_auroc"]),
    }


def find_selected_checkpoint(log_path, csv_epoch):
    """Find the checkpoint for the validation-AUPRC-selected CSVLogger epoch."""
    state_dir = log_path.parent.parent / "keras_states"
    checkpoint_epoch = int(csv_epoch) + 1
    candidates = sorted(state_dir.glob("{}.epoch{}.test*.state".format(log_path.stem, checkpoint_epoch)))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        return None
    raise RuntimeError("Multiple checkpoints found for {} epoch {}:\n{}".format(
        log_path.name, checkpoint_epoch, candidates))


def compute_auprc(y, pred):
    precision, recall, _ = precision_recall_curve(y, pred)
    return auc(recall, precision)


def compute_metric(y, pred, metric):
    y = np.asarray(y)
    pred = np.asarray(pred)
    if metric == "auroc":
        if len(np.unique(y)) < 2:
            return np.nan
        return roc_auc_score(y, pred)
    if metric == "auprc":
        return compute_auprc(y, pred)
    if metric == "brier":
        return brier_score_loss(y, pred)
    raise ValueError("Unknown metric: {}".format(metric))


def extract_patient_id(stay):
    stay = str(stay)
    m = re.match(r"^(\d+)_episode", stay)
    if m:
        return m.group(1)
    return stay.split("_")[0]


def load_prediction_csv(path):
    df = pd.read_csv(str(path))
    required = {"stay", "prediction", "y_true"}
    if not required.issubset(df.columns):
        raise ValueError("Prediction file missing required columns {}: {}".format(required, path))
    df = df.copy()
    df["patient_id"] = df["stay"].apply(extract_patient_id)
    return df


def prediction_metrics(df):
    y = df["y_true"].values
    pred = df["prediction"].values
    return {
        "test_auroc": compute_metric(y, pred, "auroc"),
        "test_auprc": compute_metric(y, pred, "auprc"),
        "test_brier": compute_metric(y, pred, "brier"),
        "number_of_stays": int(len(df)),
        "number_of_patients": int(df["patient_id"].nunique()),
    }


def summarize_with_t_ci(df, group_cols, metric):
    rows = []
    group_cols = list(group_cols)
    columns = group_cols + ["mean", "sd", "n", "sem", "ci95", "ci95_low", "ci95_high"]
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=columns)
    for keys, group in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = pd.to_numeric(group[metric], errors="coerce").dropna().astype(float).values
        n = len(values)
        mean = float(np.mean(values)) if n else np.nan
        sd = float(np.std(values, ddof=1)) if n > 1 else np.nan
        sem = sd / np.sqrt(n) if n > 1 else np.nan
        ci95 = float(t.ppf(0.975, n - 1) * sem) if n > 1 else np.nan
        row = dict(zip(group_cols, keys))
        row.update({
            "mean": mean,
            "sd": sd,
            "n": int(n),
            "sem": sem,
            "ci95": ci95,
            "ci95_low": mean - ci95 if n > 1 else np.nan,
            "ci95_high": mean + ci95 if n > 1 else np.nan,
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)




def _input_window_array(example):
    X = example["X"]
    hours = np.asarray(X[:, 0], dtype=float)
    return X[hours <= float(example["t"]) + 1e-7]


def _summarize_values(values, prefix):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {
            "mean_{}".format(prefix): np.nan,
            "median_{}".format(prefix): np.nan,
            "p25_{}".format(prefix): np.nan,
            "p75_{}".format(prefix): np.nan,
            "p95_{}".format(prefix): np.nan,
            "min_{}".format(prefix): np.nan,
            "max_{}".format(prefix): np.nan,
        }
    return {
        "mean_{}".format(prefix): float(np.mean(values)),
        "median_{}".format(prefix): float(np.percentile(values, 50)),
        "p25_{}".format(prefix): float(np.percentile(values, 25)),
        "p75_{}".format(prefix): float(np.percentile(values, 75)),
        "p95_{}".format(prefix): float(np.percentile(values, 95)),
        "min_{}".format(prefix): int(np.min(values)),
        "max_{}".format(prefix): int(np.max(values)),
    }


def observed_mask_information(discretized, discretized_header):
    names = discretized_header.split(",") if isinstance(discretized_header, str) else list(discretized_header)
    mask_columns = [
        (col_id, name[len("mask->"):])
        for col_id, name in enumerate(names)
        if name.startswith("mask->")
    ]
    if not mask_columns:
        raise RuntimeError("No mask columns found in post-transform representation")
    mask = discretized[:, [col_id for col_id, _ in mask_columns]]
    channel_counts = {
        channel: int(np.sum(discretized[:, col_id]))
        for col_id, channel in mask_columns
    }
    return {
        "token_count": int(np.sum(mask)),
        "occupied_1h_bins": int(np.sum(np.sum(mask, axis=1) > 0)),
        "channels_observed": int(sum(1 for value in channel_counts.values() if value > 0)),
        "channel_counts": channel_counts,
    }


def characterize_reader_input_information(condition_readers, dense_condition="dense"):
    """Characterize post-discretization observed information before/after sampling.

    condition_readers is a list of (condition, r, reader) tuples. The first
    column of each reader example must be Hours, and all readers must expose the
    same stays in the same order. Retained fractions use dense raw-event masks
    as the denominator, so fixed-grid dense is not mechanically 1.0.
    """
    from mimic4models.fixed_horizon_icu_exit.raw import RawSequenceEncoder
    from mimic4models.preprocessing import Discretizer

    discretizer = Discretizer(timestep=1.0, store_masks=True,
                              impute_strategy="previous", start_time="zero")
    raw_encoder = RawSequenceEncoder()
    stay_level_parts = []
    channel_rows = []
    expected_names = None
    raw_tokens_by_name = None
    raw_channel_totals_reference = None
    total_channels = None

    for condition, r, reader in condition_readers:
        rows = []
        channel_totals = None
        names = []
        for index in range(reader.get_number_of_examples()):
            example = reader.read_example(index)
            if example["header"][0] != "Hours":
                raise RuntimeError("Expected Hours as first raw column for {}".format(condition))
            X = _input_window_array(example)
            if X.shape[0] == 0:
                raise RuntimeError("No rows in 24h input window for stay {} under {}".format(
                    example["name"], condition))
            hours = np.asarray(X[:, 0], dtype=float)
            if np.any(hours < -1e-7) or np.any(hours > float(example["t"]) + 1e-7):
                raise RuntimeError("Timestamp outside 24h input window for stay {} under {}".format(
                    example["name"], condition))
            names.append(example["name"])
            total_channels = len(example["header"]) - 1
            raw_encoded, raw_encoded_header = raw_encoder.transform(
                X, header=example["header"], end=float(example["t"]))
            raw_info = observed_mask_information(raw_encoded, raw_encoded_header)
            raw_token_count = raw_info["token_count"]
            discretized, discretized_header = discretizer.transform(
                X, header=example["header"], end=float(example["t"]))
            post_info = observed_mask_information(discretized, discretized_header)
            token_count = post_info["token_count"]
            occupied_bins = post_info["occupied_1h_bins"]
            channel_counts = post_info["channel_counts"]
            channels_observed = post_info["channels_observed"]
            if channel_totals is None:
                channel_totals = dict((channel, 0) for channel in channel_counts)
            for channel, value in channel_counts.items():
                channel_totals[channel] += int(value)
            post_len = int(discretized.shape[0])
            if occupied_bins > 24:
                raise RuntimeError("Occupied 1h bins > 24 for stay {} under {}".format(
                    example["name"], condition))
            if channels_observed > total_channels:
                raise RuntimeError("Observed channels > total channels for stay {} under {}".format(
                    example["name"], condition))
            rows.append({
                "condition": condition,
                "r": r,
                "index": int(index),
                "stay_name": example["name"],
                "token_count": token_count,
                "raw_token_count": raw_token_count,
                "occupied_1h_bins": int(occupied_bins),
                "channels_observed": channels_observed,
                "post_discretization_sequence_length": post_len,
            })

        if expected_names is None:
            expected_names = names
        elif names != expected_names:
            raise RuntimeError("Input characterization stay order differs for {}".format(condition))

        part = pd.DataFrame(rows)
        if condition == dense_condition:
            raw_tokens_by_name = dict(zip(part["stay_name"], part["raw_token_count"]))
            raw_channel_totals_reference = {}
            for index in range(reader.get_number_of_examples()):
                example = reader.read_example(index)
                X = _input_window_array(example)
                raw_encoded, raw_encoded_header = raw_encoder.transform(
                    X, header=example["header"], end=float(example["t"]))
                raw_counts = observed_mask_information(raw_encoded, raw_encoded_header)["channel_counts"]
                for channel, value in raw_counts.items():
                    raw_channel_totals_reference[channel] = raw_channel_totals_reference.get(channel, 0) + int(value)
        else:
            if raw_tokens_by_name is None:
                raise RuntimeError("Dense raw reference must be characterized before {}".format(condition))
        raw_values = np.asarray([raw_tokens_by_name[x] for x in part["stay_name"]], dtype=float)
        part["raw_token_count"] = raw_values
        part["token_retained_fraction"] = part["token_count"].astype(float).values / raw_values
        if (part["token_count"].values > raw_values).any():
            raise RuntimeError("{} has token_count > dense raw for at least one stay".format(condition))
        if ((part["token_retained_fraction"] < -1e-12) |
                (part["token_retained_fraction"] > 1.0 + 1e-12)).any():
            raise RuntimeError("{} token retained fraction outside [0, 1]".format(condition))
        stay_level_parts.append(part)

        if channel_totals is not None:
            n_stays = float(len(part))
            for channel, total in channel_totals.items():
                channel_rows.append({
                    "condition": condition,
                    "r": r,
                    "channel": channel,
                    "mean_token_count": float(total) / n_stays if n_stays else np.nan,
                })

    stay_level = pd.concat(stay_level_parts, ignore_index=True) if stay_level_parts else pd.DataFrame()
    summary_rows = []
    for condition, group in stay_level.groupby("condition", sort=False):
        row = {
            "condition": condition,
            "r": group["r"].iloc[0],
            "n_stays": int(len(group)),
        }
        row.update(_summarize_values(group["token_count"].values, "token_count"))
        if "raw_token_count" in group.columns:
            row.update(_summarize_values(group["raw_token_count"].values, "raw_token_count"))
        row.update({
            "mean_token_retained_fraction": float(np.mean(group["token_retained_fraction"].values)),
            "median_token_retained_fraction": float(np.percentile(group["token_retained_fraction"].values, 50)),
            "mean_occupied_1h_bins": float(np.mean(group["occupied_1h_bins"].values)),
            "median_occupied_1h_bins": float(np.percentile(group["occupied_1h_bins"].values, 50)),
            "mean_channels_observed": float(np.mean(group["channels_observed"].values)),
            "median_channels_observed": float(np.percentile(group["channels_observed"].values, 50)),
        })
        lengths = sorted(set(group["post_discretization_sequence_length"].astype(int).tolist()))
        if len(lengths) != 1:
            raise RuntimeError("{} has inconsistent post-discretization lengths: {}".format(
                condition, lengths))
        row["post_discretization_sequence_length"] = int(lengths[0])
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    channel_summary = pd.DataFrame()
    channel_df = pd.DataFrame(channel_rows)
    if len(channel_df) and raw_channel_totals_reference is not None:
        n_reference_stays = float(len(stay_level[stay_level["condition"] == dense_condition]))
        raw_reference = pd.DataFrame([
            {"channel": channel, "mean_raw_token_count": float(total) / n_reference_stays}
            for channel, total in raw_channel_totals_reference.items()
        ])
        channel_summary = channel_df.merge(raw_reference, on="channel", how="left")
        channel_summary["mean_token_retained_fraction"] = (
            channel_summary["mean_token_count"] /
            channel_summary["mean_raw_token_count"].replace(0, np.nan)
        )
        condition_order = {
            condition: order
            for order, condition in enumerate(channel_df["condition"].drop_duplicates().tolist())
        }
        channel_summary["condition_order"] = channel_summary["condition"].map(condition_order)
        channel_summary = channel_summary.sort_values(
            ["condition_order", "channel"],
            ascending=[True, True],
        ).drop(["condition_order"], axis=1).reset_index(drop=True)
        channel_summary = channel_summary[[
            "condition", "r", "channel",
            "mean_raw_token_count", "mean_token_count",
            "mean_token_retained_fraction",
        ]]

    return stay_level, summary, channel_summary

def metric_difference(left_value, right_value, metric):
    # Positive always means the left/reference condition performs better.
    if metric == "brier":
        return right_value - left_value
    return left_value - right_value


def paired_prediction_frame(left_df, right_df):
    left = left_df[["stay", "patient_id", "y_true", "prediction"]].rename(
        columns={"y_true": "y_left", "prediction": "pred_left"})
    right = right_df[["stay", "patient_id", "y_true", "prediction"]].rename(
        columns={"y_true": "y_right", "prediction": "pred_right"})
    paired = left.merge(right, on=["stay", "patient_id"], how="inner")
    if len(paired) != len(left_df) or len(paired) != len(right_df):
        raise ValueError("Paired conditions contain different stay/patient IDs: paired={}, left={}, right={}".format(
            len(paired), len(left_df), len(right_df)))
    if not np.array_equal(paired["y_left"].values, paired["y_right"].values):
        raise ValueError("Paired conditions contain different y_true labels.")
    return paired.reset_index(drop=True)


def contrast_point_from_paired(paired, metric):
    y = paired["y_left"].values
    left_value = compute_metric(y, paired["pred_left"].values, metric)
    right_value = compute_metric(y, paired["pred_right"].values, metric)
    return left_value, right_value, metric_difference(left_value, right_value, metric)


def paired_patient_bootstrap(left_df, right_df, metric, n_boot=2000, random_state=DEFAULT_BOOTSTRAP_SEED):
    paired = paired_prediction_frame(left_df, right_df)
    left_value, right_value, point = contrast_point_from_paired(paired, metric)
    rng = np.random.RandomState(random_state)
    patient_ids = np.asarray(sorted(paired["patient_id"].unique()))
    y = paired["y_left"].values
    pred_left = paired["pred_left"].values
    pred_right = paired["pred_right"].values
    patient_to_indices = {
        pid: np.where(paired["patient_id"].values == pid)[0]
        for pid in patient_ids
    }
    boot = []
    for _ in range(n_boot):
        sampled_patients = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        sampled_indices = np.concatenate([patient_to_indices[pid] for pid in sampled_patients])
        left_b = compute_metric(y[sampled_indices], pred_left[sampled_indices], metric)
        right_b = compute_metric(y[sampled_indices], pred_right[sampled_indices], metric)
        diff = metric_difference(left_b, right_b, metric)
        if not np.isnan(diff):
            boot.append(diff)
    boot = np.asarray(boot)
    if len(boot):
        ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    else:
        ci_low, ci_high = np.nan, np.nan
    return {
        "left_metric": left_value,
        "right_metric": right_value,
        "difference": point,
        "ci_low": float(ci_low) if not np.isnan(ci_low) else np.nan,
        "ci_high": float(ci_high) if not np.isnan(ci_high) else np.nan,
        "n_patients": int(len(patient_ids)),
        "n_stays": int(len(paired)),
        "n_boot_valid": int(len(boot)),
        "boot_diffs": boot,
    }


def across_seed_patient_bootstrap(prediction_frames, keys, metric, n_boot=2000,
                                  random_state=DEFAULT_BOOTSTRAP_SEED):
    """Bootstrap the mean paired effect across model seeds using patients as units.

    prediction_frames maps each key to a prediction dataframe. keys is a list of
    (seed, left_key, right_key) tuples. All seeds must have the same patient set.
    """
    paired_by_seed = {}
    first_patients = None
    seed_point_diffs = []
    for seed, left_key, right_key in keys:
        paired = paired_prediction_frame(prediction_frames[left_key], prediction_frames[right_key])
        left_value, right_value, point = contrast_point_from_paired(paired, metric)
        seed_point_diffs.append(point)
        patients = np.asarray(sorted(paired["patient_id"].unique()))
        if first_patients is None:
            first_patients = patients
        elif not np.array_equal(first_patients, patients):
            raise ValueError("Patient sets differ across model seeds for metric {}".format(metric))
        paired_by_seed[seed] = paired

    if first_patients is None:
        return {
            "mean_difference": np.nan,
            "patient_bootstrap_ci_low": np.nan,
            "patient_bootstrap_ci_high": np.nan,
            "boot_diffs": np.asarray([]),
        }

    rng = np.random.RandomState(random_state)
    patient_ids = first_patients
    arrays_by_seed = {}
    patient_to_indices = {}
    for seed, paired in paired_by_seed.items():
        arrays_by_seed[seed] = {
            "y": paired["y_left"].values,
            "pred_left": paired["pred_left"].values,
            "pred_right": paired["pred_right"].values,
        }
        patient_to_indices[seed] = {
            pid: np.where(paired["patient_id"].values == pid)[0]
            for pid in patient_ids
        }

    boot_means = []
    for _ in range(n_boot):
        sampled_patients = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        seed_diffs = []
        for seed, arrays in arrays_by_seed.items():
            sampled_indices = np.concatenate([patient_to_indices[seed][pid] for pid in sampled_patients])
            y_b = arrays["y"][sampled_indices]
            left_b = compute_metric(y_b, arrays["pred_left"][sampled_indices], metric)
            right_b = compute_metric(y_b, arrays["pred_right"][sampled_indices], metric)
            diff = metric_difference(left_b, right_b, metric)
            if not np.isnan(diff):
                seed_diffs.append(diff)
        if len(seed_diffs):
            boot_means.append(float(np.mean(seed_diffs)))
    boot_means = np.asarray(boot_means)
    if len(boot_means):
        ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    else:
        ci_low, ci_high = np.nan, np.nan
    return {
        "mean_difference": float(np.mean(seed_point_diffs)) if len(seed_point_diffs) else np.nan,
        "patient_bootstrap_ci_low": float(ci_low) if not np.isnan(ci_low) else np.nan,
        "patient_bootstrap_ci_high": float(ci_high) if not np.isnan(ci_high) else np.nan,
        "boot_diffs": boot_means,
    }


def plot_contrast_ci(ax, df, x_col, y_col="mean_difference", low_col="patient_bootstrap_ci_low",
                     high_col="patient_bootstrap_ci_high", label_col=None, zero_line=True):
    if zero_line:
        ax.axhline(0, color="gray", linewidth=1, linestyle="--")
    x = np.arange(len(df))
    y = df[y_col].values
    low = df[low_col].values
    high = df[high_col].values
    yerr = np.vstack([y - low, high - y])
    ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=5)
    labels = df[label_col].tolist() if label_col is not None else df[x_col].tolist()
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)
    return ax

def expected_prediction_for_checkpoint(results_dir, horizon_or_timestep_dir, checkpoint):
    pred_dir = results_dir / "{}h".format(horizon_or_timestep_dir) / "test_predictions"
    return pred_dir / (checkpoint.name + ".csv")


def exact_prediction_for_checkpoint(results_dir, horizon_or_timestep_dir, checkpoint):
    expected = expected_prediction_for_checkpoint(results_dir, horizon_or_timestep_dir, checkpoint)
    return expected if expected.exists() else None


def compute_prediction_metrics(path):
    df = pd.read_csv(str(path))
    required = {"prediction", "y_true"}
    if not required.issubset(df.columns):
        raise ValueError("Prediction file missing required columns {}: {}".format(required, path))
    y = df["y_true"].values
    pred = df["prediction"].values
    return {
        "test_auroc": compute_metric(y, pred, "auroc"),
        "test_auprc": compute_metric(y, pred, "auprc"),
    }


def test_metrics(df):
    y = df["y_true"].values
    pred = df["prediction"].values
    return {
        "test_auroc": compute_metric(y, pred, "auroc"),
        "test_auprc": compute_metric(y, pred, "auprc"),
        "test_brier": compute_metric(y, pred, "brier"),
        "n_test_stays": int(len(df)),
        "n_test_patients": int(df["patient_id"].nunique()),
    }


def warn_incomplete_seed_sets(df, group_cols, seed_col="seed", expected_seeds=None):
    if expected_seeds is None:
        expected_seeds = {0, 1, 2, 3, 4}
    for keys, group in df.groupby(list(group_cols)):
        seeds = set(group[seed_col].dropna().astype(int).tolist())
        missing = set(expected_seeds) - seeds
        extra = seeds - set(expected_seeds)
        if missing or extra:
            print("PARTIAL SEED SET {}: observed={} missing={} extra={}".format(
                keys, sorted(seeds), sorted(missing), sorted(extra)))


def compute_seed_paired_drop(df, index_cols, resolution_col, reference_value, metric,
                             seed_col="seed", drop_col="drop"):
    join_cols = list(index_cols) + [seed_col]
    reference_col = "{}_reference".format(metric)
    ref = (
        df[df[resolution_col] == reference_value][join_cols + [metric]]
        .rename(columns={metric: reference_col})
    )
    base = df.drop([reference_col], axis=1) if reference_col in df.columns else df.copy()
    merged = base.merge(ref, on=join_cols, how="inner")
    merged[drop_col] = merged[reference_col] - merged[metric]
    return merged


def plot_mean_ci_curve(summary, x_col, ylabel, title, output_path, x_order,
                       series_col=None, series_order=None, xlabel="", zero_line=False,
                       raw_df=None, raw_metric=None):
    plt.figure(figsize=(7.5, 5.5))
    if series_col is None:
        sub = summary.sort_values(x_col)
        plt.errorbar(sub[x_col], sub["mean"], yerr=sub["ci95"], marker="o",
                     capsize=4, linewidth=2, markersize=6)
        if raw_df is not None and raw_metric is not None:
            plt.scatter(raw_df[x_col], raw_df[raw_metric], alpha=0.25, s=18)
    else:
        for value in series_order:
            sub = summary[summary[series_col] == value].sort_values(x_col)
            if len(sub) == 0:
                continue
            plt.errorbar(sub[x_col], sub["mean"], yerr=sub["ci95"], marker="o",
                         capsize=4, linewidth=2, label="{}".format(value))
            if raw_df is not None and raw_metric is not None:
                raw = raw_df[raw_df[series_col] == value]
                plt.scatter(raw[x_col], raw[raw_metric], alpha=0.20, s=16)
        plt.legend(title=series_col.replace("_", " "))
    if zero_line:
        plt.axhline(0, linestyle="--", linewidth=1, color="gray")
    plt.xticks(x_order, ["{}h".format(x) for x in x_order])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved:", output_path)
    print("Error bars = 95% Student-t CI across model seeds; n=1 groups have undefined CI.")

