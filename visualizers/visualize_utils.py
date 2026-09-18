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
    df = pd.read_csv(str(log_path), sep=";")
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

def exact_prediction_for_checkpoint(results_dir, horizon_or_timestep_dir, checkpoint):
    pred_dir = results_dir / "{}h".format(horizon_or_timestep_dir) / "test_predictions"
    expected = pred_dir / (checkpoint.name + ".csv")
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

