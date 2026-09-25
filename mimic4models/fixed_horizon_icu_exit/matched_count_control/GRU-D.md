# GRU-D

This repository includes a GRU-D implementation for the fixed-horizon ICU-exit experiments. The implementation is adapted from the author-maintained [`PeterChe1990/GRU-D`](https://github.com/PeterChe1990/GRU-D) repository and integrated into the MIMIC-IV benchmark pipeline.

GRU-D explicitly models informative missingness using three inputs:

- \(X\): observed values
- \(M\): observation masks
- \(S\): timestamps

The recurrent cell uses the timestamps and masks to compute the elapsed time since each variable was last observed and applies learned decay functions to the input and hidden state.

## Supported input representations

Two GRU-D input modes are supported.

### 1. Regular 1-hour GRU-D

Run with:

```bash
--network mimic4models/keras_models/grud.py \
--timestep 1.0
```

The standard benchmark discretizer converts the original irregular EHR measurements onto a regular 1-hour grid.

The model receives:

```text
timestamps: [0, 1, 2, ..., T]
values: discretized / normalized values
masks: whether each variable was actually observed in each bin
```

Missing positions may contain the benchmark's previous-value imputation, but GRU-D also receives the observation mask and ignores the supplied value whenever the mask is zero.

This representation is useful when comparing GRU-D directly against the standard benchmark LSTM at the same downstream temporal resolution.

---

### 2. Raw irregular GRU-D

Run with:

```bash
--network mimic4models/keras_models/grud.py \
--timestep 0
```

In this mode, the benchmark discretizer is not used.

The model operates on the original irregular event rows:

```text
original MIMIC event rows
        ↓
raw value / mask encoding
        ↓
actual irregular timestamps
        ↓
GRU-D
```

No hourly bins or forward filling are introduced before GRU-D.

For each patient, the model receives:

```text
X: (T, D) observed values
M: (T, D) observation masks
S: (T, 1) irregular timestamps
```

where \(T\) varies across patients.

## Timestamp convention for raw GRU-D

The raw MIMIC-IV representation stores `Hours` relative to ICU admission.

For example:

```text
Hours = [0.8, 1.4, 3.2, 5.7]
```

These original timestamps are preserved throughout:

- raw data loading,
- frequency sampling,
- normalization,
- diagnostics,
- other models.

Immediately before the timestamps are passed into GRU-D, they are shifted so that the first retained event occurs at time zero:

```text
original Hours:
[0.8, 1.4, 3.2, 5.7]

GRU-D timestamps:
[0.0, 0.6, 2.4, 4.9]
```

This follows the convention used in `PeterChe1990/GRU-D`, whose MIMIC preprocessing performs:

```python
timestamp = timestamp - timestamp[0]
```

Importantly, this is a **GRU-D-specific input transformation**. The underlying raw timestamps are not modified.

This prevents the timestamp shift from changing any observation-frequency intervention or other preprocessing behavior.

## Observation-frequency interventions

GRU-D supports the same measurement-frequency interventions as the other fixed-horizon ICU-exit models.

For example:

```bash
--sampling_strategy structured \
--sampling_interval 4
```

The processing order for raw GRU-D is:

```text
original irregular events
        ↓
frequency sampling using original Hours
        ↓
retain selected measurements at their original timestamps
        ↓
raw value/mask encoding
        ↓
GRU-D-only timestamp shift
        ↓
GRU-D
```

Thus, if structured sampling retains measurements at:

```text
[3.7, 7.5, 11.8]
```

the sampling procedure operates on exactly those timestamps.

GRU-D subsequently receives:

```text
[0.0, 3.8, 8.1]
```

The relative temporal spacing is unchanged.

Supported sampling strategies are:

```text
none
structured
random_matched
```

with sampling intervals:

```text
2h
4h
8h
```

## Normalization

Raw GRU-D uses `RawObservedNormalizer`.

Continuous variables are standardized using statistics computed from **observed training values only**.

For a continuous variable:

\[
x' = \frac{x-\mu}{\sigma}
\]

only entries with observation mask \(M=1\) contribute to the training mean and standard deviation.

Unobserved positions remain zero.

The following are not normalized:

- observation masks,
- timestamps,
- categorical one-hot indicators.

Because continuous variables are centered around the training mean, zero in normalized space corresponds approximately to the empirical mean. This is compatible with GRU-D's learned input decay toward the population mean.

## Categorical variables

The MIMIC benchmark contains categorical channels such as Glasgow Coma Scale components.

These are represented using the existing benchmark one-hot encoding.

For GRU-D, the original channel observation mask is expanded across all one-hot dimensions belonging to that variable.

For example:

```text
GCS eye opening
    ↓
GCS eye opening -> 1
GCS eye opening -> 2
GCS eye opening -> 3
GCS eye opening -> 4
```

All four encoded dimensions receive the same observation mask.

Therefore:

```text
X.shape == M.shape
```

as required by the GRU-D implementation.

## Variable-length sequences and padding

Raw sequences contain different numbers of event rows.

Training therefore uses batch-local padding:

```text
patient A: T = 120
patient B: T = 147
patient C: T = 131

batch tensor: T = 147
```

Padded rows contain zero values and zero observation masks.

A timestep mask is derived from the observation masks so that padded rows are ignored by the recurrent GRU-D computation.

Padding timestamps are kept at zero.

The test suite verifies that a patient's prediction is unchanged when the same sequence is:

1. evaluated alone, or
2. evaluated in a batch that requires additional padding.

## Elapsed-time calculation

GRU-D internally tracks the last time each variable was observed.

For variable \(d\), elapsed time is conceptually:

\[
\Delta_t^d =
S_t - S_{\text{last observed},d}.
\]

For example, with:

```text
timestamps:   [0.0, 0.5, 2.2, 4.9]
HR observed:  [1,   0,   0,   1]
```

the elapsed-time signal for heart rate grows until the variable is observed again.

Unlike a standard LSTM, GRU-D therefore has direct access to the amount of time since each measurement was last recorded.

## Relationship to PeterChe GRU-D

The recurrent implementation is intentionally kept compatible with the public author-maintained `PeterChe1990/GRU-D` implementation.

This includes implementation details that differ slightly from a literal reading of the equations in the GRU-D paper.

In particular, the PeterChe implementation:

- uses `activation='sigmoid'` as the default GRU candidate activation;
- computes a decayed hidden state `h_tm1d`, but the final GRU carry branch uses `h_tm1`.

These behaviors are retained rather than silently modifying the reference implementation.

This makes the model used here best described as:

> an adaptation of the public PeterChe GRU-D implementation to the MIMIC-IV benchmark and the fixed-horizon ICU-exit tasks.

If needed, a separate paper-equation variant can be evaluated as a sensitivity analysis rather than changing the primary GRU-D baseline.

## Why both regular and raw GRU-D are included

The two versions answer different questions.

### Regular 1-hour GRU-D

```text
raw observations
    ↓
1h representation
    ↓
GRU-D
```

This controls downstream temporal resolution and makes comparison with the benchmark LSTM straightforward.

### Raw GRU-D

```text
raw observations
    ↓
original irregular sequence
    ↓
GRU-D
```

This more closely matches the irregular-timestamp input format used by the PeterChe implementation and allows GRU-D to directly observe the actual temporal spacing between retained measurement events.

Comparing the two helps separate:

- effects of measurement frequency,
- effects of discretization,
- and GRU-D's ability to exploit explicit observation timing.

## Example commands

### Raw GRU-D

```bash
for seed in 0 1 2; do
python -u -m mimic4models.fixed_horizon_icu_exit.main \
  --network mimic4models/keras_models/grud.py \
  --dim 16 \
  --depth 1 \
  --dropout 0.3 \
  --mode train \
  --batch_size 8 \
  --horizon 12 \
  --timestep 0 \
  --sampling_strategy none \
  --epochs 100 \
  --seed "$seed" \
  --data /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/data/length-of-stay \
  --normalizer_dir /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/normalizers \
  --output_dir /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/results/fixed_horizon_icu_exit/12h
done
```




### Raw GRU-D with structured 4-hour measurement frequency

```bash
python -u -m mimic4models.fixed_horizon_icu_exit.main \
  --network mimic4models/keras_models/grud.py \
  --dim 16 \
  --depth 1 \
  --dropout 0.3 \
  --mode train \
  --batch_size 8 \
  --horizon 24 \
  --seed 0 \
  --data "$DATA" \
  --normalizer_dir "$NORM" \
  --output_dir "$OUT/grud_raw/24h/structured_r4" \
  --timestep 0 \
  --sampling_strategy structured \
  --sampling_interval 4
```

### Regular 1-hour GRU-D

```bash
python -u -m mimic4models.fixed_horizon_icu_exit.main \
  --network mimic4models/keras_models/grud.py \
  --dim 16 \
  --depth 1 \
  --dropout 0.3 \
  --mode train \
  --batch_size 8 \
  --horizon 24 \
  --seed 0 \
  --data "$DATA" \
  --normalizer_dir "$NORM" \
  --output_dir "$OUT/grud_1h/24h" \
  --timestep 1.0 \
  --sampling_strategy none
```

## Tests

GRU-D input and raw-timestamp sanity checks are implemented in:

```text
mimic4models/tests/test_grud_input.py
```

Run:

```bash
python mimic4models/tests/test_grud_input.py
```

The tests cover:

- value/mask mapping,
- categorical mask expansion,
- irregular timestamp preservation,
- GRU-D timestamp re-zeroing,
- elapsed-time calculations,
- structured frequency sampling,
- ignored values when `M=0`,
- batch padding invariance,
- and regular-grid GRU-D behavior.

## Relevant files

```text
mimic4models/keras_models/grud.py
mimic4models/keras_models/grud_layers.py
mimic4models/fixed_horizon_icu_exit/raw.py
mimic4models/fixed_horizon_icu_exit/main.py
mimic4models/fixed_horizon_icu_exit/matched_count_control/sampling.py
mimic4models/tests/test_grud_input.py
```
