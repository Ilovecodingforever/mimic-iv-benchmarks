# Matched-count / representation-resolution control

This folder implements a controlled fixed-horizon ICU-exit experiment for separating three effects that are confounded when the existing preprocessing changes the `Discretizer` timestep from 1 hour to a coarser grid such as 2, 4, or 8 hours.

The three effects are:

1. Measurement quantity: fewer raw variable observations survive because several observations inside a coarse bin collapse to one value.
2. Temporal placement / sampling structure: the observations that survive are chosen by the coarse grid geometry.
3. Representation resolution: the LSTM sees fewer recurrent steps when the downstream grid is coarse.

The primary interval is `r=4h`. The same code also supports `r=2h` and `r=8h`.

| Condition | Raw observation intervention | Downstream grid | Recurrent steps for first 24h | Purpose |
| --- | --- | --- | --- | --- |
| A. Standard 1h | None | `Discretizer(timestep=1)` | 24 | Historical fine-grid baseline |
| B. Structured-r + 1h | Keep exactly the raw cells the `r`-hour discretizer would keep per stay, variable, and coarse bin | `Discretizer(timestep=1)` | 24 | Structured thinning while holding the 1h representation fixed |
| C. Random matched-count + 1h | For each stay and variable, randomly keep the same number of occupied 1h cells as B | `Discretizer(timestep=1)` | 24 | Random matched-count control for structured temporal placement |
| D. Existing r-hour | None beyond the historical `r`-hour discretizer | `Discretizer(timestep=r)` | `24/r` | Historical coarse-grid experiment |

A vs B estimates the effect of structured thinning while holding the downstream 1h representation fixed. It is not a pure count-only contrast: B both reduces the number of measurements and uses the particular coarse r-hour rule to decide which measurements survive. B vs C estimates the effect of structured temporal selection versus random matched-count selection while matching patient, variable, effective observation count, downstream grid, and sequence length. B vs D is the representation-resolution effect given the same coarse observation-selection rule: both use the same coarse observation-selection principle, but D represents the stay on a coarse grid.

This is a controlled characterization, not a perfectly additive causal decomposition. Measurement quantity remains one candidate mechanism, but A-B should not be read as an exact isolated information-quantity effect. B vs D should not be called an isolated LSTM sequence-length effect because the coarser grid also changes time alignment, mask locations, previous-value imputation trajectories, and temporal precision.

## Why B is not `Discretizer(r) -> Discretizer(1)`

The full `r`-hour `Discretizer` output is already a model matrix. It has collapsed observations, removed exact timestamps, added masks, one-hot encoded categorical channels, and imputed missing values. Feeding that matrix into another discretizer would not be a raw event sequence.

Instead B reuses only the discretizer's selection semantics:

```text
raw first-24h rows
-> for each stay x variable x r-hour bin, keep the raw row/column cell that Discretizer(timestep=r) would leave in that cell
-> preserve the original timestamp
-> reconstruct a sparse raw timeline containing only selected cells
-> run the normal 1h Discretizer
```

Selection is at raw cell granularity, not whole-row granularity. If one row contains both HR and Glucose, the sampler can keep HR without leaking Glucose.

## Why C samples occupied 1h variable-cells

If C sampled arbitrary raw observations, two sampled observations for one variable could land in the same 1h bin. The downstream 1h discretizer would then collapse them, and the model-visible count would no longer match B.

C therefore works at the occupied 1h variable-cell level. For each stay and variable it:

1. Computes `K`, the number of cells retained by B.
2. Finds all occupied 1h bins for that variable in the original first-24h data.
3. Uses the same row-order overwrite semantics as `Discretizer(timestep=1)` to choose one representative raw observation per occupied 1h bin.
4. Randomly samples exactly `K` distinct occupied 1h bins without replacement.
5. Reconstructs a sparse raw timeline from those representative cells.

Counts are matched per patient and per variable, not just globally.

Randomness is deterministic from `sampling_seed`, stay name, sampling interval, and variable name. It is independent of the model seed, so model seeds 0 through 4 see the same randomly thinned dataset when `sampling_seed` is unchanged.

## Training integration

The existing fixed-horizon ICU-exit training pipeline is reused. The sampler is a reader wrapper placed before the existing 1h discretizer:

```bash
python -m mimic4models.fixed_horizon_icu_exit.main \
  --network mimic4models/keras_models/lstm.py \
  --data /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/data/length-of-stay \
  --dim 16 \
  --depth 2 \
  --dropout 0.3 \
  --batch_size 8 \
  --horizon 12 \
  --timestep 1.0 \
  --sampling_interval 4 \
  --normalizer_dir /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/normalizers \
  --sampling_strategy structured \
  --output_dir /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/results/fixed_horizon_icu_exit/structured/12h/4h \
  --seed 0
```





For C:

```bash
python -m mimic4models.fixed_horizon_icu_exit.main \
  --network mimic4models/keras_models/lstm.py \
  --data /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/data/length-of-stay \
  --horizon 12 \
  --dim 16 \
  --depth 2 \
  --dropout 0.3 \
  --batch_size 8 \
  --timestep 1.0 \
  --sampling_interval 4 \
  --sampling_seed 100 \
  --normalizer_dir /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/normalizers \
  --sampling_strategy random_matched \
  --output_dir /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/results/fixed_horizon_icu_exit/random_matched/12h/4h \
  --seed 0
```




`structured` and `random_matched` require `--timestep 1.0`. The sampling interval and the downstream timestep are different concepts: `--sampling_interval 4 --timestep 1.0` means a 4h measurement-selection intervention represented on a 1h model grid.

Condition D remains the historical coarse-grid run, for example `--timestep 4.0 --sampling_strategy none`.

## Normalizers

Do not reuse the standard 1h normalizer for B or C. The sampled timelines have different masks and previous-value imputation trajectories.

Create B/C normalizers with the existing normalizer machinery plus the sampling flags:

```bash
python -m mimic4models.create_normalizer_state \
  --task fixed_horizon_icu_exit \
  --data /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/data/length-of-stay \
  --horizon 12 \
  --timestep 1.0 \
  --impute_strategy previous \
  --start_time zero \
  --store_masks \
  --sampling_interval 4 \
  --sampling_seed 100 \
  --output_dir /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/normalizers \
  --sampling_strategy random_matched 
```

```bash
python -m mimic4models.create_normalizer_state \
  --task fixed_horizon_icu_exit \
  --data /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/data/length-of-stay \
  --horizon 12 \
  --timestep 1.0 \
  --impute_strategy previous \
  --start_time zero \
  --store_masks \
  --sampling_interval 4 \
  --output_dir /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/normalizers \
  --sampling_strategy structured 
```

## raw data with LSTM (not discretized)

python -m mimic4models.fixed_horizon_icu_exit.main \
  --network mimic4models/keras_models/raw_lstm.py \
  --mode train \
  --timestep 0 \
  --horizon 12 \
  --sampling_strategy none \
  --dim 16 \
  --depth 2 \
  --dropout 0.3 \
  --batch_size 8 \
  --epochs 100 \
  --data /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/data/length-of-stay \
  --normalizer_dir /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/normalizers \
  --output_dir /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/results/fixed_horizon_icu_exit/12h

python -m mimic4models.fixed_horizon_icu_exit.main \
  --network mimic4models/keras_models/raw_lstm.py \
  --mode train \
  --timestep 0 \
  --horizon 12 \
  --sampling_strategy structured \
  --sampling_interval 8 \
  --dim 16 \
  --depth 2 \
  --dropout 0.3 \
  --batch_size 8 \
  --epochs 100 \
  --seed 0 \
  --data /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/data/length-of-stay \
  --normalizer_dir /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/normalizers \
  --output_dir /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/results/fixed_horizon_icu_exit/structured/12h/8h




python -m mimic4models.fixed_horizon_icu_exit.main \
  --network mimic4models/keras_models/grud.py \
  --horizon 12 \
  --timestep 1.0 \
  --dim 16 \
  --depth 1 \
  --dropout 0.3 \
  --batch_size 8 \
  --epochs 100 \
  --seed 0 \
  --sampling_strategy none \
  --data /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/data/length-of-stay \
  --normalizer_dir /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/normalizers 
  --output_dir /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/results/fixed_horizon_icu_exit/12h



The filename encodes task, sampling strategy, interval, random sampling seed where relevant, downstream timestep, imputation, mask setting, and training example count.

## Observation-frequency distribution shift

Fixed-horizon ICU-exit test runs can evaluate a checkpoint under a different observation-frequency regime than the one used for train and validation. The training flags still define the train/validation regime:

```text
--sampling_strategy
--sampling_interval
--sampling_seed
```

The test/deployment regime is controlled separately:

```text
--test_sampling_strategy
--test_sampling_interval
--test_sampling_seed
```

If a `test_sampling_*` argument is omitted, it inherits the corresponding training `sampling_*` value, so existing commands keep their old behavior. For the primary distribution-shift experiment, use `--timestep 1.0` so the downstream sequence resolution is held fixed while only the observation regime changes.

Example `1h -> 4h` evaluation from a standard 1h checkpoint:

```bash
python -m mimic4models.fixed_horizon_icu_exit.main \
  --mode test \
  --timestep 1.0 \
  --sampling_strategy none \
  --test_sampling_strategy structured \
  --test_sampling_interval 4 \
  --load_state <1h_checkpoint> \
  ...
```

The normalizer remains the one associated with the training regime. For `1h -> 8h`, that means the standard 1h training normalizer is used even though the test reader is structured-r8.

The main comparison is `r -> r` versus `1h -> r`. Both models see the same sparse test data, so the difference isolates sensitivity to train/deployment observation-regime mismatch rather than simply having fewer measurements at test time. Matched-regime test predictions keep the existing `<checkpoint>.csv` filename for backward compatibility. Shifted evaluations append the test regime, such as `testsample-structured-r4` or `testsample-random_matched-r4-sseed100`, to prevent multiple evaluations of the same checkpoint from overwriting each other.

## Checks

Toy edge-case tests:

```bash
python mimic4models/fixed_horizon_icu_exit/matched_count_control/test_sampling.py
```

Real-data validation on a small sample:

```bash
python -m mimic4models.fixed_horizon_icu_exit.matched_count_control.validate_sampling \
  --data /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/data/length-of-stay \
  --split train \
  --horizon 12 \
  --num_examples 100
```

The validation script checks unchanged names and labels, B/C per-variable count equality, B/C post-1h mask equality, deterministic random sampling, absence of selected observations after hour 24, expected sequence lengths for A/B/C/D, and `structured_vs_coarse_value_mismatches=0` for the B-vs-D observed coarse-cell equivalence check.
