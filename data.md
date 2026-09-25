## Overview

`mimic-iv-benchmarks` is a MIMIC-IV adaptation of the well-known MIMIC-III benchmark pipeline from Harutyunyan et al. Its purpose is to turn raw MIMIC-IV CSV tables into standardized ICU time-series prediction datasets, then provide baseline models and evaluation code.

The overall pipeline is:

**Raw MIMIC-IV tables → cohort filtering → per-patient events → per-ICU-stay episodes → task-specific examples → discretization/imputation → model training/evaluation**

The README says MIMIC-IV **v1.0** is the expected input version. The repository does not distribute MIMIC-IV itself.

---

## 1. Data used

The pipeline uses ICU admissions from MIMIC-IV and primarily extracts measurements from:

- `CHARTEVENTS`
- `LABEVENTS`
- `OUTPUTEVENTS`

It additionally uses tables containing:

- ICU stays
- hospital admissions
- patient demographics
- diagnoses / ICD codes

The main event extraction command defaults to:

```text
CHARTEVENTS
LABEVENTS
OUTPUTEVENTS
```

### Cohort filtering

`extract_subjects.py` constructs the ICU cohort and applies several important exclusions:

1. Removes ICU stays involving transfers.
2. Removes admissions with multiple ICU stays.
3. Removes patients younger than 18.
4. Associates admissions and patient information.
5. Creates ICU and in-hospital mortality labels.
6. Retains diagnoses corresponding to included stays.

So the benchmark is not simply “all MIMIC ICU stays.” It creates a cleaner cohort with effectively **one ICU stay per hospital admission**, which simplifies outcome attribution.

---

## 2. Raw-event processing

For every retained patient, the pipeline creates files roughly like:

```text
SUBJECT_ID/
    stays.csv
    diagnoses.csv
    events.csv
```

`events.csv` contains:

```text
SUBJECT_ID
HADM_ID
ICUSTAY_ID
CHARTTIME
ITEMID
VALUE
VALUEUOM
```

### Event validation

`validate_events.py` then cleans the event table.

It:

- removes events without `HADM_ID`,
- removes events belonging to admissions outside the selected cohort,
- attempts to recover missing ICU stay IDs from `HADM_ID`,
- removes events whose ICU stay ID conflicts with the stay associated with the admission.

This is important because the pipeline tries to ensure that every event can be confidently assigned to the correct ICU stay.

---

## 3. Converting raw ITEMIDs into clinical variables

`extract_episodes_from_subjects.py` maps raw MIMIC `ITEMID`s to standardized clinical variables using:

```text
mimic4benchmark/resources/itemid_to_variable_map.csv
```

Events are then grouped into one time series per ICU stay.

Each ICU episode becomes:

```text
episode#.csv
episode#_timeseries.csv
```

The first contains episode-level metadata/outcomes, such as:

- age
- sex
- ethnicity
- height
- weight
- mortality
- ICU length of stay
- diagnoses

The second contains longitudinal clinical measurements indexed by **hours since ICU admission**.

---

## 4. Time-series variables

The model preprocessing code uses **17 clinical channels**:

| Channel | Type |
|---|---|
| Capillary refill rate | categorical |
| Diastolic blood pressure | continuous |
| Fraction inspired oxygen | continuous |
| GCS eye opening | categorical |
| GCS motor response | categorical |
| GCS total | categorical |
| GCS verbal response | categorical |
| Glucose | continuous |
| Heart rate | continuous |
| Height | continuous |
| Mean blood pressure | continuous |
| Oxygen saturation | continuous |
| Respiratory rate | continuous |
| Systolic blood pressure | continuous |
| Temperature | continuous |
| Weight | continuous |
| pH | continuous |

These are essentially the familiar physiological variables from the Harutyunyan benchmark.

A useful point for your frequency experiments is that the **raw episode time series remains irregularly sampled at this stage**. The fixed sampling interval comes later in the model-side `Discretizer`.

---

# 5. Benchmark tasks

The code actually supports the four classic MIMIC benchmark tasks plus multitask learning, even though parts of the README emphasize mortality and LOS.

## In-hospital mortality

Question:

> Will the patient die during this hospital admission?

Input:

**first 48 hours of ICU data**

Stays shorter than 48 hours are excluded.

Each sample therefore looks conceptually like:

```text
X = measurements from ICU hour 0–48
y = in-hospital mortality
```

This is a single binary classification example per eligible ICU stay.

---

## Decompensation

Question:

> Will the patient die within the next 24 hours?

Predictions are generated repeatedly during the ICU stay.

Defaults in the script:

```text
sample_rate = 1 hour
future_time_interval = 24 hours
shortest_length = 4 hours
```

So after the first four hours, approximately one example is generated every hour:

```text
time = t
X = events from ICU admission through t
y = death during (t, t + 24h)
```

This is therefore a **dynamic prediction task**.

---

## Length of stay

Question:

> How much ICU time remains?

Like decompensation, the benchmark generates repeated predictions over the ICU stay.

Default:

```text
sample_rate = 1 hour
shortest_length = 4 hours
```

At prediction time `t`:

```text
X = observations through time t
y = total ICU LOS - t
```

The target is measured in hours.

The model can treat LOS either as:

- regression, or
- 10-bin classification.

The default example in the README uses the 10-bin `"custom"` partition.

---

## Phenotyping

Question:

> Which disease phenotype(s) does this patient have?

The input is essentially the **whole ICU stay**.

Diagnosis codes are mapped to HCUP Clinical Classification Software groups.

This is a **multilabel classification problem**.

---

## Multitask

The repository also includes a multitask dataset/model combining outcomes such as:

- mortality
- decompensation
- LOS
- phenotyping

and supplies multitask LSTM implementations.

---

# 6. Train/test/validation splitting

The pipeline first divides patients into:

```text
train/
test/
```

The split is shared across benchmark tasks.

This is a **patient-level split**, which avoids having ICU stays from the same patient appear in both train and test.

Later:

```bash
python -m mimic4models.split_train_val ...
```

extracts a validation subset from the training partition.

So conceptually:

```text
patients
   ↓
train patients ──→ train / validation
test patients  ──→ test
```

Model selection should therefore be done using validation performance, with the test set reserved for final evaluation.

---

# 7. The most important preprocessing step: discretization

Raw events are irregularly sampled.

Before feeding them to LSTM models, `mimic4models.preprocessing.Discretizer` converts them to regular time bins.

For example:

```python
Discretizer(
    timestep=1.0,
    store_masks=True,
    impute_strategy='previous',
    start_time='zero'
)
```

produces approximately:

```text
0–1 hr
1–2 hr
2–3 hr
3–4 hr
...
```

### What happens within a bin?

If a variable has multiple measurements in one bin, the code overwrites earlier measurements, effectively retaining the **last encountered observation in that interval**.

So:

```text
HR = 80 at 1.1 h
HR = 91 at 1.7 h
```

with a one-hour grid produces approximately:

```text
1–2 h → HR = 91
```

This detail is especially relevant to your measurement-frequency study.

---

# 8. Missing data handling

For the neural-network examples, the normal configuration is:

```text
impute_strategy = previous
```

That means:

- if there is an earlier observation → forward-fill it;
- if there is no prior observation → substitute a predefined normal value.

For example, configured defaults include values such as:

```text
Heart rate = 86
SpO2 = 98
Temperature = 36.6
pH = 7.4
FiO2 = 0.21
```

The pipeline additionally appends a **binary observation mask for every channel**.

So a model can distinguish:

```text
observed value
```

from:

```text
forward-filled/imputed value
```

even when the numeric input values happen to be equal.

---

# 9. Categorical variables

Variables such as GCS components are converted to one-hot representations.

Thus the 17 clinical variables expand into a larger numerical feature space.

Masks are then appended for the 17 original channels.

The standard LSTM code defaults to:

```text
input_dim = 76
```

which reflects this expanded representation rather than merely 17 numeric variables.

---

# 10. Normalization

Continuous features are standardized using a `Normalizer`.

Conceptually:

\[
x' = \frac{x-\mu}{\sigma}
\]

The saved normalizer files contain training-set means and standard deviations.

Categorical one-hot variables and mask channels are not treated like ordinary continuous measurements.

This explains the `.normalizer` files included under task directories.

---

# 11. Baseline models

The README describes seven baseline families.

### Linear models

Depending on the task:

- logistic regression
- linear regression
- LOS classification via logistic-style models

The repository includes L1/L2 regularization and some grid-search support.

For example, the documented mortality baseline uses:

```text
L2 logistic regression
C = 0.001
```

---

### Standard LSTM

A conventional stacked LSTM operates on the complete vector of measurements at each time bin:

```text
X_t = [HR, BP, glucose, ..., masks]
      ↓
     LSTM
      ↓
 prediction
```

For ordinary single-output tasks, it can use bidirectional LSTMs internally; deep-supervision configurations are unidirectional.

---

### Standard LSTM + deep supervision

Instead of predicting only from the final sequence representation, predictions are produced at multiple time steps.

This is especially useful for dynamic tasks such as:

- decompensation
- LOS

---

### Channel-wise LSTM

This model processes each clinical variable separately first.

Conceptually:

```text
Heart rate ─→ LSTM ─┐
Glucose    ─→ LSTM ─┤
Blood pressure → LSTM ─┤
...                  ├→ concatenate → LSTM → prediction
SpO2       ─→ LSTM ─┘
```

This architecture explicitly learns channel-specific temporal representations before combining variables.

---

### Channel-wise LSTM + deep supervision

Same idea, but with predictions at multiple time points.

---

### Multitask standard LSTM

A shared temporal representation feeds multiple benchmark-task outputs.

---

### Multitask channel-wise LSTM

Combines channel-specific encoders with multitask prediction heads.

---

# 12. How the common LSTM pipeline looks

For most experiments, the important sequence is:

```text
raw events
   ↓
ITEMID → clinical variable
   ↓
irregular ICU timeline
   ↓
Discretizer(timestep)
   ↓
fixed-width temporal bins
   ↓
forward-fill / normal-value imputation
   ↓
observation masks appended
   ↓
continuous-variable normalization
   ↓
LSTM / channel-wise LSTM
   ↓
prediction
```

That distinction between **raw measurement frequency** and **model discretization timestep** is very important.

Changing:

```text
--timestep 1 → 2 → 4 → 8 → 24
```

does **not necessarily simulate measurement being taken less often**. It primarily changes the temporal resolution at which already-recorded events are aggregated for the model.

For your current experiments, this is why your explicit raw-event frequency-reduction intervention is conceptually different from merely changing `--timestep`.

---

# 13. Losses

The loss depends on task.

### Mortality / decompensation

Binary classification:

```text
sigmoid output
binary cross-entropy
```

### Phenotyping

Multilabel sigmoid outputs.

### LOS

Can use either regression or classification.

For regression, the code uses:

```text
mean squared logarithmic error
```

For binned LOS:

```text
sparse categorical cross-entropy
```

with 10 LOS classes.

---

# 14. Evaluation metrics

### Mortality / decompensation

The metrics code reports:

- accuracy
- precision
- recall
- AUROC
- AUPRC
- min(precision, sensitivity)

### Phenotyping

Reports:

- per-label AUROC
- micro AUROC
- macro AUROC
- weighted AUROC

### LOS

Reports:

- mean absolute deviation / MAE
- MSE
- MAPE
- linearly weighted Cohen's kappa

The LOS classifier converts the predicted bin back into a representative number of hours before calculating regression-style metrics.

---

# 15. LOS bins

The default `"custom"` LOS setup uses 10 intervals approximately corresponding to:

```text
<1 day
1–2 days
2–3 days
3–4 days
4–5 days
5–6 days
6–7 days
7–8 days
8–14 days
>14 days
```

The code associates each bin with a representative mean LOS in hours when turning classification probabilities into continuous predictions.

---

# 16. Repository structure

The important directories can be thought of as:

```text
mimic4benchmark/
    scripts/
        extract_subjects.py
        validate_events.py
        extract_episodes_from_subjects.py

        create_in_hospital_mortality.py
        create_decompensation.py
        create_length_of_stay.py
        create_phenotyping.py
        create_multitask.py

    readers.py

    evaluation/

    resources/
        itemid_to_variable_map.csv
        phenotype definitions
        ...

mimic4models/
    preprocessing.py
    metrics.py
    keras_utils.py

    keras_models/
        lstm.py
        channel_wise_lstms.py
        multitask_lstm.py
        multitask_channel_wise_lstms.py

    in_hospital_mortality/
    decompensation/
    length_of_stay/
    phenotyping/
    multitask/
```

---

# 17. One-sentence characterization

The repository can be summarized as:

> **A MIMIC-IV implementation of the Harutyunyan ICU benchmark that transforms raw MIMIC-IV chart/lab/output events into 17-variable ICU time series, builds mortality, decompensation, LOS, and phenotype prediction tasks, discretizes irregular measurements into fixed temporal bins with imputation and observation masks, and supplies linear, standard LSTM, channel-wise LSTM, deep-supervision, and multitask baselines.**

For your project specifically, the most important conceptual separation is:

```text
RAW DATA
irregular real measurements
       ↓
your frequency intervention belongs here
       ↓
BENCHMARK PREPROCESSING
fixed timestep + imputation + masks
       ↓
MODEL
LSTM / GRU-D / etc.
```

This is why your current experiment design of reducing actual measurement events **before** the benchmark discretizer is much closer to studying *measurement frequency* than simply changing the repository's `--timestep`.