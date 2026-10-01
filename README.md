MIMIC-IV Benchmarks
=========================

Python suite to construct benchmark machine learning datasets from the MIMIC-IV clinical database. The benchmark datasets cover two key inpatient clinical prediction tasks that map onto core machine learning problems: prediction of mortality from early admission data (classification), forecasting length of stay (regression).




## Motivation

Despite rapid growth in research that applies machine learning to clinical data, progress in the field appears far less dramatic than in other applications of machine learning. In image recognition, for example, the winning error rates in the [ImageNet Large Scale Visual Recognition Challenge](http://image-net.org/challenges/LSVRC/) (ILSVRC) plummeted almost 90% from 2010 (0.2819) to 2016 (0.02991).
There are many reasonable explanations for this discrepancy: clinical data sets are [inherently noisy and uncertain](http://www-scf.usc.edu/~dkale/papers/marlin-ihi2012-ehr_clustering.pdf) and often small relative to their complexity, and for many problems of interest, [ground truth labels for training and evaluation are unavailable](https://academic.oup.com/jamia/article-abstract/23/6/1166/2399304/Learning-statistical-models-of-phenotypes-using?redirectedFrom=PDF).

However, there is another, simpler explanation: practical progress has been difficult to measure due to the absence of community benchmarks like ImageNet. Such benchmarks play an important role in accelerating progress in machine learning research. For one, they focus the community on specific problems and stoke ongoing debate about what those problems should be. They also reduce the startup overhead for researchers moving into a new area. Finally and perhaps most important, benchmarks facilitate reproducibility and direct comparison of competing ideas.

Here we present four public benchmarks for machine learning researchers interested in health care, built using data from the publicly available Medical Information Mart for Intensive Care (MIMIC-IV) database ([paper](http://www.nature.com/articles/sdata201635), [website](http://mimic.physionet.org)). Our four clinical prediction tasks are critical care variants of four opportunities to transform health care using in "big clinical data" as described in [Bates, et al, 2014](http://content.healthaffairs.org/content/33/7/1123.abstract):

* early triage and risk assessment, i.e., mortality prediction
* identification of high cost patients, i.e. length of stay forecasting




## Structure
The content of this repository can be divided into four big parts:
* Tools for creating the benchmark datasets.  
* Tools for reading the benchmark datasets.
* Evaluation scripts.
* Baseline models and helper tools.

The `mimic4benchmark/scripts` directory contains scripts for creating the benchmark datasets.
The reading tools are in `mimic4benchmark/readers.py`.
All evaluation scripts are stored in the `mimic4benchmark/evaluation` directory.
The `mimic4models` directory contains the baselines models along with some helper tools.
Those tools include discretizers, normalizers and functions for computing metrics.


## Requirements

We do not provide the MIMIC-IV data itself. You must acquire the data yourself from https://physionet.org/content/mimiciv/1.0/. Specifically, download the CSVs. Otherwise, generally we make liberal use of the following packages:

- numpy
- pandas

For logistic regression  baselines [sklearn](http://scikit-learn.org/) is required. LSTM models use [Keras](https://keras.io/).


## Building a benchmark

Here are the required steps to build the benchmark. It assumes that you already have MIMIC-IV dataset (lots of CSV files) on the disk. 

1. Clone the repo.

       git clone https://github.com/vincenzorusso12/mimic4-benchmarks/
       cd mimic4-benchmarks/
    



python mimic-iv/prepare_mimic4_russo.py \
    /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0 \
    /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/vincenzorusso3-mimic-iv-benchmarks






2. The following command takes MIMIC-IV CSVs, generates one directory per `SUBJECT_ID` and writes ICU stay information to `data/{SUBJECT_ID}/stays.csv`, diagnoses to `data/{SUBJECT_ID}/diagnoses.csv`, and events to `data/{SUBJECT_ID}/events.csv`. This step might take around an hour.

       python -m mimic4benchmark.scripts.extract_subjects /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/vincenzorusso3-mimic-iv-benchmarks /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/data/root/



3. The following command attempts to fix some issues (ICU stay ID is missing) and removes the events that have missing information. About 80% of events remain after removing all suspicious rows (more information can be found in [`mimic4benchmark/scripts/more_on_validating_events.md`](mimic4benchmark/scripts/more_on_validating_events.md)).

       python -m mimic4benchmark.scripts.validate_events /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/data/root/

4. The next command breaks up per-subject data into separate episodes (pertaining to ICU stays). Time series of events are stored in ```{SUBJECT_ID}/episode{#}_timeseries.csv``` (where # counts distinct episodes) while episode-level information (patient age, gender, ethnicity, height, weight) and outcomes (mortality, length of stay, diagnoses) are stores in ```{SUBJECT_ID}/episode{#}.csv```. This script requires two files, one that maps event ITEMIDs to clinical variables and another that defines valid ranges for clinical variables (for detecting outliers, etc.). **Outlier detection is disabled in the current version**.

       python -m mimic4benchmark.scripts.extract_episodes_from_subjects /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/data/root/

5. The next command splits the whole dataset into training and testing sets. Note that the train/test split is the same of all tasks.

       python -m mimic4benchmark.scripts.split_train_and_test /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/data/root/
	
6. The following commands will generate task-specific datasets, which can later be used in models. These commands are independent, if you are going to work only on one benchmark task, you can run only the corresponding command.

       python -m mimic4benchmark.scripts.create_in_hospital_mortality /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/data/root/ /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/data/in-hospital-mortality/

       python -m mimic4benchmark.scripts.create_decompensation /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/data/root/ /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/data/decompensation/

       python -m mimic4benchmark.scripts.create_length_of_stay /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/data/root/ /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/data/length-of-stay/

       python -m mimic4benchmark.scripts.create_phenotyping /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/data/root/ /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/data/phenotyping/

       python -m mimic4benchmark.scripts.create_multitask /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/data/root/ /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/data/multitask/

After the above commands are done, there will be a directory `data/{task}` for each created benchmark task.
These directories have two sub-directories: `train` and `test`.
Each of them contains bunch of ICU stays and one file with name `listfile.csv`, which lists all samples in that particular set.
Each row of `listfile.csv` has the following form: `icu_stay, period_length, label(s)`.
A row specifies a sample for which the input is the collection of ICU event of `icu_stay` that occurred in the first `period_length` hours of the stay and the target is/are `label(s)`.
In in-hospital mortality prediction task `period_length` is always 48 hours, so it is not listed in corresponding listfiles.


python -m mimic4models.split_train_val /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/data/in-hospital-mortality/

python -m mimic4models.split_train_val /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/data/length-of-stay/

python -m mimic4models.fixed_horizon_icu_exit.main \
  --data /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/data/length-of-stay/ \
  --print_stats

python -m mimic4models.split_train_val /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/data/decompensation/


# 1. Build 8h normalization statistics
python -m mimic4models.create_normalizer_state \
  --task ihm \
  --timestep 8.0 \
  --impute_strategy previous \
  --start_time zero \
  --store_masks \
  --data /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/data/in-hospital-mortality/ \
  --output_dir /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/normalizers/

python -m mimic4models.create_normalizer_state \
  --task fixed_horizon_icu_exit \
  --timestep 8.0 \
  --impute_strategy previous \
  --start_time zero \
  --store_masks \
  --data /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/data/length-of-stay/ \
  --output_dir /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/normalizers/


for ts in 1.0; do
python -m mimic4models.create_normalizer_state \
  --task decomp \
  --n_samples 100000 \
  --timestep "$ts" \
  --impute_strategy previous \
  --start_time zero \
  --store_masks \
  --data /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/data/decompensation/ \
  --output_dir /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/normalizers/
done


# 2. Train 8h LSTM
python -u -m mimic4models.in_hospital_mortality.main \
  --network mimic4models/keras_models/lstm.py \
  --dim 16 \
  --timestep 8.0 \
  --depth 2 \
  --dropout 0.3 \
  --mode train \
  --batch_size 8 \
  --data /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/data/in-hospital-mortality/ \
  --normalizer_state /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/normalizers/ihm_ts:8.00_impute:previous_start:zero_masks:True_n:15579.normalizer \
  --output_dir /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/results/in-hospital-mortality/8h \
  --seed 0


  python -u -m mimic4models.in_hospital_mortality.main \
    --network mimic4models/keras_models/raw_lstm.py \
    --dim 16 \
    --depth 2 \
    --dropout 0.3 \
    --mode train \
    --batch_size 8 \
    --epochs 100 \
    --data /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/data/in-hospital-mortality \
    --normalizer_dir /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/normalizers \
    --output_dir /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/results/in-hospital-mortality/raw \
    --timestep 0 \
    --seed 0



TODO: 
decomp: all with seed 4, raw with all seeds




for ts in 24.0; do
python -m mimic4models.create_normalizer_state \
  --task decomp \
  --timestep "$ts" \
  --impute_strategy previous \
  --start_time zero \
  --store_masks \
  --data /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/data/decompensation/ \
  --output_dir /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/normalizers/ \
  --n_samples 100000
done



for seed in 0; do
    for ts in 1 2 4 8 12 24; do
python -u -m mimic4models.decompensation.main \
  --network mimic4models/keras_models/lstm.py \
  --dim 16 \
  --depth 2 \
  --dropout 0.3 \
  --timestep "$ts" \
  --mode train \
  --batch_size 8 \
  --epochs 100 \
  --seed "$seed" \
  --data /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/data/decompensation \
  --normalizer_dir /heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/normalizers \
  --output_dir "/heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/results/decompensation/${ts}h"
done
done



DATA=/heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/data/length-of-stay
NORM=/heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/normalizers
OUT=/heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/results/fixed_horizon_icu_exit

for seed in 0 1 2 3 4; do
    for horizon in 12 24 48 96 168; do
      echo "========================================"
      echo "horizon=${horizon}h seed=${seed}"
      echo "========================================"
      python -u -m mimic4models.fixed_horizon_icu_exit.main \
        --network mimic4models/keras_models/lstm.py \
        --dim 16 \
        --depth 2 \
        --dropout 0.3 \
        --mode train \
        --batch_size 8 \
        --horizon "$horizon" \
        --seed "$seed" \
        --data "$DATA" \
        --normalizer_dir "$NORM" \
        --output_dir "$OUT/${horizon}h" \
        --timestep 1
  done
done


for seed in 4; do
    for horizon in 12 24 48 96 168; do
      python -u -m mimic4models.fixed_horizon_icu_exit.main \
        --network mimic4models/keras_models/raw_lstm.py \
        --dim 16 \
        --depth 2 \
        --dropout 0.3 \
        --mode train \
        --batch_size 8 \
        --horizon "$horizon" \
        --seed "$seed" \
        --data "$DATA" \
        --normalizer_dir "$NORM" \
        --output_dir "$OUT/${horizon}h" \
        --timestep 0
done
done



## Readers
To simplify the reading of benchmark data we wrote special classes.
The `mimic4benchmark/readers.py` contains class `Reader` and five other task-specific classes derived from it.
These are designed to simplify reading of benchmark data. The classes require a directory containing ICU stays and a listfile specifying the samples.
Again, we encourage to use these readers to avoid mistakes in the reading step (for example using events that happened after the first `period_length` hours).  
For more information about using readers view the [`mimic4benchmark/more_on_readers.md`](mimic4benchmark/more_on_readers.md) file.


## Evaluation
For each of the four tasks we provide scripts for evaluating models.
These scripts receive a `csv` file containing the predictions and produce a `json` file containing the scores and confidence intervals for different metrics.
We highly encourage to use these scripts to prevent any mistake in the evaluation step.
For details about the usage of the evaluation scripts view the [`mimic4benchmark/evaluation/README.md`](mimic4benchmark/evaluation/README.md) file.


## Baselines
For each of the four main tasks we provide 7 baselines:  
* Linear/logistic regression
* Standard LSTM
* Standard LSTM + deep supervision
* Channel-wise LSTM
* Channel-wise LSTM + deep supervision
* Multitask standard LSTM
* Multitask channel-wise LSTM

The detailed descriptions of the baselines will appear in the next version of the paper.

Linear models can be found in `mimic4models/{task}/logistic` directories.
LSTM-based models are in `mimic4models/keras_models` directory.

Please note that running linear models can take hours because of extensive grid search and feature extraction.
You can change the size of the training data of linear models in the scripts and they will became faster (of course the performance will not be the same).

### Train / validation split

Use the following command to extract validation set from the training set. This step is required for running the baseline models. Likewise the train/test split, the train/validation split is the same for all tasks.

       python -m mimic4models.split_train_val {dataset-directory}
       
`{dataset-directory}` can be either `data/in-hospital-mortality`, `data/decompensation`, `data/length-of-stay`, `data/phenotyping` or `data/multitask`.


### In-hospital mortality prediction

Run the following command to train the neural network which gives the best result. We got the best performance on validation set after 28 epochs.
       
       python -um mimic4models.in_hospital_mortality.main --network mimic4models/keras_models/lstm.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --batch_size 8 --output_dir mimic4models/in_hospital_mortality

Use the following command to train logistic regression. The best model we got used L2 regularization with `C=0.001`:
       
       python -um mimic4models.in_hospital_mortality.logistic.main --l2 --C 0.001 --output_dir mimic4models/in_hospital_mortality/logistic



### Length of stay prediction

The best model we got for this task was trained for 19 chunks.
       
       python -um mimic4models.length_of_stay.main --network mimic4models/keras_models/lstm.py --dim 64 --timestep 1.0 --depth 1 --dropout 0.3 --mode train --batch_size 8 --partition custom --output_dir mimic4models/length_of_stay

Use the following command to train a logistic regression. It will have L1 regularization with `C=0.00001`. To run a grid search over a space of hyper-parameters add `--grid-search` to the command.
       
       python -um mimic4models.length_of_stay.logistic.main_cf --output_dir mimic4models/length_of_stay/logistic

To run a linear regression use this command:

        python -um mimic4models.length_of_stay.logistic.main --output_dir mimic4models/length_of_stay/logistic






### Note on normalizer construction

The current `create_normalizer_state.py` uses each task's default `train/listfile.csv` unless a task explicitly passes `train_listfile.csv`. Since `split_train_val.py` creates separate `train_listfile.csv` and `val_listfile.csv` files without modifying `train/listfile.csv`, normalizers generated with the current code may include examples that are later assigned to the validation split.

This is a mild validation-data leakage through feature normalization statistics only; labels are not used to fit the normalizer. Test data are not involved.

Affected tasks/experiments:

- **Experiment 1 — In-hospital mortality**
  - grid timesteps `1h, 2h, 4h, 8h, 12h, 24h` and raw

Not affected:

- **Fixed-horizon ICU-exit experiments (Exp2A/Exp2B/Exp3)**, because their normalizer construction already explicitly uses `train_listfile.csv`.

For final experiments, `create_normalizer_state.py` should be updated so all tasks explicitly use `train_listfile.csv`, and the affected normalizers/models should be regenerated. Existing exploratory results can still be used provisionally, but should be rerun before final reporting.


### Decompensation batch size and epoch definition

The decompensation training loop follows the original MIMIC-III benchmark design. Unlike in-hospital mortality, which has one prediction example per ICU stay and trains with a conventional full-dataset epoch, decompensation is a repeated prediction task with many prediction-time examples per stay. The original benchmark therefore uses a streaming generator and defines each epoch as a fixed number of minibatches: 2,000 training batches and 1,000 validation batches. Length-of-stay uses the same pattern, while in-hospital mortality and fixed-horizon ICU exit do not.

A consequence of this design is that changing `batch_size` also changes the number of examples processed per epoch. For example, the historical setting `batch_size=8` corresponds to 16,000 training examples and 8,000 validation examples per epoch, while `batch_size=32` would otherwise increase those totals to 64,000 and 32,000. This makes larger batch sizes change both optimization batch size and the effective data budget per epoch.

To preserve the historical `batch_size=8` behavior while allowing larger batches for faster execution, decompensation is changed to use a fixed per-epoch example budget of 16,000 training examples and 8,000 validation examples, with the number of minibatches computed from the selected batch size. Thus `batch_size=8` still gives 2,000/1,000 batches exactly as before, while `batch_size=32` gives 500/250 batches. Previous experiments run with the old code and `batch_size=8` remain directly comparable. For future experiments, the same batch size should be used across all conditions and seeds within an experiment, because larger batches still imply fewer optimizer updates per epoch even when the number of examples is held fixed.

LOS might need this change as well.

Test evaluation is unchanged. Test mode already iterates over the full test set exactly once; changing `batch_size` only changes the number of inference batches, not which examples are evaluated.

