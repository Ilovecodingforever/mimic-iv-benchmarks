from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import argparse
import os
import imp
import re
import random
import glob
import tensorflow as tf

from mimic4models.in_hospital_mortality import utils
from mimic4benchmark.readers import InHospitalMortalityReader

from mimic4models.preprocessing import Discretizer, Normalizer
from mimic4models.fixed_horizon_icu_exit.raw import (
    RawBatchSequence, RawObservedNormalizer, RawSequenceEncoder, is_raw_timestep,
    load_raw_data_unpadded)
from mimic4models import metrics
from mimic4models import keras_utils
from mimic4models import common_utils

from keras.callbacks import ModelCheckpoint, CSVLogger


def ihm_normalizer_pattern(normalizer_dir, timestep, imputation):
    if is_raw_timestep(timestep):
        return os.path.join(normalizer_dir, 'ihm_raw_ts:0.00_observed_only_n:*.normalizer')
    return os.path.join(
        normalizer_dir,
        'ihm_ts:{:.2f}_impute:{}_start:zero_masks:True_n:*.normalizer'.format(
            float(timestep), imputation))


def resolve_normalizer_state(normalizer_state, normalizer_dir, timestep, imputation):
    if normalizer_state is not None:
        return normalizer_state
    if normalizer_dir is None:
        legacy = 'ihm_ts{}.input_str_{}.start_time_zero.normalizer'.format(timestep, imputation)
        return os.path.join(os.path.dirname(__file__), legacy)

    pattern = ihm_normalizer_pattern(normalizer_dir, timestep, imputation)
    matches = sorted(glob.glob(pattern))
    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        raise IOError('No in-hospital mortality normalizer found for pattern: {}'.format(pattern))
    raise IOError('Multiple in-hospital mortality normalizers found for pattern: {}. Matches: {}'.format(
        pattern, matches))


def raw_bucket_size(batch_size):
    return max(int(batch_size) * 100, int(batch_size))


def _flat_model_predictions(outputs):
    if isinstance(outputs, list):
        outputs = outputs[0]
    return np.array(outputs).flatten()


def predict_raw_batches(model, sequences, labels, batch_size):
    data_seq = RawBatchSequence(sequences, labels, batch_size=batch_size, shuffle=False,
                                bucket_size=raw_bucket_size(batch_size), target_repl=False)
    predictions = np.zeros((len(sequences),), dtype=float)
    for batch_i in range(len(data_seq)):
        x_batch, _ = data_seq[batch_i]
        batch_pred = model.predict(x_batch, batch_size=batch_size, verbose=0)
        predictions[data_seq.batch_indices(batch_i)] = _flat_model_predictions(batch_pred)
    return predictions


parser = argparse.ArgumentParser()
common_utils.add_common_arguments(parser)
parser.add_argument('--target_repl_coef', type=float, default=0.0)
parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                    default=os.path.join(os.path.dirname(__file__), '../../data/in-hospital-mortality/'))
parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='.')
parser.add_argument('--normalizer_dir', type=str, default=None,
                    help='Directory containing in-hospital mortality normalizer states.')
parser.add_argument('--seed', type=int, default=49297)
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
tf.set_random_seed(args.seed)
print(args)

if args.small_part:
    args.save_every = 2**30

raw_mode = is_raw_timestep(args.timestep)
target_repl = (args.target_repl_coef > 0.0 and args.mode == 'train')

# Build readers, discretizers, normalizers
train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                         listfile=os.path.join(args.data, 'train_listfile.csv'),
                                         period_length=48.0)

val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                       listfile=os.path.join(args.data, 'val_listfile.csv'),
                                       period_length=48.0)

if raw_mode:
    representation = RawSequenceEncoder()
    feature_header = representation.header()
    normalizer = RawObservedNormalizer(representation.continuous_value_fields(),
                                       representation.continuous_mask_fields())
else:
    representation = Discretizer(timestep=float(args.timestep),
                                 store_masks=True,
                                 impute_strategy='previous',
                                 start_time='zero')
    feature_header = representation.transform(train_reader.read_example(0)["X"])[1].split(',')
    cont_channels = [i for (i, x) in enumerate(feature_header) if x.find("->") == -1]
    normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize

normalizer_state = resolve_normalizer_state(
    args.normalizer_state, args.normalizer_dir, args.timestep, args.imputation)
print("==> normalizer_state:", normalizer_state)
normalizer.load_params(normalizer_state)

args_dict = dict(args._get_kwargs())
args_dict['header'] = feature_header
args_dict['task'] = 'ihm'
args_dict['target_repl'] = target_repl

# Build the model
print("==> using model {}".format(args.network))
model_module = imp.load_source(os.path.basename(args.network), args.network)
model = model_module.Network(**args_dict)
suffix = ".bs{}{}{}.ts{}{}.seed{}".format(args.batch_size,
                                             ".L1{}".format(args.l1) if args.l1 > 0 else "",
                                             ".L2{}".format(args.l2) if args.l2 > 0 else "",
                                             args.timestep,
                                             ".trc{}".format(args.target_repl_coef) if args.target_repl_coef > 0 else "",
                                             args.seed)
model.final_name = args.prefix + model.say_name() + suffix
print("==> model.final_name:", model.final_name)


# Compile the model
print("==> compiling the model")
optimizer_config = {'class_name': args.optimizer,
                    'config': {'lr': args.lr,
                               'beta_1': args.beta_1}}

# NOTE: one can use binary_crossentropy even for (B, T, C) shape.
#       It will calculate binary_crossentropies for each class
#       and then take the mean over axis=-1. Tre results is (B, T).
if target_repl:
    loss = ['binary_crossentropy'] * 2
    loss_weights = [1 - args.target_repl_coef, args.target_repl_coef]
else:
    loss = 'binary_crossentropy'
    loss_weights = None

model.compile(optimizer=optimizer_config,
              loss=loss,
              loss_weights=loss_weights)
model.summary()

# Load model weights
n_trained_chunks = 0
if args.load_state != "":
    keras_utils.patch_legacy_keras_h5py_attrs()
    model.load_weights(args.load_state)
    n_trained_chunks = int(re.match(".*epoch([0-9]+).*", args.load_state).group(1))


# Read data
if raw_mode:
    train_raw = load_raw_data_unpadded(train_reader, representation, normalizer, args.small_part)
    val_raw = load_raw_data_unpadded(val_reader, representation, normalizer, args.small_part)
else:
    train_raw = utils.load_data(train_reader, representation, normalizer, args.small_part)
    val_raw = utils.load_data(val_reader, representation, normalizer, args.small_part)

if target_repl and not raw_mode:
    T = train_raw[0][0].shape[0]

    def extend_labels(data):
        data = list(data)
        labels = np.array(data[1])  # (B,)
        data[1] = [labels, None]
        data[1][1] = np.expand_dims(labels, axis=-1).repeat(T, axis=1)  # (B, T)
        data[1][1] = np.expand_dims(data[1][1], axis=-1)  # (B, T, 1)
        return data

    train_raw = extend_labels(train_raw)
    val_raw = extend_labels(val_raw)

if args.mode == 'train':

    # Prepare training
    csv_append = keras_utils.prepare_keras_training_run(
        args.output_dir, model.final_name, args.load_state, args.mode)
    path = os.path.join(args.output_dir, 'keras_states/' + model.final_name + '.epoch{epoch}.test{val_loss}.state')

    train_sequence = None
    val_sequence = None
    if raw_mode:
        train_sequence = RawBatchSequence(
            train_raw[0], train_raw[1], batch_size=args.batch_size,
            shuffle=True, bucket_size=raw_bucket_size(args.batch_size),
            seed=args.seed, target_repl=target_repl)
        val_sequence = RawBatchSequence(
            val_raw[0], val_raw[1], batch_size=args.batch_size,
            shuffle=False, bucket_size=raw_bucket_size(args.batch_size),
            target_repl=target_repl)
        train_sequence.print_diagnostics()

    metrics_callback = keras_utils.InHospitalMortalityMetrics(train_data=train_sequence or train_raw,
                                                              val_data=val_sequence or val_raw,
                                                              target_repl=(args.target_repl_coef > 0),
                                                              batch_size=args.batch_size,
                                                              verbose=args.verbose,
                                                              skip_train_metrics=raw_mode)
    # make sure save directory exists
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    saver = ModelCheckpoint(path, verbose=1, period=args.save_every)

    keras_logs = os.path.join(args.output_dir, 'keras_logs')
    if not os.path.exists(keras_logs):
        os.makedirs(keras_logs)
    csv_logger = CSVLogger(os.path.join(keras_logs, model.final_name + '.csv'),
                           append=csv_append, separator=';')

    print("==> training")
    if raw_mode:
        model.fit_generator(generator=train_sequence.iter_batches(),
                            steps_per_epoch=len(train_sequence),
                            validation_data=val_sequence.iter_batches(),
                            validation_steps=len(val_sequence),
                            epochs=n_trained_chunks + args.epochs,
                            initial_epoch=n_trained_chunks,
                            callbacks=[metrics_callback, saver, csv_logger],
                            verbose=args.verbose)
    else:
        model.fit(x=train_raw[0],
                  y=train_raw[1],
                  validation_data=val_raw,
                  epochs=n_trained_chunks + args.epochs,
                  initial_epoch=n_trained_chunks,
                  callbacks=[metrics_callback, saver, csv_logger],
                  shuffle=True,
                  verbose=args.verbose,
                  batch_size=args.batch_size)

elif args.mode == 'test':

    # ensure that the code uses test_reader
    del train_reader
    del val_reader
    del train_raw
    del val_raw

    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
                                            listfile=os.path.join(args.data, 'test_listfile.csv'),
                                            period_length=48.0)
    if raw_mode:
        ret = load_raw_data_unpadded(test_reader, representation, normalizer, args.small_part,
                                     return_names=True)
    else:
        ret = utils.load_data(test_reader, representation, normalizer, args.small_part,
                              return_names=True)

    data = ret["data"][0]
    labels = ret["data"][1]
    names = ret["names"]

    if raw_mode:
        predictions = predict_raw_batches(model, data, labels, args.batch_size)
    else:
        predictions = model.predict(data, batch_size=args.batch_size, verbose=1)
        predictions = np.array(predictions)[:, 0]
    metrics.print_metrics_binary(labels, predictions)

    path = os.path.join(args.output_dir, "test_predictions", os.path.basename(args.load_state)) + ".csv"
    utils.save_results(names, predictions, labels, path)

else:
    raise ValueError("Wrong value for args.mode")
