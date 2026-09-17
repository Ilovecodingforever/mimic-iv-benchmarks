from __future__ import absolute_import
from __future__ import print_function

import argparse
import imp
import os
import random
import re

import numpy as np
from mimic4benchmark.readers import FixedHorizonIcuExitReader
from mimic4models import common_utils
from mimic4models import metrics
from mimic4models.in_hospital_mortality import utils
from mimic4models.preprocessing import Discretizer, Normalizer
from mimic4models.fixed_horizon_icu_exit.matched_count_control.sampling import (
    DEFAULT_SAMPLING_SEED, SamplingReader, validate_sampling_args)


HORIZONS = FixedHorizonIcuExitReader.VALID_HORIZONS
DEFAULT_NORMALIZER_DIR = '/heinz-georgenas/users/mingzhul/Simultaneous-EHR/data/physionet.org/files/mimiciv/1.0/russo/normalizers/'


def build_reader(data_dir, split, horizon):
    if split == 'test':
        dataset_dir = os.path.join(data_dir, 'test')
        listfile = os.path.join(data_dir, 'test_listfile.csv')
    else:
        dataset_dir = os.path.join(data_dir, 'train')
        listfile = os.path.join(data_dir, '{}_listfile.csv'.format(split))
    return FixedHorizonIcuExitReader(dataset_dir=dataset_dir, listfile=listfile, horizon=horizon)


def validate_split(data_dir, split):
    readers = [build_reader(data_dir, split, horizon) for horizon in HORIZONS]
    names = readers[0].get_stay_names()
    for reader in readers[1:]:
        if reader.get_stay_names() != names:
            raise AssertionError('Horizon readers do not use the same stay set for {}.'.format(split))
    labels_by_horizon = [reader.get_labels() for reader in readers]
    for stay_i, labels in enumerate(zip(*labels_by_horizon)):
        if list(labels) != sorted(labels):
            raise AssertionError('Labels are not monotonic for {} row {}.'.format(split, stay_i))
    return readers


def validate_data(data_dir):
    for split in ('train', 'val', 'test'):
        validate_split(data_dir, split)


def default_normalizer_state_path(normalizer_dir, timestep, imputation, n_examples,
                                  sampling_strategy='none', sampling_interval=None,
                                  sampling_seed=DEFAULT_SAMPLING_SEED):
    if sampling_strategy == 'none':
        file_name = 'fixed_horizon_icu_exit_ts:{:.2f}_impute:{}_start:zero_masks:True_n:{}.normalizer'.format(
            timestep, imputation, n_examples)
    else:
        seed_part = ''
        if sampling_strategy == 'random_matched':
            seed_part = '_seed:{}'.format(sampling_seed)
        file_name = ('fixed_horizon_icu_exit_sampling:{}_r:{}{}_ts:{:.2f}'
                     '_impute:{}_start:zero_masks:True_n:{}.normalizer').format(
                         sampling_strategy, int(sampling_interval), seed_part,
                         timestep, imputation, n_examples)
    return os.path.join(normalizer_dir, file_name)


def print_stats(data_dir):
    validate_data(data_dir)
    for split in ('train', 'val', 'test'):
        print('\n{}:'.format(split))
        for horizon in HORIZONS:
            reader = build_reader(data_dir, split, horizon)
            labels = np.array(reader.get_labels(), dtype=int)
            positives = int(labels.sum())
            total = int(labels.shape[0])
            prevalence = float(positives) / total if total else 0.0
            print('  horizon={:>3}h n={} positives={} prevalence={:.6f}'.format(
                horizon, total, positives, prevalence))


def maybe_wrap_reader(reader, sampling_strategy, sampling_interval, sampling_seed):
    if sampling_strategy == 'none':
        return reader
    return SamplingReader(reader, sampling_strategy, int(sampling_interval), sampling_seed)


def load_train_val_raw(train_reader, val_reader, discretizer, normalizer, small_part):
    train_raw = utils.load_data(train_reader, discretizer, normalizer, small_part)
    val_raw = utils.load_data(val_reader, discretizer, normalizer, small_part)
    return train_raw, val_raw


def load_test_raw(test_reader, discretizer, normalizer, small_part):
    return utils.load_data(test_reader, discretizer, normalizer, small_part,
                           return_names=True)


def main():
    parser = argparse.ArgumentParser()
    common_utils.add_common_arguments(parser)
    parser.add_argument('--horizon', type=int, default=24,
                        choices=HORIZONS,
                        help='Prediction horizon in hours.')
    parser.add_argument('--target_repl_coef', type=float, default=0.0)
    parser.add_argument('--data', type=str, help='Path to the existing length-of-stay task data',
                        default=os.path.join(os.path.dirname(__file__), '../../data/length-of-stay/'))
    parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                        default='.')
    parser.add_argument('--normalizer_dir', type=str, default=DEFAULT_NORMALIZER_DIR,
                        help='Directory containing fixed-horizon ICU-exit normalizer states.')
    parser.add_argument('--seed', type=int, default=49297)
    parser.add_argument('--sampling_strategy', type=str, default='none',
                        choices=['none', 'structured', 'random_matched'],
                        help='Matched-count sampling intervention before downstream discretization.')
    parser.add_argument('--sampling_interval', type=float, default=4.0,
                        choices=[2.0, 4.0, 8.0],
                        help='Coarse interval used only for measurement selection.')
    parser.add_argument('--sampling_seed', type=int, default=DEFAULT_SAMPLING_SEED,
                        help='Seed for random matched-count sampling; independent of model seed.')
    parser.add_argument('--print_stats', action='store_true',
                        help='Print fixed-24h cohort counts and prevalence for every horizon, then exit.')
    for action in parser._actions:
        if action.dest == 'network':
            action.required = False

    args = parser.parse_args()

    if not args.print_stats and args.network is None:
        parser.error('--network is required unless --print_stats is used')

    print(args)

    validate_sampling_args(args.sampling_strategy, args.sampling_interval, args.timestep)

    if args.print_stats:
        print_stats(args.data)
        return

    import tensorflow as tf
    from keras.callbacks import CSVLogger, ModelCheckpoint
    from mimic4models import keras_utils

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    validate_data(args.data)

    if args.small_part:
        args.save_every = 2**30

    target_repl = (args.target_repl_coef > 0.0 and args.mode == 'train')

    train_reader = build_reader(args.data, 'train', args.horizon)
    train_reader = maybe_wrap_reader(train_reader, args.sampling_strategy,
                                     args.sampling_interval, args.sampling_seed)

    discretizer = Discretizer(timestep=float(args.timestep),
                              store_masks=True,
                              impute_strategy='previous',
                              start_time='zero')

    first_train = train_reader.read_example(0)
    discretizer_header = discretizer.transform(first_train["X"], end=first_train["t"])[1].split(',')
    cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

    normalizer = Normalizer(fields=cont_channels)
    normalizer_state = args.normalizer_state
    if normalizer_state is None:
        normalizer_state = default_normalizer_state_path(
            args.normalizer_dir, args.timestep, args.imputation,
            train_reader.get_number_of_examples(), args.sampling_strategy,
            args.sampling_interval, args.sampling_seed)
    if not os.path.exists(normalizer_state):
        raise IOError('Normalizer state file does not exist: {}'.format(normalizer_state))
    normalizer.load_params(normalizer_state)

    args_dict = dict(args._get_kwargs())
    args_dict['header'] = discretizer_header
    args_dict['task'] = 'ihm'
    args_dict['target_repl'] = target_repl
    args_dict['sampling_strategy'] = args.sampling_strategy
    args_dict['sampling_interval'] = args.sampling_interval
    args_dict['sampling_seed'] = args.sampling_seed

    print("==> using model {}".format(args.network))
    model_module = imp.load_source(os.path.basename(args.network), args.network)
    model = model_module.Network(**args_dict)
    sampling_suffix = ""
    if args.sampling_strategy != 'none':
        sampling_suffix = ".sample{}.r{}".format(args.sampling_strategy, int(args.sampling_interval))
        if args.sampling_strategy == 'random_matched':
            sampling_suffix += ".sseed{}".format(args.sampling_seed)
    suffix = ".h{}.bs{}{}{}.ts{}{}{}.seed{}".format(args.horizon,
                                                       args.batch_size,
                                                       ".L1{}".format(args.l1) if args.l1 > 0 else "",
                                                       ".L2{}".format(args.l2) if args.l2 > 0 else "",
                                                       args.timestep,
                                                       sampling_suffix,
                                                       ".trc{}".format(args.target_repl_coef) if args.target_repl_coef > 0 else "",
                                                       args.seed)
    model.final_name = args.prefix + model.say_name() + suffix
    print("==> model.final_name:", model.final_name)

    print("==> compiling the model")
    optimizer_config = {'class_name': args.optimizer,
                        'config': {'lr': args.lr,
                                   'beta_1': args.beta_1}}

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

    n_trained_chunks = 0
    if args.load_state != "":
        model.load_weights(args.load_state)
        n_trained_chunks = int(re.match(".*epoch([0-9]+).*", args.load_state).group(1))

    if args.mode == 'train':
        val_reader = build_reader(args.data, 'val', args.horizon)
        val_reader = maybe_wrap_reader(val_reader, args.sampling_strategy,
                                       args.sampling_interval, args.sampling_seed)
        train_raw, val_raw = load_train_val_raw(train_reader, val_reader,
                                                discretizer, normalizer, args.small_part)

        if target_repl:
            T = train_raw[0][0].shape[0]

            def extend_labels(data):
                data = list(data)
                labels = np.array(data[1])
                data[1] = [labels, None]
                data[1][1] = np.expand_dims(labels, axis=-1).repeat(T, axis=1)
                data[1][1] = np.expand_dims(data[1][1], axis=-1)
                return data

            train_raw = extend_labels(train_raw)
            val_raw = extend_labels(val_raw)

        path = os.path.join(args.output_dir, 'keras_states/' + model.final_name + '.epoch{epoch}.test{val_loss}.state')

        metrics_callback = keras_utils.InHospitalMortalityMetrics(train_data=train_raw,
                                                                  val_data=val_raw,
                                                                  target_repl=(args.target_repl_coef > 0),
                                                                  batch_size=args.batch_size,
                                                                  verbose=args.verbose)
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        saver = ModelCheckpoint(path, verbose=1, period=args.save_every)

        keras_logs = os.path.join(args.output_dir, 'keras_logs')
        if not os.path.exists(keras_logs):
            os.makedirs(keras_logs)
        csv_logger = CSVLogger(os.path.join(keras_logs, model.final_name + '.csv'),
                               append=True, separator=';')

        print("==> training")
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
        test_reader = build_reader(args.data, 'test', args.horizon)
        test_reader = maybe_wrap_reader(test_reader, args.sampling_strategy,
                                        args.sampling_interval, args.sampling_seed)
        ret = load_test_raw(test_reader, discretizer, normalizer, args.small_part)

        data = ret["data"][0]
        labels = ret["data"][1]
        names = ret["names"]

        predictions = model.predict(data, batch_size=args.batch_size, verbose=1)
        predictions = np.array(predictions)[:, 0]
        metrics.print_metrics_binary(labels, predictions)

        path = os.path.join(args.output_dir, "test_predictions", os.path.basename(args.load_state)) + ".csv"
        utils.save_results(names, predictions, labels, path)

    else:
        raise ValueError("Wrong value for args.mode")


if __name__ == '__main__':
    main()
