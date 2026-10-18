from __future__ import absolute_import
from __future__ import print_function

import argparse
import imp
import os
import random
import re

import numpy as np

from mimic4benchmark.readers import FixedHorizonDecompensationReader
from mimic4models import common_utils
from mimic4models import metrics
from mimic4models.in_hospital_mortality import utils
from mimic4models.preprocessing import Discretizer, Normalizer
from mimic4models.fixed_horizon_icu_exit.main import (
    DEFAULT_NORMALIZER_DIR,
    default_normalizer_state_path,
    load_test_raw,
    load_train_val_raw,
    maybe_prepare_model_data,
    maybe_prepare_model_input,
    predict_raw_batches,
    raw_bucket_size,
    check_raw_grud_real_row_masks,
    check_raw_padding_prediction_equivalence,
)
from mimic4models.fixed_horizon_icu_exit.raw import (
    RawBatchSequence, RawObservedNormalizer, RawSequenceEncoder,
    is_raw_timestep,
)


HORIZONS = FixedHorizonDecompensationReader.VALID_HORIZONS


def build_reader(data_dir, split, horizon, metadata_path):
    if split == 'test':
        dataset_dir = os.path.join(data_dir, 'test')
        listfile = os.path.join(data_dir, 'test_listfile.csv')
    else:
        dataset_dir = os.path.join(data_dir, 'train')
        listfile = os.path.join(data_dir, '{}_listfile.csv'.format(split))
    return FixedHorizonDecompensationReader(
        dataset_dir=dataset_dir, listfile=listfile,
        horizon=horizon, metadata_path=metadata_path)


def validate_split(data_dir, split, horizon, metadata_path):
    readers = [build_reader(data_dir, split, current_horizon, metadata_path)
               for current_horizon in HORIZONS]
    names = readers[0].get_stay_names()
    for reader in readers[1:]:
        if reader.get_stay_names() != names:
            raise AssertionError(
                'Horizon readers do not use the same stay set for {}.'.format(split))
    labels_by_horizon = [reader.get_labels() for reader in readers]
    for stay_i, labels in enumerate(zip(*labels_by_horizon)):
        if list(labels) != sorted(labels):
            raise AssertionError(
                'Labels are not monotonic for {} row {}.'.format(split, stay_i))


def validate_data(data_dir, metadata_path):
    for split in ('train', 'val', 'test'):
        validate_split(data_dir, split, HORIZONS[0], metadata_path)


def print_stats(data_dir, metadata_path):
    validate_data(data_dir, metadata_path)
    for split in ('train', 'val', 'test'):
        print('\n{}:'.format(split))
        for horizon in HORIZONS:
            reader = build_reader(data_dir, split, horizon, metadata_path)
            labels = np.asarray(reader.get_labels(), dtype=int)
            positives = int(labels.sum())
            total = int(labels.shape[0])
            prevalence = float(positives) / total if total else 0.0
            print('  horizon={:>3}h n={} positives={} prevalence={:.6f}'.format(
                horizon, total, positives, prevalence))


def normalizer_path(normalizer_dir, timestep, imputation, n_examples):
    return default_normalizer_state_path(
        normalizer_dir, timestep, imputation, n_examples,
        task_name='fixed_horizon_decompensation')


def main():
    parser = argparse.ArgumentParser()
    common_utils.add_common_arguments(parser)
    parser.add_argument('--horizon', type=int, default=24, choices=HORIZONS,
                        help='Prediction horizon in hours.')
    parser.add_argument('--target_repl_coef', type=float, default=0.0)
    parser.add_argument('--data', type=str, required=True,
                        help='Path to the existing decompensation task data.')
    parser.add_argument('--metadata_path', type=str, default=None,
                        help='Path to mortality_metadata.csv.')
    parser.add_argument('--output_dir', type=str, default='.')
    parser.add_argument('--normalizer_dir', type=str, default=DEFAULT_NORMALIZER_DIR)
    parser.add_argument('--seed', type=int, default=49297)
    parser.add_argument('--print_stats', action='store_true',
                        help='Print fixed-24h cohort counts and prevalence for every horizon, then exit.')
    for action in parser._actions:
        if action.dest == 'network':
            action.required = False

    args = parser.parse_args()
    if args.metadata_path is None:
        args.metadata_path = os.path.join(args.data, 'mortality_metadata.csv')
    if not args.print_stats and args.network is None:
        parser.error('--network is required unless --print_stats is used')
    print(args)

    if args.print_stats:
        print_stats(args.data, args.metadata_path)
        return

    import tensorflow as tf
    from keras import backend as K
    from keras.callbacks import CSVLogger, ModelCheckpoint
    from mimic4models import keras_utils

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    validate_data(args.data, args.metadata_path)
    if args.small_part:
        args.save_every = 2 ** 30
    target_repl = args.target_repl_coef > 0.0 and args.mode == 'train'
    raw_mode = is_raw_timestep(args.timestep)

    train_reader = build_reader(args.data, 'train', args.horizon, args.metadata_path)
    if raw_mode:
        representation = RawSequenceEncoder()
        feature_header = representation.header()
        normalizer = RawObservedNormalizer(
            representation.continuous_value_fields(),
            representation.continuous_mask_fields())
    else:
        representation = Discretizer(
            timestep=float(args.timestep), store_masks=True,
            impute_strategy='previous', start_time='zero')
        first_train = train_reader.read_example(0)
        feature_header = representation.transform(
            first_train['X'], end=first_train['t'])[1].split(',')
        cont_channels = [i for i, x in enumerate(feature_header)
                         if '->' not in x]
        normalizer = Normalizer(fields=cont_channels)

    normalizer_state = args.normalizer_state
    if normalizer_state is None:
        normalizer_state = normalizer_path(
            args.normalizer_dir, args.timestep, args.imputation,
            train_reader.get_number_of_examples())
    if not os.path.exists(normalizer_state):
        raise IOError('Normalizer state file does not exist: {}'.format(normalizer_state))
    normalizer.load_params(normalizer_state)

    args_dict = dict(args._get_kwargs())
    args_dict['header'] = feature_header
    args_dict['task'] = 'ihm'
    args_dict['target_repl'] = target_repl
    model_module = imp.load_source(os.path.basename(args.network), args.network)
    model = model_module.Network(**args_dict)
    model.final_name = '{}{}.h{}.bs{}{}{}.ts{}.seed{}'.format(
        args.prefix, model.say_name(), args.horizon, args.batch_size,
        '.L1{}'.format(args.l1) if args.l1 > 0 else '',
        '.L2{}'.format(args.l2) if args.l2 > 0 else '',
        args.timestep, args.seed)
    print('==> model.final_name:', model.final_name)

    optimizer_config = {
        'class_name': args.optimizer,
        'config': {'lr': args.lr, 'beta_1': args.beta_1},
    }
    if target_repl:
        loss = ['binary_crossentropy'] * 2
        loss_weights = [1 - args.target_repl_coef, args.target_repl_coef]
    else:
        loss = 'binary_crossentropy'
        loss_weights = None
    model.compile(optimizer=optimizer_config, loss=loss, loss_weights=loss_weights)
    model.summary()

    n_trained_chunks = 0
    if args.load_state != '':
        keras_utils.patch_legacy_keras_h5py_attrs()
        model.load_weights(args.load_state)
        n_trained_chunks = int(re.match('.*epoch([0-9]+).*', args.load_state).group(1))

    if args.mode == 'train':
        val_reader = build_reader(args.data, 'val', args.horizon, args.metadata_path)
        train_raw, val_raw = load_train_val_raw(
            train_reader, val_reader, representation, normalizer,
            args.small_part, raw_mode=raw_mode)
        raw_uses_timestamps = raw_mode and bool(
            getattr(model_module, 'USES_RAW_TIMESTAMPS', False))
        if raw_mode and raw_uses_timestamps:
            check_raw_grud_real_row_masks(model_module, train_raw[0], feature_header, 'train')
            check_raw_grud_real_row_masks(model_module, val_raw[0], feature_header, 'validation')
        raw_prepare_input = None
        if raw_mode and raw_uses_timestamps and hasattr(model_module, 'prepare_input'):
            raw_prepare_input = lambda X: model_module.prepare_input(
                X, header=feature_header, timestep=args.timestep)
        if not raw_mode:
            train_raw = maybe_prepare_model_data(
                model_module, train_raw, feature_header, args.timestep)
            val_raw = maybe_prepare_model_data(
                model_module, val_raw, feature_header, args.timestep)

        train_sequence = None
        val_sequence = None
        if raw_mode:
            train_sequence = RawBatchSequence(
                train_raw[0], train_raw[1], args.batch_size, shuffle=True,
                bucket_size=raw_bucket_size(args.batch_size), seed=args.seed,
                target_repl=target_repl, prepare_input=raw_prepare_input)
            val_sequence = RawBatchSequence(
                val_raw[0], val_raw[1], args.batch_size, shuffle=False,
                bucket_size=raw_bucket_size(args.batch_size),
                target_repl=target_repl, prepare_input=raw_prepare_input)
            check_raw_padding_prediction_equivalence(
                model, train_raw[0], train_raw[1], args.batch_size,
                prepare_input=raw_prepare_input)
        elif target_repl:
            T = train_raw[0][0].shape[0] if isinstance(train_raw[0], list) else train_raw[0].shape[1]

            def extend_labels(data):
                data = list(data)
                labels = np.asarray(data[1])
                data[1] = [labels, np.expand_dims(labels, -1).repeat(T, axis=1)]
                data[1][1] = np.expand_dims(data[1][1], -1)
                return data

            train_raw = extend_labels(train_raw)
            val_raw = extend_labels(val_raw)

        csv_append = keras_utils.prepare_keras_training_run(
            args.output_dir, model.final_name, args.load_state, args.mode)
        state_path = os.path.join(
            args.output_dir, 'keras_states',
            model.final_name + '.epoch{epoch}.test{val_loss}.state')
        metrics_callback = keras_utils.InHospitalMortalityMetrics(
            train_data=train_sequence or train_raw,
            val_data=val_sequence or val_raw,
            target_repl=args.target_repl_coef > 0,
            batch_size=args.batch_size, verbose=args.verbose,
            skip_train_metrics=raw_mode)
        common_utils.create_directory(os.path.dirname(state_path))
        saver = ModelCheckpoint(state_path, verbose=1, period=args.save_every)
        log_dir = os.path.join(args.output_dir, 'keras_logs')
        common_utils.create_directory(log_dir)
        csv_logger = CSVLogger(
            os.path.join(log_dir, model.final_name + '.csv'),
            append=csv_append, separator=';')

        if raw_mode:
            model.fit_generator(
                generator=train_sequence.iter_batches(),
                steps_per_epoch=len(train_sequence),
                validation_data=val_sequence.iter_batches(),
                validation_steps=len(val_sequence),
                epochs=n_trained_chunks + args.epochs,
                initial_epoch=n_trained_chunks,
                callbacks=[metrics_callback, saver, csv_logger],
                verbose=args.verbose)
        else:
            model.fit(
                x=train_raw[0], y=train_raw[1], validation_data=val_raw,
                epochs=n_trained_chunks + args.epochs,
                initial_epoch=n_trained_chunks,
                callbacks=[metrics_callback, saver, csv_logger],
                shuffle=True, verbose=args.verbose, batch_size=args.batch_size)

    elif args.mode == 'test':
        test_reader = build_reader(args.data, 'test', args.horizon, args.metadata_path)
        ret = load_test_raw(test_reader, representation, normalizer,
                            args.small_part, raw_mode=raw_mode)
        data = ret['data'][0]
        labels = ret['data'][1]
        names = ret['names']
        raw_uses_timestamps = raw_mode and bool(
            getattr(model_module, 'USES_RAW_TIMESTAMPS', False))
        raw_prepare_input = None
        if raw_mode and raw_uses_timestamps and hasattr(model_module, 'prepare_input'):
            raw_prepare_input = lambda X: model_module.prepare_input(
                X, header=feature_header, timestep=args.timestep)
        if raw_mode:
            predictions = predict_raw_batches(
                model, data, labels, args.batch_size,
                prepare_input=raw_prepare_input)
        else:
            data = maybe_prepare_model_input(
                model_module, data, feature_header, args.timestep)
            predictions = np.asarray(model.predict(
                data, batch_size=args.batch_size, verbose=1))[:, 0]
        metrics.print_metrics_binary(labels, predictions)
        path = os.path.join(
            args.output_dir, 'test_predictions',
            os.path.basename(args.load_state) + '.csv')
        utils.save_results(names, predictions, labels, path)
    else:
        raise ValueError('Wrong value for args.mode')


if __name__ == '__main__':
    main()
