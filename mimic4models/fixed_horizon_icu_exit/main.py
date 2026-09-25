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
from mimic4models.fixed_horizon_icu_exit.raw import (
    RawBatchSequence, RawObservedNormalizer, RawSequenceEncoder, is_raw_timestep,
    load_raw_data, load_raw_data_unpadded, print_raw_sequence_length_stats)
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
    if is_raw_timestep(timestep):
        if sampling_strategy == 'none':
            file_name = 'fixed_horizon_icu_exit_raw_ts:0.00_observed_only_n:{}.normalizer'.format(
                n_examples)
        else:
            seed_part = ''
            if sampling_strategy == 'random_matched':
                seed_part = '_seed:{}'.format(sampling_seed)
            file_name = ('fixed_horizon_icu_exit_raw_sampling:{}_r:{}{}'
                         '_ts:0.00_observed_only_n:{}.normalizer').format(
                             sampling_strategy, int(sampling_interval), seed_part, n_examples)
    elif sampling_strategy == 'none':
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


def resolve_test_sampling_regime(sampling_strategy, sampling_interval, sampling_seed,
                                 test_sampling_strategy=None,
                                 test_sampling_interval=None,
                                 test_sampling_seed=None):
    strategy = test_sampling_strategy if test_sampling_strategy is not None else sampling_strategy
    interval = test_sampling_interval if test_sampling_interval is not None else sampling_interval
    seed = test_sampling_seed if test_sampling_seed is not None else sampling_seed
    return strategy, interval, seed


def resolve_test_sampling_args(args):
    return resolve_test_sampling_regime(
        args.sampling_strategy, args.sampling_interval, args.sampling_seed,
        args.test_sampling_strategy, args.test_sampling_interval,
        args.test_sampling_seed)


def validate_all_sampling_args(args):
    validate_sampling_args(args.sampling_strategy, args.sampling_interval, args.timestep)
    test_sampling_strategy, test_sampling_interval, test_sampling_seed = resolve_test_sampling_args(args)
    validate_sampling_args(test_sampling_strategy, test_sampling_interval, args.timestep)
    return test_sampling_strategy, test_sampling_interval, test_sampling_seed


def sampling_regime_label(sampling_strategy, sampling_interval, sampling_seed):
    if sampling_strategy == 'none':
        return 'none'
    label = '{}-r{}'.format(sampling_strategy, int(sampling_interval))
    if sampling_strategy == 'random_matched':
        label += '-sseed{}'.format(sampling_seed)
    return label


def effective_sampling_regime(sampling_strategy, sampling_interval, sampling_seed):
    if sampling_strategy == 'none':
        return ('none',)
    if sampling_strategy == 'structured':
        return ('structured', int(sampling_interval))
    if sampling_strategy == 'random_matched':
        return ('random_matched', int(sampling_interval), sampling_seed)
    raise ValueError('Unknown sampling_strategy {}'.format(sampling_strategy))


def sampling_regimes_differ(train_sampling_strategy, train_sampling_interval, train_sampling_seed,
                            test_sampling_strategy, test_sampling_interval, test_sampling_seed):
    train_regime = effective_sampling_regime(
        train_sampling_strategy, train_sampling_interval, train_sampling_seed)
    test_regime = effective_sampling_regime(
        test_sampling_strategy, test_sampling_interval, test_sampling_seed)
    return train_regime != test_regime


def test_prediction_path(output_dir, load_state, train_sampling_strategy,
                         train_sampling_interval, train_sampling_seed,
                         test_sampling_strategy, test_sampling_interval,
                         test_sampling_seed):
    path = os.path.join(output_dir, "test_predictions", os.path.basename(load_state))
    if not sampling_regimes_differ(train_sampling_strategy, train_sampling_interval, train_sampling_seed,
                                   test_sampling_strategy, test_sampling_interval, test_sampling_seed):
        return path + ".csv"
    suffix = 'testsample-{}'.format(
        sampling_regime_label(test_sampling_strategy, test_sampling_interval, test_sampling_seed))
    return path + ".{}.csv".format(suffix)


def load_train_val_raw(train_reader, val_reader, representation, normalizer, small_part, raw_mode=False):
    if raw_mode:
        train_raw = load_raw_data_unpadded(train_reader, representation, normalizer, small_part)
        val_raw = load_raw_data_unpadded(val_reader, representation, normalizer, small_part)
    else:
        train_raw = utils.load_data(train_reader, representation, normalizer, small_part)
        val_raw = utils.load_data(val_reader, representation, normalizer, small_part)
    return train_raw, val_raw


def load_test_raw(test_reader, representation, normalizer, small_part, raw_mode=False):
    if raw_mode:
        return load_raw_data_unpadded(test_reader, representation, normalizer, small_part,
                                      return_names=True)
    return utils.load_data(test_reader, representation, normalizer, small_part,
                           return_names=True)


def maybe_prepare_model_input(model_module, X, header, timestep):
    if hasattr(model_module, 'prepare_input'):
        return model_module.prepare_input(X, header=header, timestep=timestep)
    return X


def maybe_prepare_model_data(model_module, data, header, timestep):
    return (maybe_prepare_model_input(model_module, data[0], header, timestep), data[1])


def check_raw_grud_real_row_masks(model_module, sequences, header, split_name):
    if not hasattr(model_module, 'print_raw_grud_real_row_mask_sanity'):
        raise ValueError('Raw GRU-D timestamp model must expose '
                         'print_raw_grud_real_row_mask_sanity().')
    report = model_module.print_raw_grud_real_row_mask_sanity(
        sequences, header, label=split_name)
    if report['all_zero_real_rows'] > 0:
        raise ValueError(
            'Found {} real raw rows with no modeled observations in {} data. '
            'First violations: {}'.format(
                report['all_zero_real_rows'], split_name, report['violations']))
    return report


def raw_bucket_size(batch_size):
    return max(int(batch_size) * 100, int(batch_size))


def _flat_model_predictions(outputs):
    if isinstance(outputs, list):
        outputs = outputs[0]
    return np.array(outputs).flatten()


def predict_raw_batches(model, sequences, labels, batch_size, prepare_input=None):
    data_seq = RawBatchSequence(sequences, labels, batch_size=batch_size, shuffle=False,
                                bucket_size=raw_bucket_size(batch_size), target_repl=False,
                                prepare_input=prepare_input)
    predictions = np.zeros((len(sequences),), dtype=float)
    for batch_i in range(len(data_seq)):
        x_batch, _ = data_seq[batch_i]
        batch_pred = model.predict(x_batch, batch_size=batch_size, verbose=0)
        predictions[data_seq.batch_indices(batch_i)] = _flat_model_predictions(batch_pred)
    return predictions


def check_raw_padding_prediction_equivalence(model, sequences, labels, batch_size, max_examples=32,
                                             atol=1e-6, rtol=1e-5, prepare_input=None):
    if len(sequences) == 0:
        return np.nan
    lengths = np.asarray([x.shape[0] for x in sequences], dtype=int)
    n_check = min(int(max_examples), len(sequences))
    sorted_indices = np.argsort(lengths)
    if n_check == len(sequences):
        check_indices = sorted_indices
    else:
        positions = np.linspace(0, len(sorted_indices) - 1, n_check).astype(int)
        check_indices = sorted_indices[positions]
    check_sequences = [sequences[i] for i in check_indices]
    check_labels = np.asarray(labels)[check_indices]

    global_x = common_utils.pad_zeros(check_sequences)
    if global_x.dtype != np.float32:
        global_x = global_x.astype(np.float32)
    for row_i, seq in enumerate(check_sequences):
        if seq.shape[0] < global_x.shape[1]:
            padding = global_x[row_i, seq.shape[0]:]
            if not np.all(padding == 0.0):
                raise AssertionError('Global padding contains non-zero rows')
    global_input = prepare_input(global_x) if prepare_input is not None else global_x
    pred_global = _flat_model_predictions(model.predict(global_input, batch_size=batch_size, verbose=0))
    pred_local = predict_raw_batches(model, check_sequences, check_labels, batch_size,
                                     prepare_input=prepare_input)

    if pred_global.shape[0] != len(check_sequences) or pred_local.shape[0] != len(check_sequences):
        raise AssertionError('Raw padding check prediction count mismatch')
    if check_labels.shape[0] != len(check_sequences):
        raise AssertionError('Raw padding check label count mismatch')
    max_abs_diff = float(np.max(np.abs(pred_global - pred_local))) if len(check_sequences) else 0.0
    if not np.allclose(pred_global, pred_local, atol=atol, rtol=rtol):
        raise AssertionError(
            'Global-vs-batch-local raw padding predictions differ; max_abs_diff={}'.format(max_abs_diff))
    print('Raw padding equivalence check: n={} max_abs_diff={:.8g}'.format(
        len(check_sequences), max_abs_diff))
    return max_abs_diff


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
    parser.add_argument('--test_sampling_strategy', type=str, default=None,
                        choices=['none', 'structured', 'random_matched'],
                        help='Test/deployment sampling intervention. Defaults to --sampling_strategy.')
    parser.add_argument('--test_sampling_interval', type=float, default=None,
                        choices=[2.0, 4.0, 8.0],
                        help='Test/deployment coarse interval. Defaults to --sampling_interval.')
    parser.add_argument('--test_sampling_seed', type=int, default=None,
                        help='Test/deployment random sampling seed. Defaults to --sampling_seed.')
    parser.add_argument('--print_stats', action='store_true',
                        help='Print fixed-24h cohort counts and prevalence for every horizon, then exit.')
    parser.add_argument('--print_raw_sequence_stats', action='store_true',
                        help='Print raw sequence-length statistics for the training split, then exit.')
    for action in parser._actions:
        if action.dest == 'network':
            action.required = False

    args = parser.parse_args()

    if not args.print_stats and not args.print_raw_sequence_stats and args.network is None:
        parser.error('--network is required unless --print_stats or --print_raw_sequence_stats is used')

    print(args)

    test_sampling_strategy, test_sampling_interval, test_sampling_seed = validate_all_sampling_args(args)

    if args.print_stats:
        print_stats(args.data)
        return

    raw_mode = is_raw_timestep(args.timestep)
    if args.print_raw_sequence_stats:
        if not raw_mode:
            raise ValueError('--print_raw_sequence_stats requires --timestep 0')
        stats_reader = build_reader(args.data, 'train', args.horizon)
        stats_reader = maybe_wrap_reader(stats_reader, args.sampling_strategy,
                                         args.sampling_interval, args.sampling_seed)
        print_raw_sequence_length_stats(stats_reader, RawSequenceEncoder(), args.small_part)
        return

    import tensorflow as tf
    from keras.callbacks import CSVLogger, ModelCheckpoint
    from mimic4models import keras_utils
    from keras import backend as K

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    validate_data(args.data)

    if args.small_part:
        args.save_every = 2**30

    target_repl = (args.target_repl_coef > 0.0 and args.mode == 'train')

    print("==> using model {}".format(args.network))
    model_module = imp.load_source(os.path.basename(args.network), args.network)
    raw_uses_timestamps = raw_mode and bool(getattr(model_module, 'USES_RAW_TIMESTAMPS', False))

    train_reader = build_reader(args.data, 'train', args.horizon)
    train_reader = maybe_wrap_reader(train_reader, args.sampling_strategy,
                                     args.sampling_interval, args.sampling_seed)

    if raw_mode:
        representation = RawSequenceEncoder(include_timestamps=raw_uses_timestamps)
        feature_header = representation.header()
        normalizer = RawObservedNormalizer(representation.continuous_value_fields(),
                                           representation.continuous_mask_fields())
    else:
        representation = Discretizer(timestep=float(args.timestep),
                                     store_masks=True,
                                     impute_strategy='previous',
                                     start_time='zero')
        first_train = train_reader.read_example(0)
        feature_header = representation.transform(first_train["X"], end=first_train["t"])[1].split(',')
        cont_channels = [i for (i, x) in enumerate(feature_header) if x.find("->") == -1]
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
    args_dict['header'] = feature_header
    args_dict['task'] = 'ihm'
    args_dict['target_repl'] = target_repl
    args_dict['sampling_strategy'] = args.sampling_strategy
    args_dict['sampling_interval'] = args.sampling_interval
    args_dict['sampling_seed'] = args.sampling_seed
    args_dict['raw_sequence_mask'] = raw_uses_timestamps

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
                                                representation, normalizer, args.small_part,
                                                raw_mode=raw_mode)
        if raw_mode and raw_uses_timestamps:
            check_raw_grud_real_row_masks(model_module, train_raw[0], feature_header, 'train')
            check_raw_grud_real_row_masks(model_module, val_raw[0], feature_header, 'validation')
        raw_prepare_input = None
        if raw_mode and raw_uses_timestamps and hasattr(model_module, 'prepare_input'):
            def raw_prepare_input(X):
                return model_module.prepare_input(X, header=feature_header, timestep=args.timestep)
        if not raw_mode:
            train_raw = maybe_prepare_model_data(model_module, train_raw, feature_header, args.timestep)
            val_raw = maybe_prepare_model_data(model_module, val_raw, feature_header, args.timestep)

        train_sequence = None
        val_sequence = None
        if raw_mode:
            train_sequence = RawBatchSequence(
                train_raw[0], train_raw[1], batch_size=args.batch_size,
                shuffle=True, bucket_size=raw_bucket_size(args.batch_size),
                seed=args.seed, target_repl=target_repl, prepare_input=raw_prepare_input)
            val_sequence = RawBatchSequence(
                val_raw[0], val_raw[1], batch_size=args.batch_size,
                shuffle=False, bucket_size=raw_bucket_size(args.batch_size),
                target_repl=target_repl, prepare_input=raw_prepare_input)
            train_sequence.print_diagnostics()
            check_raw_padding_prediction_equivalence(
                model, train_raw[0], train_raw[1], batch_size=args.batch_size,
                prepare_input=raw_prepare_input)
        elif target_repl:
            if isinstance(train_raw[0], list):
                T = train_raw[0][0].shape[1]
            else:
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

        csv_append = keras_utils.prepare_keras_training_run(
            args.output_dir, model.final_name, args.load_state, args.mode)
        path = os.path.join(args.output_dir, 'keras_states/' + model.final_name + '.epoch{epoch}.test{val_loss}.state')

        metrics_callback = keras_utils.InHospitalMortalityMetrics(train_data=train_sequence or train_raw,
                                                                  val_data=val_sequence or val_raw,
                                                                  target_repl=(args.target_repl_coef > 0),
                                                                  batch_size=args.batch_size,
                                                                  verbose=args.verbose,
                                                                  skip_train_metrics=raw_mode)
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
        test_reader = build_reader(args.data, 'test', args.horizon)
        test_reader = maybe_wrap_reader(test_reader, test_sampling_strategy,
                                        test_sampling_interval, test_sampling_seed)
        ret = load_test_raw(test_reader, representation, normalizer, args.small_part,
                            raw_mode=raw_mode)

        data = ret["data"][0]
        labels = ret["data"][1]
        names = ret["names"]

        if raw_mode and raw_uses_timestamps:
            check_raw_grud_real_row_masks(model_module, data, feature_header, 'test')

        if raw_mode:
            raw_prepare_input = None
            if raw_uses_timestamps and hasattr(model_module, 'prepare_input'):
                def raw_prepare_input(X):
                    return model_module.prepare_input(X, header=feature_header, timestep=args.timestep)
            predictions = predict_raw_batches(model, data, labels, args.batch_size,
                                              prepare_input=raw_prepare_input)
        else:
            data = maybe_prepare_model_input(model_module, data, feature_header, args.timestep)
            predictions = model.predict(data, batch_size=args.batch_size, verbose=1)
            predictions = np.array(predictions)[:, 0]
        metrics.print_metrics_binary(labels, predictions)

        path = test_prediction_path(args.output_dir, args.load_state,
                                    args.sampling_strategy, args.sampling_interval,
                                    args.sampling_seed, test_sampling_strategy,
                                    test_sampling_interval, test_sampling_seed)
        utils.save_results(names, predictions, labels, path)

    else:
        raise ValueError("Wrong value for args.mode")


if __name__ == '__main__':
    main()
