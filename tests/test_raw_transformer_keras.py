from __future__ import absolute_import

import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from keras import backend as K
    from mimic4models.fixed_horizon_icu_exit.raw import (
        RawBatchSequence, RawSequenceEncoder, pad_raw_batch)
    from mimic4models.fixed_horizon_icu_exit.matched_count_control.sampling import (
        structured_sample)
    from mimic4models.keras_models.grud import Network as GrudNetwork
    from mimic4models.keras_models.lstm import Network as LstmNetwork
    from mimic4models.keras_models.raw_transformer import (
        Network, prepare_input)
    KERAS_IMPORT_ERROR = None
except ImportError as exc:
    K = None
    Network = None
    KERAS_IMPORT_ERROR = exc


@unittest.skipUnless(Network is not None,
                     'legacy Keras is unavailable: {}'.format(KERAS_IMPORT_ERROR))
class RawTransformerKerasTests(unittest.TestCase):

    def setUp(self):
        self.timestamp_encoder = RawSequenceEncoder(include_timestamps=True)
        self.plain_encoder = RawSequenceEncoder()
        self.header = ['Hours'] + self.timestamp_encoder._id_to_channel
        self.channel = next(name for name in self.timestamp_encoder._id_to_channel
                            if not self.timestamp_encoder._is_categorical_channel[name])
        self.channel_i = self.header.index(self.channel)

    def _rows(self, times, values):
        rows = []
        for time, value in zip(times, values):
            row = [''] * len(self.header)
            row[0] = str(time)
            row[self.channel_i] = str(value)
            rows.append(row)
        return np.asarray(rows, dtype=object)

    def _timestamp_sequence(self, times, values):
        return self.timestamp_encoder.transform(
            self._rows(times, values), header=self.header, end=24.0)[0]

    def _model(self, mode='timestamp', dropout=0.0):
        encoder = self.timestamp_encoder if mode == 'timestamp' else self.plain_encoder
        return Network(dropout=dropout, task='ihm', header=encoder.header(),
                       input_dim=encoder.feature_dim,
                       positional_encoding=mode)

    def test_timestamp_and_index_models_have_same_parameter_count(self):
        timestamp = self._model('timestamp')
        index = self._model('index')
        self.assertEqual(timestamp.count_params(), index.count_params())
        self.assertEqual(timestamp.count_params(), 7521)
        self.assertNotEqual(timestamp.say_name(), index.say_name())
        self.assertIn('pe_timestamp', timestamp.say_name())
        self.assertIn('pe_index', index.say_name())

    def test_transformer_is_parameter_matched_to_existing_baselines(self):
        lstm = LstmNetwork(dim=16, batch_norm=False, dropout=0.3,
                           rec_dropout=0.0, task='ihm', depth=2,
                           input_dim=self.timestamp_encoder.feature_dim)
        grud = GrudNetwork(dim=16, batch_norm=False, dropout=0.3,
                           rec_dropout=0.0, task='ihm', depth=1,
                           input_dim=self.timestamp_encoder.feature_dim,
                           header=self.timestamp_encoder.header(),
                           raw_sequence_mask=True)
        transformer = self._model('timestamp', dropout=0.3)
        self.assertEqual(lstm.count_params(), 7633)
        self.assertEqual(grud.count_params(), 7575)
        self.assertLess(abs(transformer.count_params() - lstm.count_params()) /
                        float(lstm.count_params()), 0.05)
        self.assertLess(abs(transformer.count_params() - grud.count_params()) /
                        float(grud.count_params()), 0.05)

    def test_forward_training_step_and_single_row_sequence(self):
        model = self._model('timestamp', dropout=0.0)
        model.compile(optimizer='adam', loss='binary_crossentropy')
        one = self._timestamp_sequence([0.0], [0.0])
        irregular = self._timestamp_sequence([0.5, 3.25, 11.0], [70.0, 71.0, 72.0])
        batch = pad_raw_batch([one, irregular])
        loss = model.train_on_batch(prepare_input(batch, self.header),
                                    np.asarray([0.0, 1.0]))
        self.assertTrue(np.isfinite(float(loss)))
        self.assertEqual(model.predict(prepare_input(batch, self.header), verbose=0).shape,
                         (2, 1))

    def test_prediction_with_dropout_uses_inference_learning_phase(self):
        model = self._model('timestamp', dropout=0.3)
        sequence = self._timestamp_sequence([0.0, 2.5], [70.0, 71.0])
        prediction = model.predict(
            prepare_input(sequence[None, :, :], self.header), verbose=0)
        self.assertEqual(prediction.shape, (1, 1))

    def test_padding_is_excluded_from_attention_and_pooling(self):
        model = self._model('timestamp', dropout=0.0)
        sequence = self._timestamp_sequence([0.0, 2.5, 8.0], [0.0, 71.0, 72.0])
        short_batch = sequence[None, :, :]
        padded_batch = np.pad(short_batch, ((0, 0), (0, 3), (0, 0)), mode='constant')
        short_prediction = model.predict(prepare_input(short_batch, self.header), verbose=0)
        padded_prediction = model.predict(prepare_input(padded_batch, self.header), verbose=0)
        self.assertTrue(np.allclose(short_prediction, padded_prediction,
                                   atol=1e-6, rtol=1e-5))

    def test_zero_timestamp_valid_row_is_not_padding(self):
        model = self._model('timestamp', dropout=0.0)
        valid = self._timestamp_sequence([0.0], [0.0])
        padded = pad_raw_batch([valid, np.zeros((0, valid.shape[1]), dtype=np.float32)])
        self.assertTrue(np.any(valid[0, :-1] != 0.0))
        self.assertEqual(padded.shape[1], 1)
        self.assertEqual(model.predict(prepare_input(padded, self.header), verbose=0).shape,
                         (2, 1))

    def test_timestamp_pe_uses_retained_dense_to_sparse_times(self):
        dense = self._rows([0.5, 1.0, 3.0, 3.5, 4.1, 7.0],
                           [501, 999, 503, 535, 541, 570])
        sparse, _ = structured_sample(dense, self.header, 4, end=24.0)
        dense_encoded = self.timestamp_encoder.transform(
            dense, header=self.header, end=24.0)[0]
        sparse_encoded = self.timestamp_encoder.transform(
            sparse, header=self.header, end=24.0)[0]
        dense_times = dense_encoded[[3, 5], -1]
        sparse_times = sparse_encoded[:, -1]
        self.assertTrue(np.array_equal(dense_times, sparse_times))

        model = self._model('timestamp', dropout=0.0)
        dense_input = prepare_input(dense_encoded[None, :, :], self.header)
        sparse_input = prepare_input(sparse_encoded[None, :, :], self.header)
        self.assertTrue(np.array_equal(dense_input[1][0, [3, 5], 0],
                                       sparse_input[1][0, :, 0]))
        self.assertEqual(model.predict(sparse_input, verbose=0).shape, (1, 1))

    def test_raw_batch_sequence_dense_to_sparse_smoke(self):
        dense = self._timestamp_sequence([0.5, 1.0, 3.0, 3.5, 4.1, 7.0],
                                         [501, 999, 503, 535, 541, 570])
        raw_sparse, _ = structured_sample(
            self._rows([0.5, 1.0, 3.0, 3.5, 4.1, 7.0],
                       [501, 999, 503, 535, 541, 570]),
            self.header, 4, end=24.0)
        sparse = self.timestamp_encoder.transform(
            raw_sparse, header=self.header, end=24.0)[0]
        model = self._model('timestamp', dropout=0.0)
        model.compile(optimizer='adam', loss='binary_crossentropy')
        prepare = lambda batch: prepare_input(batch, self.header)
        batches = RawBatchSequence([dense, sparse], [0.0, 1.0], batch_size=2,
                                   shuffle=False, prepare_input=prepare)
        x_batch, y_batch = batches[0]
        predictions = model.predict(x_batch, verbose=0)
        self.assertEqual(predictions.shape, (2, 1))
        self.assertEqual(y_batch.shape, (2,))


if __name__ == '__main__':
    unittest.main()
