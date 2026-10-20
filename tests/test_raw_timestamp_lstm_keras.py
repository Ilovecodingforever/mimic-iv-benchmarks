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
    from mimic4models.keras_models.grud import split_grud_inputs
    from mimic4models.keras_models.lstm import Network
    KERAS_IMPORT_ERROR = None
except ImportError as exc:
    K = None
    Network = None
    split_grud_inputs = None
    KERAS_IMPORT_ERROR = exc

from mimic4models.fixed_horizon_icu_exit.raw import RawSequenceEncoder


@unittest.skipUnless(Network is not None, 'legacy Keras is unavailable: {}'.format(KERAS_IMPORT_ERROR))
class RawTimestampLstmKerasTests(unittest.TestCase):

    def _model(self, input_dim):
        return Network(dim=4, batch_norm=False, dropout=0.0,
                        rec_dropout=0.0, task='ihm', input_dim=input_dim)

    def test_timestamp_flag_adds_exactly_one_lstm_input_feature(self):
        plain = RawSequenceEncoder()
        delta = RawSequenceEncoder(include_time_deltas=True)
        plain_model = self._model(plain.feature_dim)
        delta_model = self._model(delta.feature_dim)

        self.assertEqual(plain_model.input_shape[-1], plain.feature_dim)
        self.assertEqual(delta_model.input_shape[-1], delta.feature_dim)
        self.assertEqual(delta.feature_dim, plain.feature_dim + 1)

    def test_lstm_masking_ignores_zero_padding_but_keeps_valid_rows(self):
        encoder = RawSequenceEncoder(include_time_deltas=True)
        header = encoder.header()
        mask_i = header.index(next(name for name in header if name.startswith('mask->')))
        gap_i = header.index('TimeSincePreviousInputDays')
        batch = np.zeros((2, 3, encoder.feature_dim), dtype=np.float32)

        # A valid zero-valued measurement is kept by its observation mask.
        batch[0, 0, mask_i] = 1.0
        # A valid sparse row can be represented only by its positive time gap.
        batch[0, 1, gap_i] = 1.0 / 24.0
        batch[1, 0, mask_i] = 1.0

        model = self._model(encoder.feature_dim)
        masking = next(layer for layer in model.layers
                        if layer.__class__.__name__ == 'Masking')
        mask = K.eval(masking.compute_mask(K.constant(batch), None))

        self.assertTrue(np.array_equal(mask, [[True, True, False],
                                              [True, False, False]]))
        self.assertTrue(np.all(np.any(batch[[0, 0, 1], [0, 1, 0]] != 0.0, axis=-1)))
        self.assertTrue(np.all(batch[:, 2] == 0.0))

    def test_grud_still_consumes_raw_hours_not_lstm_time_deltas(self):
        encoder = RawSequenceEncoder(include_timestamps=True)
        header = ['Hours'] + encoder._id_to_channel
        rows = np.asarray([
            ['0.0'] + [''] * len(encoder._id_to_channel),
            ['2.0'] + [''] * len(encoder._id_to_channel),
        ], dtype=object)
        channel_i = next(i + 1 for i, channel in enumerate(encoder._id_to_channel)
                          if not encoder._is_categorical_channel[channel])
        rows[0, channel_i] = '70'
        rows[1, channel_i] = '80'
        encoded, encoded_header = encoder.transform(rows, header=header, end=24.0)
        values, masks, timestamps = split_grud_inputs(
            encoded[None, :, :], encoded_header.split(','), timestep=0.0)

        self.assertEqual(encoded_header.split(',')[-1], 'Hours')
        self.assertEqual(values.shape, masks.shape)
        self.assertTrue(np.allclose(timestamps[0, :, 0], [0.0, 2.0]))
        self.assertEqual(values.shape[-1], masks.shape[-1])


if __name__ == '__main__':
    unittest.main()
