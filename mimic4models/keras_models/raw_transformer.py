from __future__ import absolute_import
from __future__ import print_function

import math

import numpy as np
from keras import backend as K
from keras.layers import Dense, Dropout, Input, Layer
from keras.models import Model
from mimic4models.keras_models.raw_transformer_math import sinusoidal_position_encoding


USES_RAW_TIMESTAMPS = True
OPTIONAL_RAW_TIMESTAMPS = True
IS_RAW_TRANSFORMER = True
USES_RAW_GRUD = False

D_MODEL = 16
NUM_HEADS = 4
HEAD_DIM = 4
NUM_LAYERS = 2
D_FF = 64


def _as_header_list(header):
    if isinstance(header, str):
        return header.split(',')
    return list(header)


def _clinical_indices(header, positional_encoding):
    names = _as_header_list(header)
    if positional_encoding == 'timestamp':
        if 'Hours' not in names:
            raise ValueError('timestamp positional encoding requires an Hours feature')
        return [i for i, name in enumerate(names) if name != 'Hours']
    return list(range(len(names)))


def _mask_indices(header, clinical_indices):
    names = _as_header_list(header)
    mask_indices = [i for i, name in enumerate(names) if name.startswith('mask->')]
    position = dict((source_i, clinical_i)
                   for clinical_i, source_i in enumerate(clinical_indices))
    try:
        return [position[i] for i in mask_indices]
    except KeyError:
        raise ValueError('raw Transformer requires observation-mask features')


def prepare_input(X, header, timestep=0.0, **kwargs):
    """Split raw clinical features and preserved Hours for timestamp PE."""
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError('raw Transformer prepare_input expects a 3D tensor, got {}'.format(X.shape))
    names = _as_header_list(header)
    if 'Hours' not in names:
        return X.astype('float32')
    timestamp_i = names.index('Hours')
    clinical = np.delete(X, timestamp_i, axis=2).astype('float32')
    timestamps = X[:, :, timestamp_i:timestamp_i + 1].astype('float32')
    return [clinical, timestamps]


class ObservationMask(Layer):
    """Create one valid-timestep bit from the raw per-variable masks."""

    def __init__(self, mask_indices, **kwargs):
        self.mask_indices = list(mask_indices)
        super(ObservationMask, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        pieces = [inputs[:, :, i:i + 1] for i in self.mask_indices]
        masks = K.concatenate(pieces, axis=-1)
        return K.cast(K.any(K.not_equal(masks, 0.0), axis=-1), K.floatx())

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1])


class SinusoidalPositionalEncoding(Layer):
    """Add fixed sinusoidal encodings from Hours or integer token positions."""

    def __init__(self, d_model=D_MODEL, mode='timestamp', **kwargs):
        if mode not in ('timestamp', 'index'):
            raise ValueError('positional encoding mode must be timestamp or index')
        self.d_model = int(d_model)
        self.mode = mode
        super(SinusoidalPositionalEncoding, self).__init__(**kwargs)

    def _index_positions(self, inputs):
        n = K.shape(inputs)[1]
        positions = K.cast(K.arange(0, n), K.floatx())
        positions = K.reshape(positions, (1, -1, 1))
        return K.tile(positions, (K.shape(inputs)[0], 1, 1))

    def call(self, inputs, **kwargs):
        if self.mode == 'timestamp':
            values, positions = inputs
            positions = K.cast(positions, K.floatx())
        else:
            values = inputs
            positions = self._index_positions(values)
        inverse = K.exp(K.arange(0, self.d_model, 2, dtype=K.floatx()) *
                        (-math.log(10000.0) / float(self.d_model)))
        angles = positions * inverse
        interleaved = K.stack([K.sin(angles), K.cos(angles)], axis=-1)
        encoding = K.reshape(interleaved, K.shape(values))
        return values + encoding

    def compute_output_shape(self, input_shape):
        return input_shape[0] if isinstance(input_shape, list) else input_shape


class TransformerEncoderBlock(Layer):
    def __init__(self, d_model=D_MODEL, num_heads=NUM_HEADS, d_ff=D_FF,
                 dropout=0.0, **kwargs):
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.d_ff = int(d_ff)
        self.dropout_rate = float(dropout)
        if self.d_model % self.num_heads != 0:
            raise ValueError('d_model must be divisible by num_heads')
        self.head_dim = self.d_model // self.num_heads
        self.attention_dropout = Dropout(self.dropout_rate)
        self.output_dropout = Dropout(self.dropout_rate)
        self.ffn_dropout = Dropout(self.dropout_rate)
        super(TransformerEncoderBlock, self).__init__(**kwargs)

    def build(self, input_shape):
        width = int(input_shape[0][-1])
        if width != self.d_model:
            raise ValueError('block input width {} != d_model {}'.format(width, self.d_model))
        initializer = 'glorot_uniform'
        self.query_kernel = self.add_weight((self.d_model, self.d_model), name='query_kernel',
                                            initializer=initializer, trainable=True)
        self.key_kernel = self.add_weight((self.d_model, self.d_model), name='key_kernel',
                                           initializer=initializer, trainable=True)
        self.value_kernel = self.add_weight((self.d_model, self.d_model), name='value_kernel',
                                             initializer=initializer, trainable=True)
        self.output_kernel = self.add_weight((self.d_model, self.d_model), name='output_kernel',
                                             initializer=initializer, trainable=True)
        self.gamma1 = self.add_weight((self.d_model,), name='gamma1',
                                      initializer='ones', trainable=True)
        self.beta1 = self.add_weight((self.d_model,), name='beta1',
                                     initializer='zeros', trainable=True)
        self.gamma2 = self.add_weight((self.d_model,), name='gamma2',
                                      initializer='ones', trainable=True)
        self.beta2 = self.add_weight((self.d_model,), name='beta2',
                                     initializer='zeros', trainable=True)
        self.ffn_kernel_1 = self.add_weight((self.d_model, self.d_ff), name='ffn_kernel_1',
                                            initializer='glorot_uniform', trainable=True)
        self.ffn_kernel_2 = self.add_weight((self.d_ff, self.d_model), name='ffn_kernel_2',
                                            initializer='glorot_uniform', trainable=True)
        super(TransformerEncoderBlock, self).build(input_shape)

    def _split_heads(self, values):
        values = K.reshape(values, (K.shape(values)[0], K.shape(values)[1],
                                    self.num_heads, self.head_dim))
        return K.permute_dimensions(values, (0, 2, 1, 3))

    def _normalize(self, values, gamma, beta):
        mean = K.mean(values, axis=-1, keepdims=True)
        variance = K.mean(K.square(values - mean), axis=-1, keepdims=True)
        return gamma * (values - mean) / K.sqrt(variance + 1e-6) + beta

    def call(self, inputs, **kwargs):
        values, valid = inputs
        query = self._split_heads(K.dot(values, self.query_kernel))
        key = self._split_heads(K.dot(values, self.key_kernel))
        value = self._split_heads(K.dot(values, self.value_kernel))
        key_t = K.permute_dimensions(key, (0, 1, 3, 2))
        scores = K.batch_dot(query, key_t, axes=[3, 2])
        scores *= 1.0 / math.sqrt(float(self.head_dim))
        key_mask = K.expand_dims(K.expand_dims(valid, axis=1), axis=1)
        scores += (1.0 - key_mask) * -1e9
        weights = self.attention_dropout(K.softmax(scores))
        attended = K.batch_dot(weights, value, axes=[3, 2])
        attended = K.permute_dimensions(attended, (0, 2, 1, 3))
        attended = K.reshape(attended, K.shape(values))
        attended = self.output_dropout(K.dot(attended, self.output_kernel))
        attended *= K.expand_dims(valid, axis=-1)

        values = self._normalize(values + attended, self.gamma1, self.beta1)
        values *= K.expand_dims(valid, axis=-1)
        feed_forward = K.relu(K.dot(values, self.ffn_kernel_1))
        feed_forward = K.dot(feed_forward, self.ffn_kernel_2)
        feed_forward = self.ffn_dropout(feed_forward)
        values = self._normalize(values + feed_forward, self.gamma2, self.beta2)
        return values * K.expand_dims(valid, axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class MaskedMeanPooling(Layer):
    def call(self, inputs, **kwargs):
        values, valid = inputs
        valid = K.expand_dims(valid, axis=-1)
        total = K.sum(values * valid, axis=1)
        count = K.maximum(K.sum(valid, axis=1), K.constant(1.0))
        return total / count

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][-1])


class Network(Model):
    def __init__(self, dim=16, batch_norm=False, dropout=0.3, rec_dropout=0.0,
                 task='ihm', target_repl=False, deep_supervision=False,
                 num_classes=1, depth=2, input_dim=76, header=None,
                 positional_encoding='timestamp', **kwargs):
        del dim, batch_norm, rec_dropout, depth, input_dim
        if positional_encoding not in ('timestamp', 'index'):
            raise ValueError('positional_encoding must be timestamp or index')
        if target_repl or deep_supervision:
            raise ValueError('raw Transformer supports only one pooled prediction')
        if num_classes != 1:
            raise ValueError('raw Transformer supports one sigmoid output')
        if task not in ('ihm', 'decomp', 'ph'):
            raise ValueError('Wrong value for task')
        if header is None:
            raise ValueError('raw Transformer requires the raw feature header')

        names = _as_header_list(header)
        clinical_indices = _clinical_indices(names, positional_encoding)
        mask_indices = _mask_indices(names, clinical_indices)
        clinical_dim = len(clinical_indices)
        self.positional_encoding = positional_encoding
        self.d_model = D_MODEL
        self.num_heads = NUM_HEADS
        self.num_layers = NUM_LAYERS
        self.d_ff = D_FF
        self.dropout = float(dropout)

        clinical_input = Input(shape=(None, clinical_dim), name='X')
        valid = ObservationMask(mask_indices, name='observation_mask')(clinical_input)
        values = Dense(D_MODEL, use_bias=True, name='clinical_projection')(clinical_input)
        if positional_encoding == 'timestamp':
            timestamp_input = Input(shape=(None, 1), name='Hours')
            values = SinusoidalPositionalEncoding(
                D_MODEL, 'timestamp', name='timestamp_pe')([values, timestamp_input])
            inputs = [clinical_input, timestamp_input]
        else:
            values = SinusoidalPositionalEncoding(
                D_MODEL, 'index', name='index_pe')(values)
            inputs = [clinical_input]

        for layer_i in range(NUM_LAYERS):
            values = TransformerEncoderBlock(
                D_MODEL, NUM_HEADS, D_FF, dropout=self.dropout,
                name='transformer_block_{}'.format(layer_i + 1))([values, valid])
        pooled = MaskedMeanPooling(name='masked_mean')([values, valid])
        output = Dense(1, activation='sigmoid', name='prediction')(pooled)
        super(Network, self).__init__(inputs=inputs, outputs=[output])

    def say_name(self):
        return 'raw_transformer.pe_{}.n{}h{}l{}ff{}.d{}'.format(
            self.positional_encoding, self.d_model, self.num_heads,
            self.num_layers, self.d_ff, self.dropout)
