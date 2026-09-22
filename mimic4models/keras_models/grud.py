from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.layers.wrappers import TimeDistributed
from mimic4models.keras_utils import LastTimestep
from mimic4models.keras_models.grud_layers import GRUD


def _as_header_list(header):
    if isinstance(header, str):
        return header.split(',')
    return list(header)


def _source_channel(feature_name):
    if '->' in feature_name:
        return feature_name.split('->', 1)[0]
    return feature_name


def value_mask_mapping(header):
    """Return value-column indices and matching original-mask indices.

    The discretizer emits encoded value columns first, followed by one
    ``mask->channel`` column per original clinical channel. Categorical one-hot
    columns inherit the mask of their parent channel. PeterChe GRU-D treats each
    input dimension numerically, so this first version keeps the benchmark's
    existing one-hot categorical encoding instead of adding embeddings.
    """
    names = _as_header_list(header)
    value_indices = [i for i, name in enumerate(names) if not name.startswith('mask->')]
    mask_by_channel = dict((name[len('mask->'):], i)
                           for i, name in enumerate(names) if name.startswith('mask->'))
    if not mask_by_channel:
        raise ValueError('GRU-D requires discretizer outputs with store_masks=True.')
    mask_indices = []
    sources = []
    for value_i in value_indices:
        source = _source_channel(names[value_i])
        if source not in mask_by_channel:
            raise ValueError('No mask column found for value column {}'.format(names[value_i]))
        mask_indices.append(mask_by_channel[source])
        sources.append(source)
    return value_indices, mask_indices, sources


def timestamps_for_batch(n_examples, n_timesteps, timestep):
    stamps = np.arange(n_timesteps, dtype='float32') * float(timestep)
    stamps = stamps.reshape((1, n_timesteps, 1))
    return np.tile(stamps, (int(n_examples), 1, 1)).astype('float32')


def split_grud_inputs(X, header, timestep):
    """Split normalized discretizer output into GRU-D values, masks, timestamps."""
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError('GRU-D prepare_input expects a 3D tensor, got shape {}'.format(X.shape))
    value_indices, mask_indices, _ = value_mask_mapping(header)
    values = X[:, :, value_indices].astype('float32')
    masks = X[:, :, mask_indices].astype('float32')
    if values.shape != masks.shape:
        raise AssertionError('GRU-D values and masks must have identical shape; got {} and {}'.format(
            values.shape, masks.shape))
    if not np.all((masks == 0.0) | (masks == 1.0)):
        raise AssertionError('GRU-D expanded masks must remain binary after normalization.')
    timestamps = timestamps_for_batch(X.shape[0], X.shape[1], timestep)
    return values, masks, timestamps


def prepare_input(X, header, timestep=1.0, **kwargs):
    """Model-specific runner hook returning ``[X_values, M_values, S]``.

    Existing preprocessing and normalization run before this hook. Missing
    positions may contain previous-value imputations in ``X_values``, but GRU-D
    receives the expanded observation mask and ignores those values whenever the
    mask is zero inside the recurrent cell.
    """
    values, masks, timestamps = split_grud_inputs(X, header, timestep)
    return [values, masks, timestamps]


def elapsed_since_observed(timestamps, masks):
    """Numpy mirror of GRU-D timestamp-state delta logic for sanity checks."""
    timestamps = np.asarray(timestamps, dtype='float32')
    masks = np.asarray(masks, dtype='float32')
    if timestamps.ndim == 2:
        timestamps = timestamps[:, :, None]
    if timestamps.shape[:2] != masks.shape[:2] or timestamps.shape[-1] != 1:
        raise ValueError('timestamps must have shape (N, T, 1) matching mask batch/time axes')
    prev = np.tile(timestamps[:, 0, :], (1, masks.shape[-1]))
    out = np.zeros_like(masks, dtype='float32')
    for t in range(masks.shape[1]):
        now = np.tile(timestamps[:, t, :], (1, masks.shape[-1]))
        out[:, t, :] = now - prev
        prev = np.where(masks[:, t, :] == 1.0, now, prev)
    return out


class Network(Model):

    def __init__(self, dim, batch_norm, dropout, rec_dropout, task,
                 target_repl=False, deep_supervision=False, num_classes=1,
                 depth=1, input_dim=76, header=None, **kwargs):

        print('==> not used params in network class:', kwargs.keys())
        if depth != 1:
            raise ValueError('Current GRU-D implementation supports only depth=1.')
        if deep_supervision:
            raise ValueError('GRU-D Network does not implement deep supervision.')
        if header is None:
            raise ValueError('GRU-D Network requires the discretizer header.')

        value_indices, _, _ = value_mask_mapping(header)
        grud_input_dim = len(value_indices)

        self.dim = dim
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.rec_dropout = rec_dropout
        self.depth = depth
        self.input_dim = grud_input_dim

        if task in ['decomp', 'ihm', 'ph']:
            final_activation = 'sigmoid'
        elif task in ['los']:
            final_activation = 'relu' if num_classes == 1 else 'softmax'
        else:
            raise ValueError('Wrong value for task')

        X = Input(shape=(None, grud_input_dim), name='X')
        M = Input(shape=(None, grud_input_dim), name='M')
        S = Input(shape=(None, 1), name='S')
        inputs = [X, M, S]

        return_sequences = bool(target_repl)
        L = GRUD(units=dim,
                 return_sequences=return_sequences,
                 dropout=dropout,
                 recurrent_dropout=rec_dropout,
                 x_imputation='zero',
                 input_decay='exp_relu',
                 hidden_decay='exp_relu',
                 feed_masking=True)(inputs)

        if dropout > 0:
            L = Dropout(dropout)(L)

        if target_repl:
            y = TimeDistributed(Dense(num_classes, activation=final_activation), name='seq')(L)
            y_last = LastTimestep(name='single')(y)
            outputs = [y_last, y]
        else:
            outputs = [Dense(num_classes, activation=final_activation)(L)]

        super(Network, self).__init__(inputs=inputs, outputs=outputs)

    def say_name(self):
        return '{}.n{}{}{}{}.dep{}'.format('k_grud',
                                           self.dim,
                                           '.bn' if self.batch_norm else '',
                                           '.d{}'.format(self.dropout) if self.dropout > 0 else '',
                                           '.rd{}'.format(self.rec_dropout) if self.rec_dropout > 0 else '',
                                           self.depth)
