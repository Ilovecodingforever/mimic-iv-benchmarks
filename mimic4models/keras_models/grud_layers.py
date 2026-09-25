from __future__ import absolute_import
from __future__ import print_function

from keras import activations, backend as K
from keras import constraints, initializers, regularizers
from keras.engine import InputSpec
from keras.layers.recurrent import GRU, GRUCell, RNN
from keras.utils.generic_utils import custom_object_scope, has_arg, serialize_keras_object


# Adapted from PeterChe1990/GRU-D nn_utils/grud_layers.py for this repo's
# Keras 2.1.2 / TensorFlow 1.4 stack. Unrelated bidirectional wrappers and
# external masking helpers are intentionally omitted.

_SUPPORTED_IMPUTATION = ['zero', 'forward', 'raw']


def exp_relu(x):
    """GRU-D decay activation: exp(-relu(x))."""
    return K.exp(-K.relu(x))


def get_activation(identifier):
    if identifier is None:
        return None
    with custom_object_scope({'exp_relu': exp_relu}):
        return activations.get(identifier)


class GRUDCell(GRUCell):
    """GRU-D cell with trainable input/hidden decays and mask feeding.

    Inputs are concatenated per timestep as ``[x_t, m_t, s_t]`` by ``GRUD``.
    ``s_t`` is a scalar timestamp; variable-specific elapsed times are inferred
    from ``m_t`` and the previous observed timestamp state.
    """

    def __init__(self, units,
                 x_imputation='zero', input_decay='exp_relu', hidden_decay='exp_relu',
                 use_decay_bias=True, feed_masking=True, masking_decay=None,
                 decay_initializer='zeros', decay_regularizer=None, decay_constraint=None,
                 **kwargs):
        if 'implementation' in kwargs and kwargs['implementation'] != 1:
            raise ValueError('GRU-D supports only implementation=1 in this Keras stack.')
        kwargs['implementation'] = 1
        super(GRUDCell, self).__init__(units, **kwargs)
        if x_imputation not in _SUPPORTED_IMPUTATION:
            raise ValueError('Unsupported x_imputation {}'.format(x_imputation))
        self.x_imputation = x_imputation
        self.input_decay = get_activation(input_decay)
        self.hidden_decay = get_activation(hidden_decay)
        self.use_decay_bias = use_decay_bias
        if (not feed_masking) and masking_decay not in (None, 'None'):
            raise ValueError('Mask needs to be fed into GRU-D to enable mask decay.')
        self.feed_masking = feed_masking
        self.masking_decay = get_activation(masking_decay) if feed_masking else None
        self._masking_dropout_mask = None
        self.decay_initializer = initializers.get(decay_initializer)
        self.decay_regularizer = regularizers.get(decay_regularizer)
        self.decay_constraint = constraints.get(decay_constraint)

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 3:
            raise ValueError('GRU-D must be called on [x, mask, timestamp].')
        if input_shape[0] != input_shape[1]:
            raise ValueError('GRU-D x and mask shapes must match; got {} and {}'.format(
                input_shape[0], input_shape[1]))
        if input_shape[0][0] != input_shape[2][0]:
            raise ValueError('GRU-D x and timestamp batch sizes must match; got {} and {}'.format(
                input_shape[0], input_shape[2]))
        if input_shape[2][-1] != 1:
            raise ValueError('GRU-D timestamp input must have final dimension 1.')

        super(GRUDCell, self).build(input_shape[0])
        input_dim = input_shape[0][-1]
        self.true_input_dim = input_dim
        self.state_size = (self.units, input_dim, input_dim)

        if self.input_decay is not None:
            self.input_decay_kernel = self.add_weight(
                shape=(input_dim,), name='input_decay_kernel', initializer=self.decay_initializer,
                regularizer=self.decay_regularizer, constraint=self.decay_constraint)
            if self.use_decay_bias:
                self.input_decay_bias = self.add_weight(
                    shape=(input_dim,), name='input_decay_bias', initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer, constraint=self.bias_constraint)

        if self.hidden_decay is not None:
            self.hidden_decay_kernel = self.add_weight(
                shape=(input_dim, self.units), name='hidden_decay_kernel', initializer=self.decay_initializer,
                regularizer=self.decay_regularizer, constraint=self.decay_constraint)
            if self.use_decay_bias:
                self.hidden_decay_bias = self.add_weight(
                    shape=(self.units,), name='hidden_decay_bias', initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer, constraint=self.bias_constraint)

        if self.feed_masking:
            self.masking_kernel = self.add_weight(
                shape=(input_dim, self.units * 3), name='masking_kernel', initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer, constraint=self.kernel_constraint)
            self.masking_kernel_z = self.masking_kernel[:, :self.units]
            self.masking_kernel_r = self.masking_kernel[:, self.units:self.units * 2]
            self.masking_kernel_h = self.masking_kernel[:, self.units * 2:]
            if self.masking_decay is not None:
                self.masking_decay_kernel = self.add_weight(
                    shape=(input_dim,), name='masking_decay_kernel', initializer=self.decay_initializer,
                    regularizer=self.decay_regularizer, constraint=self.decay_constraint)
                if self.use_decay_bias:
                    self.masking_decay_bias = self.add_weight(
                        shape=(input_dim,), name='masking_decay_bias', initializer=self.bias_initializer,
                        regularizer=self.bias_regularizer, constraint=self.bias_constraint)
        self.built = True

    def _generate_masking_dropout_mask(self, inputs, training=None):
        if 0 < self.dropout < 1:
            ones = K.ones_like(K.squeeze(inputs[:, 0:1, :], axis=1))

            def dropped_inputs():
                return K.dropout(ones, self.dropout)

            self._masking_dropout_mask = [K.in_train_phase(
                dropped_inputs, ones, training=training) for _ in range(3)]
        else:
            self._masking_dropout_mask = None

    def call(self, inputs, states, training=None):
        input_x = inputs[:, :self.true_input_dim]
        input_m = inputs[:, self.true_input_dim:-1]
        input_s = inputs[:, -1:]
        if K.backend() == 'theano':
            input_s = K.pattern_broadcast(input_s, [False, True])

        h_tm1, x_keep_tm1, s_prev_tm1 = states
        input_1m = K.cast_to_floatx(1.) - input_m
        input_d = input_s - s_prev_tm1

        dp_mask = self._dropout_mask
        rec_dp_mask = self._recurrent_dropout_mask
        m_dp_mask = self._masking_dropout_mask

        if self.input_decay is not None:
            gamma_di = input_d * self.input_decay_kernel
            if self.use_decay_bias:
                gamma_di = K.bias_add(gamma_di, self.input_decay_bias)
            gamma_di = self.input_decay(gamma_di)
        if self.hidden_decay is not None:
            gamma_dh = K.dot(input_d, self.hidden_decay_kernel)
            if self.use_decay_bias:
                gamma_dh = K.bias_add(gamma_dh, self.hidden_decay_bias)
            gamma_dh = self.hidden_decay(gamma_dh)
        if self.feed_masking and self.masking_decay is not None:
            gamma_dm = input_d * self.masking_decay_kernel
            if self.use_decay_bias:
                gamma_dm = K.bias_add(gamma_dm, self.masking_decay_bias)
            gamma_dm = self.masking_decay(gamma_dm)

        if self.input_decay is not None:
            x_keep_t = K.switch(input_m, input_x, x_keep_tm1)
            x_t = K.switch(input_m, input_x, gamma_di * x_keep_t)
        elif self.x_imputation == 'forward':
            x_t = K.switch(input_m, input_x, x_keep_tm1)
            x_keep_t = x_t
        elif self.x_imputation == 'zero':
            x_t = K.switch(input_m, input_x, K.zeros_like(input_x))
            x_keep_t = x_t
        elif self.x_imputation == 'raw':
            x_t = input_x
            x_keep_t = x_t
        else:
            raise ValueError('Invalid x_imputation {}'.format(self.x_imputation))

        h_tm1d = gamma_dh * h_tm1 if self.hidden_decay is not None else h_tm1

        if self.feed_masking:
            m_t = input_1m
            if self.masking_decay is not None:
                m_t = gamma_dm * m_t

        if 0. < self.dropout < 1.:
            x_z, x_r, x_h = x_t * dp_mask[0], x_t * dp_mask[1], x_t * dp_mask[2]
            if self.feed_masking:
                m_z, m_r, m_h = m_t * m_dp_mask[0], m_t * m_dp_mask[1], m_t * m_dp_mask[2]
        else:
            x_z, x_r, x_h = x_t, x_t, x_t
            if self.feed_masking:
                m_z, m_r, m_h = m_t, m_t, m_t

        if 0. < self.recurrent_dropout < 1.:
            h_tm1_z, h_tm1_r = h_tm1d * rec_dp_mask[0], h_tm1d * rec_dp_mask[1]
        else:
            h_tm1_z, h_tm1_r = h_tm1d, h_tm1d

        z_t = K.dot(x_z, self.kernel_z) + K.dot(h_tm1_z, self.recurrent_kernel_z)
        r_t = K.dot(x_r, self.kernel_r) + K.dot(h_tm1_r, self.recurrent_kernel_r)
        hh_t = K.dot(x_h, self.kernel_h)
        if self.feed_masking:
            z_t += K.dot(m_z, self.masking_kernel_z)
            r_t += K.dot(m_r, self.masking_kernel_r)
            hh_t += K.dot(m_h, self.masking_kernel_h)
        if self.use_bias:
            z_t = K.bias_add(z_t, self.bias_z)
            r_t = K.bias_add(r_t, self.bias_r)
            hh_t = K.bias_add(hh_t, self.bias_h)
        z_t = self.recurrent_activation(z_t)
        r_t = self.recurrent_activation(r_t)

        h_tm1_h = r_t * h_tm1d
        if 0. < self.recurrent_dropout < 1.:
            h_tm1_h = h_tm1_h * rec_dp_mask[2]
        hh_t = self.activation(hh_t + K.dot(h_tm1_h, self.recurrent_kernel_h))
        h_t = z_t * h_tm1 + (1 - z_t) * hh_t

        if 0. < self.dropout + self.recurrent_dropout and training is None:
            h_t._uses_learning_phase = True

        s_prev_t = K.switch(input_m, K.tile(input_s, [1, self.state_size[-1]]), s_prev_tm1)
        return h_t, [h_t, x_keep_t, s_prev_t]

    def get_config(self):
        config = {
            'x_imputation': self.x_imputation,
            'input_decay': serialize_keras_object(self.input_decay),
            'hidden_decay': serialize_keras_object(self.hidden_decay),
            'use_decay_bias': self.use_decay_bias,
            'feed_masking': self.feed_masking,
            'masking_decay': serialize_keras_object(self.masking_decay),
            'decay_initializer': initializers.serialize(self.decay_initializer),
            'decay_regularizer': regularizers.serialize(self.decay_regularizer),
            'decay_constraint': constraints.serialize(self.decay_constraint),
        }
        base_config = super(GRUDCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GRUD(GRU):
    """GRU-D recurrent layer accepting ``[values, masks, timestamps]``."""

    def __init__(self, units,
                 activation='sigmoid', recurrent_activation='hard_sigmoid', use_bias=True,
                 kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                 bias_initializer='zeros', kernel_regularizer=None,
                 recurrent_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None, kernel_constraint=None,
                 recurrent_constraint=None, bias_constraint=None,
                 dropout=0., recurrent_dropout=0., implementation=1,
                 return_sequences=False, return_state=False, go_backwards=False,
                 stateful=False, unroll=False,
                 x_imputation='zero', input_decay='exp_relu', hidden_decay='exp_relu',
                 use_decay_bias=True, feed_masking=True, masking_decay=None,
                 decay_initializer='zeros', decay_regularizer=None, decay_constraint=None,
                 **kwargs):
        if unroll:
            raise ValueError('GRU-D does not support unroll.')
        cell = GRUDCell(
            units, activation=activation, recurrent_activation=recurrent_activation,
            use_bias=use_bias, kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer, kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint, bias_constraint=bias_constraint,
            dropout=dropout, recurrent_dropout=recurrent_dropout, implementation=implementation,
            x_imputation=x_imputation, input_decay=input_decay, hidden_decay=hidden_decay,
            use_decay_bias=use_decay_bias, feed_masking=feed_masking, masking_decay=masking_decay,
            decay_initializer=decay_initializer, decay_regularizer=decay_regularizer,
            decay_constraint=decay_constraint)
        super(GRU, self).__init__(cell, return_sequences=return_sequences,
                                  return_state=return_state, go_backwards=go_backwards,
                                  stateful=stateful, unroll=unroll, **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=3), InputSpec(ndim=3)]

    def compute_output_shape(self, input_shape):
        output_shape = super(GRUD, self).compute_output_shape(input_shape)
        return output_shape[:-2] if self.return_state else output_shape

    def compute_mask(self, inputs, mask):
        output_mask = super(GRUD, self).compute_mask(inputs, mask)
        return output_mask[:-2] if self.return_state else output_mask

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) < 3:
            raise ValueError('input_shape of GRU-D should be [x, mask, timestamp].')
        input_shape = input_shape[:3]
        batch_size = input_shape[0][0] if self.stateful else None
        self.input_spec[0] = InputSpec(shape=(batch_size, None, input_shape[0][-1]))
        self.input_spec[1] = InputSpec(shape=(batch_size, None, input_shape[1][-1]))
        self.input_spec[2] = InputSpec(shape=(batch_size, None, 1))
        step_input_shape = [(i_s[0],) + i_s[2:] for i_s in input_shape]
        self.cell.build(step_input_shape)
        state_size = list(self.cell.state_size)
        if self.state_spec is not None:
            if [spec.shape[-1] for spec in self.state_spec] != state_size:
                raise ValueError('Initial state is not compatible with GRU-D state_size {}'.format(
                    self.cell.state_size))
        else:
            self.state_spec = [InputSpec(shape=(None, dim)) for dim in state_size]
        if self.stateful:
            self.reset_states()
        self.built = True

    def get_initial_state(self, inputs):
        initial_state = K.zeros_like(inputs[0])
        initial_state = K.sum(initial_state, axis=(1, 2))
        initial_state = K.expand_dims(initial_state)
        ret = [K.tile(initial_state, [1, dim]) for dim in self.cell.state_size[:-1]]
        if self.go_backwards:
            return ret + [K.tile(K.max(inputs[2], axis=1), [1, self.cell.state_size[-1]])]
        return ret + [K.tile(inputs[2][:, 0, :], [1, self.cell.state_size[-1]])]

    def __call__(self, inputs, initial_state=None, **kwargs):
        inputs, initial_state = _standardize_grud_args(inputs, initial_state)
        if initial_state is None:
            return super(RNN, self).__call__(inputs, **kwargs)
        kwargs['initial_state'] = initial_state
        additional_inputs = list(initial_state)
        self.state_spec = [InputSpec(shape=K.int_shape(state)) for state in initial_state]
        original_input_spec = self.input_spec
        self.input_spec = self.input_spec + self.state_spec
        try:
            return super(RNN, self).__call__(inputs + additional_inputs, **kwargs)
        finally:
            self.input_spec = original_input_spec

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.cell._dropout_mask = None
        self.cell._recurrent_dropout_mask = None
        self.cell._masking_dropout_mask = None
        inputs = inputs[:3]
        self.cell._generate_dropout_mask(inputs[0], training=training)
        self.cell._generate_recurrent_dropout_mask(inputs[0], training=training)
        self.cell._generate_masking_dropout_mask(inputs[1], training=training)

        if initial_state is not None:
            pass
        elif self.stateful:
            initial_state = self.states
        else:
            initial_state = self.get_initial_state(inputs)
        if len(initial_state) != len(self.states):
            raise ValueError('Layer has {} states but was passed {} initial states.'.format(
                len(self.states), len(initial_state)))
        timesteps = K.int_shape(inputs[0])[1]
        kwargs = {}
        if has_arg(self.cell.call, 'training'):
            kwargs['training'] = training

        if isinstance(mask, list):
            mask = mask[0]
        if mask is not None:
            mask = K.expand_dims(K.cast(mask, K.floatx()), axis=-1)

            def step(step_inputs, states):
                step_mask = step_inputs[:, -1:]
                step_inputs = step_inputs[:, :-1]
                output, new_states = self.cell.call(step_inputs, states, **kwargs)
                output = step_mask * output + (1.0 - step_mask) * states[0]
                new_states = [step_mask * new_state + (1.0 - step_mask) * old_state
                              for new_state, old_state in zip(new_states, states)]
                return output, new_states

            concatenated_inputs = K.concatenate(inputs + [mask], axis=-1)
            rnn_mask = None
        else:
            def step(step_inputs, states):
                return self.cell.call(step_inputs, states, **kwargs)

            concatenated_inputs = K.concatenate(inputs, axis=-1)
            rnn_mask = None
        last_output, outputs, states = K.rnn(
            step, concatenated_inputs, initial_state, go_backwards=self.go_backwards,
            mask=rnn_mask, unroll=self.unroll, input_length=timesteps)
        if self.stateful:
            updates = []
            for i, state in enumerate(states):
                updates.append((self.states[i], state))
            self.add_update(updates, inputs)
        output = outputs if self.return_sequences else last_output
        if getattr(last_output, '_uses_learning_phase', False):
            output._uses_learning_phase = True
            for state in states:
                state._uses_learning_phase = True
        if self.return_state:
            return [output] + list(states)[:-2]
        return output

    @property
    def units(self):
        return self.cell.units

    def get_config(self):
        config = {
            'x_imputation': self.cell.x_imputation,
            'input_decay': serialize_keras_object(self.cell.input_decay),
            'hidden_decay': serialize_keras_object(self.cell.hidden_decay),
            'use_decay_bias': self.cell.use_decay_bias,
            'feed_masking': self.cell.feed_masking,
            'masking_decay': serialize_keras_object(self.cell.masking_decay),
            'decay_initializer': initializers.serialize(self.cell.decay_initializer),
            'decay_regularizer': regularizers.serialize(self.cell.decay_regularizer),
            'decay_constraint': constraints.serialize(self.cell.decay_constraint),
        }
        base_config = super(GRUD, self).get_config()
        base_config.pop('reset_after', None)
        return dict(list(base_config.items()) + list(config.items()))


def _standardize_grud_args(inputs, initial_state):
    if not isinstance(inputs, list) or len(inputs) < 3:
        raise ValueError('inputs to GRU-D should be [x, mask, timestamp].')
    if initial_state is None:
        if len(inputs) > 3:
            initial_state = inputs[3:]
        inputs = inputs[:3]
    if initial_state is None or isinstance(initial_state, list):
        return inputs, initial_state
    if isinstance(initial_state, tuple):
        return inputs, list(initial_state)
    return inputs, [initial_state]


def _get_grud_layers_scope_dict():
    return {'GRUDCell': GRUDCell, 'GRUD': GRUD, 'exp_relu': exp_relu}
