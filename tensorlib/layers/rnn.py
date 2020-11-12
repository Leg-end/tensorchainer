from tensorlib import engine
from tensorlib.engine import base_lib as F
from tensorlib import initializers
from tensorlib import regularizers
from tensorlib import activation_ops
from tensorlib.utils import valid_value, to_list
from tensorlib.hooks import RNNSpecHook
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops as fops
from tensorflow.python import ops
from tensorflow.python.ops import state_ops


class RNNCellBase(engine.Layer):

    def __init__(self,
                 units,
                 num_chunks,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activation=None,
                 **kwargs):
        super(RNNCellBase, self).__init__(activation=activation, **kwargs)
        self.units = units
        self.num_chunks = num_chunks
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.state_size = self.units
        self.output_size = self.units
        self.kernel = None
        self.recurrent = None
        self.bias = None

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.num_chunks * self.units),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      name='kernel')
        self.recurrent = self.add_weight(shape=(self.units, self.num_chunks * self.units),
                                         initializer=self.recurrent_initializer,
                                         regularizer=self.recurrent_regularizer,
                                         name='recurrent')
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.num_chunks * self.units,),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        name='bias')

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class StackedRNNCells(engine.LayerList):

    @property
    def state_size(self):
        state_size = []
        for cell in self[::-1] if self.reverse_state_order else self:
            if hasattr(cell.state_size, '__len__'):
                state_size.extend(cell.state_size)
            else:
                state_size.append(cell.state_size)
        return state_size

    @property
    def output_size(self):
        return self[-1].output_size or (
            self[-1].state_size[0] if hasattr(
                self[-1].state_size, '__len__')
            else self[-1].state_size)

    def __init__(self, *cells: RNNCellBase, **kwargs):
        self.reverse_state_order = kwargs.pop('reverse_state_order', False)
        super(StackedRNNCells, self).__init__(*cells, **kwargs)

    def build(self, input_shape):
        for cell in self:
            cell.build(input_shape)
            input_shape = (
                input_shape[0], cell.output_size or (
                    cell.state_size[0] if hasattr(
                        cell.state_size, '__len__')
                    else cell.state_size))

    def forward(self, inputs, states, **kwargs):
        nested_states = []
        for cell in self:
            if hasattr(cell.state_size, '__len__'):
                nested_states.append(states[:len(cell.state_size)])
                states = states[len(cell.state_size):]
            else:
                nested_states.append(states[0])
                states = states[1:]
        new_nested_states = []
        for cell, states in zip(self, nested_states):
            inputs, states = cell.forward(inputs, states, **kwargs)
            new_nested_states.append(states)
        if self.reverse_state_order:
            new_nested_states = new_nested_states[::-1]
        new_states = []
        for cell_states in new_nested_states:
            new_states += cell_states
        return inputs, new_states


class RNNCell(RNNCellBase):
    """
        h' = \tanh(W_{ih} x + b_{ih}  +  W_{hh} h + b_{hh})
    """

    def __init__(self,
                 units,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activation='tanh',
                 **kwargs):
        super(RNNCell, self).__init__(units=units,
                                      num_chunks=1,
                                      use_bias=use_bias,
                                      kernel_initializer=kernel_initializer,
                                      recurrent_initializer=recurrent_initializer,
                                      bias_initializer=bias_initializer,
                                      kernel_regularizer=kernel_regularizer,
                                      recurrent_regularizer=recurrent_regularizer,
                                      bias_regularizer=bias_regularizer,
                                      activation=activation,
                                      **kwargs)

    def forward(self, inputs, states):
        h = F.batch_dot(inputs, self.kernel)
        if self.use_bias:
            h = F.bias_add(h, self.bias)
        outputs = h + F.batch_dot(states[0], self.recurrent)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs, [outputs]


class LSTMCell(RNNCellBase):
    """
        i = \\sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
        f = \\sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
        g = \tanh(W_{ig} x + b_{ig} + W_{hg} h + b_{hg}) \\
        o = \\sigma(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
        c' = f * c + i * g \\
        h' = o * \tanh(c') \\
    """

    def __init__(self,
                 units,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 recurrent_activation='sigmoid',
                 activation='tanh',
                 implementation=1,
                 **kwargs):
        super(LSTMCell, self).__init__(units=units,
                                       num_chunks=4,
                                       use_bias=use_bias,
                                       kernel_initializer=kernel_initializer,
                                       recurrent_initializer=recurrent_initializer,
                                       bias_initializer=bias_initializer,
                                       kernel_regularizer=kernel_regularizer,
                                       recurrent_regularizer=recurrent_regularizer,
                                       bias_regularizer=bias_regularizer,
                                       activation=activation,
                                       **kwargs)
        self.recurrent_activation = activation_ops.get(recurrent_activation)
        self.unit_forget_bias = unit_forget_bias
        self.implementation = valid_value(implementation, [1, 2])
        self.state_size = (self.units, self.units)
        self.kernel_i = None
        self.kernel_f = None
        self.kernel_c = None
        self.kernel_o = None
        self.recurrent_i = None
        self.recurrent_f = None
        self.recurrent_c = None
        self.recurrent_o = None
        self.bias_i = None
        self.bias_f = None
        self.bias_c = None
        self.bias_o = None

    def build(self, input_shape):
        if self.use_bias:
            if self.unit_forget_bias:
                bias_init = self.bias_initializer

                def _bias_initializer(_, *args, **kwargs):
                    return array_ops.concat([
                        bias_init((self.units,), *args, **kwargs),
                        initializers.Ones()((self.units,), *args, **kwargs),
                        bias_init((self.units * 2,), *args, **kwargs)
                    ], axis=0)

                self.bias_initializer = _bias_initializer
        super(LSTMCell, self).build(input_shape=input_shape)
        self.kernel_i = self.kernel[:, :self.units]
        self.kernel_f = self.kernel[:, self.units: self.units * 2]
        self.kernel_c = self.kernel[:, self.units * 2: self.units * 3]
        self.kernel_o = self.kernel[:, self.units * 3:]
        self.recurrent_i = self.recurrent[:, :self.units]
        self.recurrent_f = self.recurrent[:, self.units: self.units * 2]
        self.recurrent_c = self.recurrent[:, self.units * 2: self.units * 3]
        self.recurrent_o = self.recurrent[:, self.units * 3:]
        if self.use_bias:
            self.bias_i = self.bias[:self.units]
            self.bias_f = self.bias[self.units: self.units * 2]
            self.bias_c = self.bias[self.units * 2: self.units * 3]
            self.bias_o = self.bias[self.units * 3:]

    def forward_separately(self, inputs, h, c):
        x_i = F.batch_dot(inputs, self.kernel_i)
        x_f = F.batch_dot(inputs, self.kernel_f)
        x_c = F.batch_dot(inputs, self.kernel_c)
        x_o = F.batch_dot(inputs, self.kernel_o)
        if self.use_bias:
            x_i = F.bias_add(x_i, self.bias_i)
            x_f = F.bias_add(x_f, self.bias_f)
            x_c = F.bias_add(x_c, self.bias_c)
            x_o = F.bias_add(x_o, self.bias_o)
        i = self.recurrent_activation(x_i + F.batch_dot(h, self.recurrent_i))
        f = self.recurrent_activation(x_f + F.batch_dot(h, self.recurrent_f))
        c = f * c + i * self.activation(x_c + F.batch_dot(h, self.recurrent_c))
        o = self.recurrent_activation(x_o + F.batch_dot(h, self.recurrent_o))
        h = o * self.activation(c)
        return h, [h, c]

    def forward_jointly(self, inputs, h, c):
        z = F.batch_dot(inputs, self.kernel)
        z += F.batch_dot(h, self.recurrent)
        if self.use_bias:
            z = F.bias_add(z, self.bias)
        i = self.recurrent_activation(z[:, :self.units])
        f = self.recurrent_activation(z[:, self.units: self.units * 2])
        c = f * c + i * self.activation(z[:, self.units * 2: self.units * 3])
        o = self.recurrent_activation(z[:, self.units * 3:])
        h = o * self.activation(c)
        return h, [h, c]

    def forward(self, inputs, states):
        h, c = states[0], states[1]
        if self.implementation == 1:
            return self.forward_separately(inputs, h, c)
        else:
            return self.forward_jointly(inputs, h, c)


class GRUCell(RNNCellBase):

    def __init__(self,
                 units,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 recurrent_activation='sigmoid',
                 activation='tanh',
                 implementation=1,
                 reset_after=False,
                 **kwargs):
        super(GRUCell, self).__init__(units=units,
                                      num_chunks=3,
                                      use_bias=use_bias,
                                      kernel_initializer=kernel_initializer,
                                      recurrent_initializer=recurrent_initializer,
                                      bias_initializer=bias_initializer,
                                      kernel_regularizer=kernel_regularizer,
                                      recurrent_regularizer=recurrent_regularizer,
                                      bias_regularizer=bias_regularizer,
                                      activation=activation,
                                      **kwargs)
        self.implementation = valid_value(implementation, [1, 2])
        self.reset_after = reset_after
        self.recurrent_activation = activation_ops.get(recurrent_activation)
        self.recurrent_bias = None
        self.kernel_z = None
        self.recurrent_z = None
        self.bias_z = None
        self.recurrent_bias_z = None
        self.kernel_r = None
        self.recurrent_r = None
        self.bias_r = None
        self.recurrent_bias_r = None
        self.kernel_h = None
        self.recurrent_h = None
        self.bias_h = None
        self.recurrent_bias_h = None

    def build(self, input_shape):
        super(GRUCell, self).build(input_shape)
        if self.use_bias and self.reset_after:
            self.recurrent_bias = self.add_weight(
                shape=(self.num_chunks * self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                name='recurrent_bias')
        # update gate
        self.kernel_z = self.kernel[:, :self.units]
        self.recurrent_z = self.recurrent[:, :self.units]
        # reset gate
        self.kernel_r = self.kernel[:, self.units: self.units * 2]
        self.recurrent_r = self.recurrent[:, self.units: self.units * 2]
        # new gate
        self.kernel_h = self.kernel[:, self.units * 2:]
        self.recurrent_h = self.recurrent[:, self.units * 2:]
        if self.use_bias:
            self.bias_z = self.bias[:self.units]
            self.bias_r = self.bias[self.units: self.units * 2]
            self.bias_h = self.bias[self.units * 2:]
            if self.reset_after:
                self.recurrent_bias_z = self.bias[:self.units]
                self.recurrent_bias_r = self.bias[self.units: self.units * 2]
                self.recurrent_bias_h = self.bias[self.units * 2:]

    def forward_separately(self, inputs, state):
        x_z = F.batch_dot(inputs, self.kernel_z)
        x_r = F.batch_dot(inputs, self.kernel_r)
        x_h = F.batch_dot(inputs, self.kernel_h)
        if self.use_bias:
            x_z = F.bias_add(x_z, self.bias_z)
            x_r = F.bias_add(x_r, self.bias_r)
            x_h = F.bias_add(x_h, self.bias_h)
        h_z = F.batch_dot(state, self.recurrent_z)
        h_r = F.batch_dot(state, self.recurrent_r)
        if self.reset_after and self.use_bias:
            h_z = F.bias_add(h_z, self.recurrent_bias_z)
            h_r = F.bias_add(h_r, self.recurrent_bias_r)
        z = self.recurrent_activation(x_z + h_z)
        r = self.recurrent_activation(x_r + h_r)
        if self.reset_after:
            h_h = F.batch_dot(state, self.recurrent_h)
            if self.use_bias:
                h_h = F.bias_add(h_h, self.recurrent_bias_h)
            h_h = r * h_h
        else:
            h_h = F.batch_dot(r * state, self.recurrent_h)
        h = self.activation(x_h + h_h)
        h = z * state + (1 - z) * h
        return h, [h]

    def forward_jointly(self, inputs, state):
        mat_x = F.batch_dot(inputs, self.kernel)
        if self.use_bias:
            mat_x = F.bias_add(mat_x, self.bias)
        x_z = mat_x[:, :self.units]
        x_r = mat_x[:, self.units: self.units * 2]
        x_h = mat_x[:, self.units * 2:]
        if self.reset_after:
            mat_h = F.batch_dot(state, self.recurrent)
            if self.use_bias:
                mat_h = F.bias_add(mat_h, self.recurrent_bias)
        else:
            mat_h = F.batch_dot(state, self.recurrent[:, :self.units * 2])
        h_z = mat_h[:, :self.units]
        h_r = mat_h[:, self.units: self.units * 2]
        z = self.recurrent_activation(x_z + h_z)
        r = self.recurrent_activation(x_r + h_r)
        if self.reset_after:
            h_h = r * mat_h[:, self.units * 2:]
        else:
            h_h = F.batch_dot(r * state, self.recurrent[:, self.units * 2:])
        h = self.activation(x_h + h_h)
        h = z * state + (1 - z) * h
        return h, [h]

    def forward(self, inputs, states):
        if self.implementation == 1:
            return self.forward_separately(inputs, states[0])
        else:
            return self.forward_jointly(inputs, states[0])


class RNNBase(engine.Layer):

    @property
    def state_size(self):
        return self.cell.state_size

    @property
    def units(self):
        return self.cell.units

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.bias_regularizer

    @property
    def states(self):
        if self.stateful:
            return self._states
        else:
            return [None] * len(to_list(self.cell.state_size))

    def __init__(self,
                 cell,
                 return_sequence=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        super(RNNBase, self).__init__(**kwargs)
        if isinstance(cell, (list, tuple)):
            cell = StackedRNNCells(*cell)
        self.cell = cell
        self.return_sequence = return_sequence
        self.return_state = return_state
        self.go_backwards = go_backwards
        self.stateful = stateful
        self.unroll = unroll
        self._states = None
        self.batch_size = None
        self.add_hook(RNNSpecHook())

    def get_initial_state(self, inputs):
        with ops.name_scope('initial_state'):
            initial_state = array_ops.zeros_like(inputs)  # (b, t, i)
            initial_state = math_ops.reduce_sum(initial_state, axis=(1, 2))  # (b,)
            initial_state = array_ops.expand_dims(initial_state, axis=1)  # (b, 1)
            return [array_ops.tile(initial_state, [1, dim])
                    for dim in to_list(self.cell.state_size)]

    def build(self, input_shape):
        self.cell.build(input_shape)
        self.batch_size = input_shape[0]
        if self.stateful:
            if self.batch_size is None:
                raise ValueError("If a RNN is stateful, the last state for"
                                 " each sample at index i in a batch will"
                                 " be used as initial state for the sample"
                                 " of index i in the following batch, that"
                                 " means it needs to know its batch size")
            self._states = [self.add_weight(
                initial_value=array_ops.zeros(self.batch_size, dim),
                shape=(self.batch_size, dim),
                dtype=self.dtype,
                trainable=False,
                name='state_%d' % i) for i, dim in enumerate(
                to_list(self.cell.state_size))]

    def forward(self, inputs, initial_state=None):
        if isinstance(inputs, list):
            initial_state = inputs[1:]
            inputs = inputs[0]
        elif self.stateful:
            initial_state = self.states
        else:
            initial_state = self.get_initial_state(inputs)
        if len(initial_state) != len(self.states):
            raise ValueError("Expect %d states, but received %d" % (
                len(self.states), len(initial_state)))
        time_steps = engine.int_shape(inputs)[1]
        if self.unroll and time_steps in [None, 1]:
            raise ValueError("To unroll a RNN, time dimension in inputs"
                             " must be known, and must larger than 1")
        last_output, outputs, states = F.rnn(step_fn=self.cell.forward,
                                             inputs=inputs,
                                             initial_states=initial_state,
                                             go_backwards=self.go_backwards,
                                             unroll=self.unroll,
                                             input_length=time_steps)
        if self.stateful:
            with fops.control_dependencies(self.states):
                updates = [state_ops.assign(self.states[i], states[i])
                           for i in range(len(self.states))]
                fops.add_to_collection(fops.GraphKeys.UPDATE_OPS, updates)
        if not self.return_sequence:
            outputs = last_output
        if self.return_state:
            return [outputs] + to_list(states)
        else:
            return outputs


class RNN(RNNBase):

    def __init__(self,
                 units,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activation='tanh',
                 return_sequence=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        cell = RNNCell(units=units,
                       use_bias=use_bias,
                       kernel_initializer=kernel_initializer,
                       recurrent_initializer=recurrent_initializer,
                       bias_initializer=bias_initializer,
                       kernel_regularizer=kernel_regularizer,
                       recurrent_regularizer=recurrent_regularizer,
                       bias_regularizer=bias_regularizer,
                       activation=activation)
        super(RNN, self).__init__(cell=cell,
                                  return_sequence=return_sequence,
                                  return_state=return_state,
                                  go_backwards=go_backwards,
                                  stateful=stateful,
                                  unroll=unroll,
                                  **kwargs)


class LSTM(RNNBase):

    @property
    def implementation(self):
        return self.cell.implementation

    @property
    def unit_forget_bias(self):
        return self.cell.unit_forget_bias

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    def __init__(self,
                 units,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 recurrent_activation='sigmoid',
                 activation='tanh',
                 implementation=1,
                 return_sequence=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        cell = LSTMCell(units=units,
                        use_bias=use_bias,
                        kernel_initializer=kernel_initializer,
                        recurrent_initializer=recurrent_initializer,
                        bias_initializer=bias_initializer,
                        unit_forget_bias=unit_forget_bias,
                        kernel_regularizer=kernel_regularizer,
                        recurrent_regularizer=recurrent_regularizer,
                        bias_regularizer=bias_regularizer,
                        recurrent_activation=recurrent_activation,
                        activation=activation,
                        implementation=implementation)
        super(LSTM, self).__init__(cell=cell,
                                   return_sequence=return_sequence,
                                   return_state=return_state,
                                   go_backwards=go_backwards,
                                   stateful=stateful,
                                   unroll=unroll,
                                   **kwargs)


class GRU(RNNBase):

    @property
    def reset_after(self):
        return self.cell.reset_after

    @property
    def implementation(self):
        return self.cell.implementation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    def __init__(self,
                 units,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 recurrent_activation='sigmoid',
                 activation='tanh',
                 implementation=1,
                 reset_after=False,
                 return_sequence=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        cell = GRUCell(units=units,
                       use_bias=use_bias,
                       kernel_initializer=kernel_initializer,
                       recurrent_initializer=recurrent_initializer,
                       bias_initializer=bias_initializer,
                       kernel_regularizer=kernel_regularizer,
                       recurrent_regularizer=recurrent_regularizer,
                       bias_regularizer=bias_regularizer,
                       recurrent_activation=recurrent_activation,
                       activation=activation,
                       implementation=implementation,
                       reset_after=reset_after)
        super(GRU, self).__init__(cell=cell,
                                  return_sequence=return_sequence,
                                  return_state=return_state,
                                  go_backwards=go_backwards,
                                  stateful=stateful,
                                  unroll=unroll,
                                  **kwargs)
