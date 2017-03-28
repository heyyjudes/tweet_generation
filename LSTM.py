
import collections
import tensorflow as tf
import math

_LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h", "z"))

def hard_sigmoid(x, alpha=1):
    hard_sigmoid = tf.maximum(0, tf.minimum(1, ((alpha*x)+1)/2.0))
    return hard_sigmoid

def init_weight(fan_in, fan_out, name=None, stddev=1.0):
    #Initialize with Xavier initialization
    weights = tf.Variable(tf.truncated_normal([fan_in, fan_out], stddev=stddev/math.sqrt(float(fan_in))), name=name)
    return weights

def init_bias(fan_out, name=None, stddev=1.0):
    #Initialize with zero
    bias = tf.Variable(tf.zeros([fan_out]), name=name)
    return bias

def ln(tensor, scope = None, epsilon = 1e-5):
    """ Layer normalizes a 2D tensor along its second axis """
    assert(len(tensor.get_shape()) == 2)
    m, v = tf.nn.moments(tensor, [1], keep_dims=True)
    if not isinstance(scope, str):
        scope = ''
    with tf.variable_scope(scope + 'layer_norm'):
        scale = tf.get_variable('scale',
                                shape=[tensor.get_shape()[1]],
                                initializer=tf.constant_initializer(1))
        shift = tf.get_variable('shift',
                                shape=[tensor.get_shape()[1]],
                                initializer=tf.constant_initializer(0))
    LN_initial = (tensor - m) / tf.sqrt(v + epsilon)

    return LN_initial * scale + shift

class LSTMStateTuple(_LSTMStateTuple):
    """Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.
    Stores two elements: `(c, h)`, in that order.
    Only used when `state_is_tuple=True`.
    """
    __slots__ = ()

    @property
    def dtype(self):
        (c, h, z) = self
        if not c.dtype == h.dtype:
          raise TypeError("Inconsistent internal state: %s vs %s" %
                          (str(c.dtype), str(h.dtype)))
        return c.dtype


class LSTM_TOP(tf.nn.rnn_cell.RNNCell):
    """Hierarchical Multiscale Recurrent Cell https://arxiv.org/pdf/1609.01704.pdf."""

    def __init__(self, num_units, nlp_dim, forget_bias=1.0, input_size=None,
               state_is_tuple=True, input_keep_prob=1.0, output_keep_prob=1.0):
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._nlp_dim = nlp_dim
        self._input_keep_prob = input_keep_prob
        self._output_keep_prob = output_keep_prob

    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units, 1)
            if self._state_is_tuple else 3 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__): 
          # Parameters of gates are concatenated into one multiply for efficiency.
            c_l, h_l, z_l = state

            x = tf.nn.dropout(inputs[0], keep_prob=self._input_keep_prob)
            gates = tf.nn.rnn_cell._linear([x], 4 * self._num_units, True)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.split(1,4,gates)

            i = ln(i, scope = 'i/')
            j = ln(j, scope = 'j/')
            f = ln(f, scope = 'f/')
            o = ln(o, scope = 'o/')

            new_c_l = (c_l * tf.nn.sigmoid(f + self._forget_bias) + tf.nn.sigmoid(i) *
                   tf.nn.tanh(j))
            new_h_l = tf.nn.tanh(ln(new_c_l, scope = 'new_h/')) * tf.nn.sigmoid(o)

            new_h_l = tf.nn.dropout(new_h_l, keep_prob=self._output_keep_prob)

            new_z_l = 1

            new_state = LSTMStateTuple(new_c_l, new_h_l, new_z_l)

        return new_h_l, new_state

class LSTM(tf.nn.rnn_cell.RNNCell):
    """Hierarchical Multiscale Recurrent Cell https://arxiv.org/pdf/1609.01704.pdf."""

    def __init__(self, num_units, nlp_dim, forget_bias=1.0, input_size=None,
               state_is_tuple=True, input_keep_prob=1.0, output_keep_prob=1.0):
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._nlp_dim = nlp_dim
        self._input_keep_prob = input_keep_prob
        self._output_keep_prob = output_keep_prob

    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units, 1)
            if self._state_is_tuple else 3 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__): 
          # Parameters of gates are concatenated into one multiply for efficiency.
            c_l, h_l, z_l = state

            h_lminus1 = tf.nn.dropout(inputs[0], keep_prob=self._input_keep_prob)
            z_lminus1 = inputs[1]
            h_lplus1 = tf.nn.dropout(inputs[2], keep_prob=self._input_keep_prob)
            #s_recurrent  <=>  U * h_l
            #s_topdown <=> h_lplus1 * z_l
            #s_bottom_up <=> h_lminus1 * z_lminus1atT
            with tf.variable_scope("gates"):
                gates = tf.nn.rnn_cell._linear([h_l, z_l * h_lplus1, z_lminus1 * h_lminus1], 4 * self._num_units, True)
            with tf.variable_scope("boundary_detect"):
                z_pre = tf.nn.rnn_cell._linear([h_l, z_l * h_lplus1, z_lminus1 * h_lminus1], 1, True)
            # i = input_gate, j = new_input, f = forget_gate, o = output_gate, z = edge detector
            i, j, f, o = tf.split(1,4,gates)

            i = ln(i, scope = 'i/')
            j = ln(j, scope = 'j/')
            f = ln(f, scope = 'f/')
            o = ln(o, scope = 'o/')
            
            new_c_l = (c_l * tf.nn.sigmoid(f + self._forget_bias) + tf.nn.sigmoid(i) *
                   tf.nn.tanh(j))
            new_h_l = tf.nn.tanh(ln(new_c_l, scope = 'new_h/')) * tf.nn.sigmoid(o)

            new_h_l = tf.nn.dropout(new_h_l, keep_prob=self._output_keep_prob)

            new_z_l = hard_sigmoid(z_pre)

            new_state = LSTMStateTuple(new_c_l, new_h_l, new_z_l)

        return new_h_l, new_state
