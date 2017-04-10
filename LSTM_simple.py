import collections
import tensorflow as tf
import math

_LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h"))

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
        (c, h) = self
        if not c.dtype == h.dtype:
          raise TypeError("Inconsistent internal state: %s vs %s" %
                          (str(c.dtype), str(h.dtype)))
        return c.dtype


class CustomCell(tf.nn.rnn_cell.RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

    def __init__(self, num_units, nlp_dim, forget_bias=1.0, input_size=None,
               state_is_tuple=True, input_keep_prob=1.0, output_keep_prob=1.0):
        """Initialize the basic LSTM cell.
        Args:
          num_units: int, The number of units in the LSTM cell.
          forget_bias: float, The bias added to forget gates (see above).
          input_size: Deprecated and unused.
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  The latter behavior will soon be deprecated.
          activation: Activation function of the inner states.
          reuse: (optional) Python boolean describing whether to reuse variables
            in an existing scope.  If not `True`, and the existing scope already has
            the given variables, an error is raised.
        """
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._nlp_dim = nlp_dim
        self._input_keep_prob = input_keep_prob
        self._output_keep_prob = output_keep_prob

    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units)
            if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__): 
            with tf.name_scope('LSTM_V'):
              # Parameters of gates are concatenated into one multiply for efficiency.
                c, h = state

                x = tf.nn.dropout(inputs, keep_prob=self._input_keep_prob)
                gates = tf.nn.rnn_cell._linear([x, h], 4 * self._num_units, True)

                # i = input_gate, j = new_input, f = forget_gate, o = output_gate
                i, j, f, o = tf.split(1,4,gates)
       
                i = ln(i, scope = 'i/')
                j = ln(j, scope = 'j/')
                f = ln(f, scope = 'f/')
                o = ln(o, scope = 'o/')

                new_c = (c * tf.nn.sigmoid(f + self._forget_bias) + tf.nn.sigmoid(i) *
                       tf.nn.tanh(j))
                new_h = tf.nn.tanh(ln(new_c, scope = 'new_h/')) * tf.nn.sigmoid(o)

                new_h = tf.nn.dropout(new_h, keep_prob=self._output_keep_prob)

                new_state = LSTMStateTuple(new_c, new_h)

        return new_h, new_state