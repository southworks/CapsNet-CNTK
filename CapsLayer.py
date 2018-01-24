import cntk as ct
import numpy as np

from cntk.ops.functions import BlockFunction
from user_matmul import user_matmul

def Length(name='Length', epsilon = 1e-9):
    '''
    Length of the instantiation vector to represent the probability that a capsule’s entity exists.

    Args:
        name (str, optional): The name of the Function instance in the network.
        epsilon (float, optional): A small constant for numerical stability.

    '''

    @BlockFunction('LengthLayer', name)
    def length(input):
        return ct.reshape(
            ct.sqrt(ct.reduce_sum(ct.square(input), axis=1) + epsilon),
            (10, 1)
        )

    return length

def DigitCaps(input, num_capsules, dim_out_vector, routings=3, name='DigitCaps'):
    '''
    Function to create an instance of a digit capsule.

    Args:
        input: Input Tensor
        num_capsules (int): Number of output capsules
        dim_out_vector (int): Number of dimensions of the capsule output vector
        routings (int, optional): The number of routing iterations
        name (str, optional): The name of the Function instance in the network.
    '''
    # Learnable Parameters
    W = ct.Parameter(shape=(1152, 10, 16, 8), init=ct.normal(0.01), name=name + '_Weights')

    # reshape input for broadcasting on all output capsules
    input = ct.reshape(input, (1152, 1, 8, 1), name='reshape_input')

    # Output shape = [#](1152, 10, 16, 1)
    u_hat = user_matmul(W, input, shape=(1152, 10, 16, 1))

    # we don't need gradients on routing
    u_hat_stopped = ct.stop_gradient(u_hat, name='stop_gradient')

    # all the routing logits (Bij) are initialized to zero for each routing.
    Bij = ct.Constant(np.zeros((1152, 10, 1, 1), dtype=np.float32))

    # line 3, for r iterations do
    for r_iter in range(routings):
        # line 4: for all capsule i in layer l: ci ← softmax(bi) => Cij
        # Output shape = [#][1152, 10, 1, 1]
        Cij = ct.softmax(Bij, axis=1)

        # At last iteration, use `u_hat` in order to receive gradients from the following graph
        if r_iter == routings - 1:
            # line 5: for all capsule j in layer (l + 1): sj ← sum(cij * u_hat)
            # Output shape = [#][1152, 10, 16, 1]
            Sj = ct.reduce_sum(ct.element_times(Cij, u_hat, 'weighted_u_hat'), axis=0)

            # line 6: for all capsule j in layer (l + 1): vj ← squash(sj)
            # Output shape = [#][1, 10, 16, 1]
            Vj = Squash(Sj)
        elif r_iter < routings - 1:
            # line 5: for all capsule j in layer (l + 1): sj ← sum(cij * u_hat)
            # Output shape = [#][1152, 10, 16, 1]
            Sj = ct.reduce_sum(ct.element_times(Cij, u_hat_stopped), axis=0)

            # line 6: for all capsule j in layer (l + 1): vj ← squash(sj)
            # Output shape = [#][1, 10, 16, 1]
            Vj = Squash(Sj)

            # line 7: for all capsule i in layer l and capsule j in layer (l + 1): bij ← bij + ^uj|i * vj
            # Output shape = [#][1, 10, 1, 16]
            Vj_Transpose = ct.transpose(ct.reshape(Vj, (1, 10, 16, 1)), (0, 1, 3, 2), name='Vj_Transpose')

            # Output shape = [#][1152, 10, 1, 1]
            UV = user_matmul(Vj_Transpose, u_hat_stopped, shape=(1152, 10, 1, 1), stop_gradients=True)
            Bij += UV

    # Output shape = [#][10, 16, 1]
    Vj = ct.reshape(Vj, (10, 16, 1), name='digit_caps_output')
    return Vj

def PrimaryCaps(num_capsules, dim_out_vector, filter_shape, strides=1, pad=False, name='PrimaryCaps'):
    """
    PrimaryCaps()
    Layer factory function to create an instance of a primary capsule.

    Args:
        num_capsules: Number of primary capsules
        dim_out_vector: Number of dimensions of the capsule output vector
        filter_shape: Convoltional filter shape

    Returns:
        cntk.ops.functions.Function
    """

    convolution = ct.layers.Convolution2D(num_filters=dim_out_vector*num_capsules,
                                          filter_shape=filter_shape,
                                          strides=strides,
                                          pad=pad,
                                          activation=ct.relu,
                                          name=name + '_conv')

    @BlockFunction('PrimaryCaps', name)
    def primaryCaps(input):
        result = convolution(input)
        result = ct.reshape(result, (-1, dim_out_vector), name=name + '_reshape')
        return Squash(result, axis=-1)

    return primaryCaps

def Squash(Sj, axis=-1, name='', epsilon=1e-9):
    '''
    Squash over the the specified axis.

    The squash non-linearity scales vector lengths onto (0, 1) without changing their orientation.
    It is analogous to Sigmoid (used in ANN) which remaps real numbers onto (0, 1).

    Args:
        Sj: Elements to squash.
        axis (int, optional): Axis along which a squash is performed.
        name (str, optional): The name of the Function instance in the network.
        epsilon (float, optional): A small constant for numerical stability.
    Returns:
        :class:`~cntk.ops.functions.Function`
    '''

    @BlockFunction('Squash', name)
    def squash(input):

        # ||Sj||^2
        Sj_squared_norm = ct.reduce_sum(ct.square(input), axis=axis)

        # ||Sj||^2 / (1 + ||Sj||^2) * (Sj / ||Sj||)
        factor = ct.element_divide(
            ct.element_divide(
                Sj_squared_norm,
                ct.plus(1, Sj_squared_norm)
            ),
            ct.sqrt(
                ct.plus(Sj_squared_norm, epsilon)
            )
        )
        return factor * input

    return squash(Sj)