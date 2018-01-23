import numpy as np
import cntk as ct
from cntk.ops.functions import UserFunction

class matmul(UserFunction):
    '''
    Custom Matrix Multiplication
    see: https://github.com/Microsoft/CNTK/issues/2817

    CPU Only.
    Does not work till all dimensions are well known.
    '''
    def __init__ (self, left, right, shape=None, stop_gradients=False, name='user_matmul'):
        super(matmul, self).__init__([left, right], name=name)
        self.shape = shape
        self.stop_gradients = stop_gradients

    def forward(self, args, outputs=None, keep_for_backward=None, device=None, as_numpy=True):
        z = np.matmul(args[0], args[1])
        return args + (self.shape, self.stop_gradients), z

    def backward(self, state, root_gradients, variables):
        if self.stop_gradients and self.shape:
            return
        for idx in range(len(self.inputs)):
            var = self.inputs[idx]
            if var in variables:
                gradients = None
                if "Parameter" in str(var):
                    gradients = np.sum(np.matmul(root_gradients, np.transpose(state[1], (0, 1, 2, 4, 3))), axis=0)/state[1].shape[0]
                else:
                    gradients = np.sum(np.matmul(np.transpose(state[0], (0, 1, 3, 2)), root_gradients), axis=2, keepdims=True)
                variables[self.inputs[idx]] = gradients

    def infer_outputs(self):
        if not self.shape:
            self.shape = self.inputs[0].shape[:-1] + (self.inputs[1].shape[-1],)
        return [ct.output_variable(self.shape, np.float32, self.inputs[1].dynamic_axes, name=self.name + '_output_shape')]

    @staticmethod
    def deserialize(inputs, name, state):
        shape = ()
        for s, v in state.items():
            if "shape" in s:
                shape += (v,)
        return matmul(inputs[0], inputs[1], shape=shape, stop_gradients=state['stop_gradients'], name=name)

    def serialize(self):
        s = dict(('s'+str(i), s) for i, s in enumerate(self.shape))
        s.update({ 'stop_gradients' : self.stop_gradients })
        return s

def user_matmul(left, right, shape=None, stop_gradients=False, name=''):
    return ct.as_composite(matmul(left, right, shape, stop_gradients), name=name)