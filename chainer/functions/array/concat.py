import numpy
import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check

import cupy
import numpy


class Concat(function.Function):

    """Concatenate multiple tensors towards specified axis."""

    # concat along the channel dimension by default
    def __init__(self, axis=1, equal=False):
        if not isinstance(axis, int):
            raise TypeError('axis must be int')

        self.axis = axis
        self.equal = equal

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() > 0)
        type_check.expect(in_types[0].ndim >
                          type_check.Variable(self.axis, 'axis'))

        type_check.expect(
            -in_types[0].ndim <= self.axis,
            self.axis < in_types[0].ndim
        )
        ndim = in_types[0].ndim.eval()
        axis = self.axis % ndim
        for i in six.moves.range(1, in_types.size().eval()):
            type_check.expect(
                in_types[0].dtype == in_types[i].dtype,
                in_types[0].ndim == in_types[i].ndim,
            )
            for d in six.moves.range(0, ndim):
                if d == axis:
                    continue
                type_check.expect(in_types[0].shape[d] == in_types[i].shape[d])

    def forward(self, xs):
        #xp = cuda.get_array_module(*xs)
        if isinstance(xs[0],cupy.ndarray):
            return cupy.concatenate(xs, axis=self.axis, equal=self.equal),
        return numpy.concatenate(xs, axis=self.axis),

    def backward(self, xs, gy):
        if len(xs) == 1:
            return gy

        #xp = cuda.get_array_module(*xs)
        xp = cupy if isinstance(xs[0],cupy.ndarray) else numpy
        sizes = numpy.array([x.shape[self.axis] for x in xs[:-1]]).cumsum()
        return xp.split(gy[0], sizes, axis=self.axis)


def concat(xs, axis=1, equal=False):
    """Concatenates given variables along an axis.

    Args:
        xs (tuple of Variables): Variables to be concatenated.
        axis (int): Axis that the input arrays are concatenated along.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return Concat(axis=axis, equal=equal)(*xs)
