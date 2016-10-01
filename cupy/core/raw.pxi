import string

import numpy
import six

from cupy import util

from cupy.cuda cimport device
from cupy.cuda cimport function


@util.memoize(for_each_device=True)
def _get_raw_kernel(
        module_code, name,
        options=()):
    module = compile_with_cache(module_code, options)
    return module.get_function(name)


cdef class RawKernel:

    """User-defined raw CUDA kernel.

    This class can be used to define a raw CUDA kernel by writing the entire
    function declaration and body as CUDA-C code.

    The kernel is compiled at an invocation of the
    :meth:`~RawKernel.__call__` method, which is cached for each device.
    The compiled binary is also cached into a file under the
    ``$HOME/.cupy/kernel_cache/`` directory with a hashed file name. The cached
    binary is reused by other processes.

    Args:
        operation (str): Raw CUDA-C/C++ code with one or more kernels.
        name (str): Name of the kernel function to call. It should be set for
            readability of the performance profiling. It must be identical
            to the name of a function defined in the CUDA-C code.
        options (tuple): Options passed to the ``nvcc`` command.

    """

    cdef:
        readonly str operation
        readonly str name
        readonly tuple options

    def __init__(self, operation,
                 name='kernel', options=()):
        self.operation = operation
        self.name = name
        self.options = options

    def __call__(self, grid=(1,), block=(1,), *args, stream=None):
        """Compiles and invokes the raw kernel.

        The compilation runs only if the kernel is not cached.

        Args:
            grid (tuple): Grid sizes (number of blocks in x,y,z dimensions).
            block (tuple): Block sizes (number of threads/block in x,y,z dims).
            args: Arguments of the kernel.
            stream: CUDA stream or None.

        Returns:
            None

        """

        cdef function.Function kern

        kern = _get_raw_kernel(
            self.operation,
            self.name, options=self.options)
        kern(grid, block, *args, stream=stream)
