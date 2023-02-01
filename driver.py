import ctypes
import glob
import os

import dpnp
import numpy as np

import numba_dpex as dpex

paths = glob.glob(
    os.path.join(
        os.path.dirname(__file__),
        "numba_dpex/core/runtime/_rt_python.cpython-310-x86_64-linux-gnu.so",
    )
)

print("path:", paths[0])

ctypes.cdll.LoadLibrary(paths[0])


@dpex.dpjit
def foo(B):
    return 1


a = dpnp.ones(10)

b = np.ones(10)

foo(a)
