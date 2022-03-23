# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dpctl.tensor.numpy_usm_shared import ndarray
from numba.core import types

from numba_dppy.dppy_array_type import DPPYArray


class UsmSharedArrayType(DPPYArray):
    """Creates a Numba type for Numpy arrays that are stored in USM shared
    memory.  We inherit from Numba's existing Numpy array type but overload
    how this type is printed during dumping of typing information and we
    implement the special __array_ufunc__ function to determine who this
    type gets combined with scalars and regular Numpy types.
    We re-use Numpy functions as well but those are going to return Numpy
    arrays allocated in USM and we use the overloaded copy function to
    convert such USM-backed Numpy arrays into typed USM arrays."""

    def __init__(
        self,
        dtype,
        ndim,
        layout,
        readonly=False,
        name=None,
        aligned=True,
        addrspace=None,
    ):
        # This name defines how this type will be shown in Numba's type dumps.
        name = "UsmArray:ndarray(%s, %sd, %s)" % (dtype, ndim, layout)
        super(UsmSharedArrayType, self).__init__(
            dtype,
            ndim,
            layout,
            # py_type=ndarray,
            readonly=readonly,
            name=name,
            addrspace=addrspace,
        )

    def copy(self, *args, **kwargs):
        retty = super(UsmSharedArrayType, self).copy(*args, **kwargs)
        if isinstance(retty, types.Array):
            return UsmSharedArrayType(
                dtype=retty.dtype, ndim=retty.ndim, layout=retty.layout
            )
        else:
            return retty

    # Tell Numba typing how to combine UsmSharedArrayType with other ndarray types.
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == "__call__":
            for inp in inputs:
                if not isinstance(
                    inp, (UsmSharedArrayType, types.Array, types.Number)
                ):
                    return None

            return UsmSharedArrayType
        else:
            return None

    @property
    def box_type(self):
        return ndarray
