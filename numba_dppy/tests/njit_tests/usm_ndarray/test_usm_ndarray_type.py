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

import numpy as np
import pytest
from dpctl.tensor import usm_ndarray
from numba.misc.special import typeof

from numba_dppy.dpctl_iface import USMNdArrayType

dtypes = [np.int32, np.float32, np.int64, np.float64]
usm_types = ["shared", "device", "host"]


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("usm_type", usm_types)
def test_usm_ndarray_type(dtype, usm_type):
    a = np.array(np.random.random(10), dtype)
    da = usm_ndarray(a.shape, dtype=a.dtype, buffer=usm_type)

    assert isinstance(typeof(da), USMNdArrayType)
    assert da.usm_type == usm_type
