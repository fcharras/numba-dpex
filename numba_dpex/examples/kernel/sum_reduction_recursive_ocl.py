# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
There are multiple ways of implementing reduction using numba_dpex. Here we
demonstrate another way of implementing reduction using recursion to compute
partial reductions in separate kernels.
"""

import dpctl
import dpctl.tensor as dpt
from numba import int32

import numba_dpex as dpex
from numba_dpex.core.kernel_interface.utils import NdRange, Range


@dpex.kernel
def sum_reduction_kernel(A, input_size, partial_sums):
    local_id = dpex.get_local_id(0)
    global_id = dpex.get_global_id(0)
    group_size = dpex.get_local_size(0)
    group_id = dpex.get_group_id(0)

    local_sums = dpex.local.array(64, int32)

    local_sums[local_id] = 0

    if global_id < input_size:
        local_sums[local_id] = A[global_id]

    # Loop for computing local_sums : divide workgroup into 2 parts
    stride = group_size // 2
    while stride > 0:
        # Waiting for each 2x2 addition into given workgroup
        dpex.barrier(dpex.LOCAL_MEM_FENCE)

        # Add elements 2 by 2 between local_id and local_id + stride
        if local_id < stride:
            local_sums[local_id] += local_sums[local_id + stride]

        stride >>= 1

    if local_id == 0:
        partial_sums[group_id] = local_sums[0]


def sum_recursive_reduction(size, group_size, Dinp, Dpartial_sums):
    result = 0
    nb_work_groups = 0
    passed_size = size

    if size <= group_size:
        nb_work_groups = 1
    else:
        nb_work_groups = size // group_size
        if size % group_size != 0:
            nb_work_groups += 1
            passed_size = nb_work_groups * group_size

    gr = Range(passed_size)
    lr = Range(group_size)

    sum_reduction_kernel[NdRange(gr, lr)](Dinp, size, Dpartial_sums)

    if nb_work_groups <= group_size:
        sum_reduction_kernel[NdRange(lr, lr)](
            Dpartial_sums, nb_work_groups, Dinp
        )
        result = int(Dinp[0])
    else:
        result = sum_recursive_reduction(
            nb_work_groups, group_size, Dpartial_sums, Dinp
        )

    return result


def sum_reduce(A):
    global_size = len(A)
    work_group_size = 64
    nb_work_groups = global_size // work_group_size
    if (global_size % work_group_size) != 0:
        nb_work_groups += 1

    partial_sums = dpt.zeros(nb_work_groups, dtype=A.dtype, device=A.device)
    result = sum_recursive_reduction(
        global_size, work_group_size, A, partial_sums
    )

    return result


def test_sum_reduce():
    N = 20000
    device = dpctl.select_default_device()
    A = dpt.ones(N, dtype=dpt.int32, device=device)

    print("Running recursive reduction")

    actual = sum_reduce(A)
    expected = N

    print("Actual:  ", actual)
    print("Expected:", expected)

    assert actual == expected

    print("Done...")


if __name__ == "__main__":
    test_sum_reduce()
