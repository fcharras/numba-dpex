# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import llvmlite.ir as lir
import llvmlite.llvmpy.core as lc
from numba.core import cgutils, types


class LLVMTypes:
    """
    A helper class to get LLVM Values for integer C types.
    """

    byte_t = lc.Type.int(8)
    byte_ptr_t = lc.Type.pointer(byte_t)
    byte_ptr_ptr_t = lc.Type.pointer(byte_ptr_t)
    int32_t = lc.Type.int(32)
    int32_ptr_t = lc.Type.pointer(int32_t)
    int64_t = lc.Type.int(64)
    int64_ptr_t = lc.Type.pointer(int64_t)
    void_t = lir.VoidType()


def get_llvm_type(context, type):
    """Returns the LLVM Value corresponsing to a Numba type.

    Args:
        context: The LLVM context or the execution state of the current IR
        generator.
        type: A Numba type object.

    Returns: An Python object wrapping an LLVM Value corresponding to the
             specified Numba type.

    """
    return context.get_value_type(type)


def get_llvm_ptr_type(type):
    """Returns an LLVM pointer type for a give LLVM type object.

    Args:
        type: An LLVM type for which we need the corresponding pointer type.

    Returns: An LLVM pointer type object corresponding to the input LLVM type.

    """
    return lc.Type.pointer(type)


def create_null_ptr(builder, context):
    """
    Allocates a new LLVM Value storing a ``void*`` and returns the Value to
    caller.

    Args:
        builder: The LLVM IR builder to be used for code generation.
        context: The LLVM IR builder context.

    Returns: An LLVM value storing a null pointer

    """
    null_ptr = cgutils.alloca_once(
        builder=builder,
        ty=context.get_value_type(types.voidptr),
        size=context.get_constant(types.uintp, 1),
    )
    builder.store(
        builder.inttoptr(
            context.get_constant(types.uintp, 0),
            get_llvm_type(context=context, type=types.voidptr),
        ),
        null_ptr,
    )
    return null_ptr


def get_zero(context):
    """Returns an LLVM Constant storing a 64 bit representation for zero.

    Args:
        context: The LLVM IR builder context.

    Returns: An LLVM Contant Value storing zero.

    """
    return context.get_constant(types.uintp, 0)


def get_one(context):
    """Returns an LLVM Constant storing a 64 bit representation for one.

    Args:
        context: The LLVM IR builder context.

    Returns: An LLVM Contant Value storing one.

    """
    return context.get_constant(types.uintp, 1)