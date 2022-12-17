# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from warnings import warn

from numba.core import sigutils, types

from numba_dpex.core.exceptions import (
    IncompleteKernelSpecializationError,
    KernelHasReturnValueError,
)
from numba_dpex.core.kernel_interface.dispatcher import (
    Dispatcher,
    SpecializedDispatcher,
    get_ordered_arg_access_types,
)
from numba_dpex.core.kernel_interface.func import (
    compile_func,
    compile_func_template,
)
from numba_dpex.utils import npytypes_array_to_dpex_array


def kernel(
    func_or_sig=None, access_types=None, debug=None, device=None, usm_types=None
):
    """A decorator to define a kernel function.

    A kernel function is conceptually equivalent to a SYCL kernel function, and
    gets compiled into either an OpenCL or a LevelZero SPIR-V binary kernel.
    A kernel decorated Python function has the following restrictions:

        * The function can not return any value.
        * All array arguments passed to a kernel should adhere to compute
          follows data programming model.
    """

    def _kernel_dispatcher(pyfunc):
        ordered_arg_access_types = get_ordered_arg_access_types(
            pyfunc, access_types
        )
        return Dispatcher(
            pyfunc=pyfunc,
            debug_flags=debug,
            array_access_specifiers=ordered_arg_access_types,
        )

    if func_or_sig is None:
        if device:
            warn(
                "The device keyword applies only when specializing a "
                + "kernel for a specific signature. For other cases, the "
                + " device is derived using the compute follows data "
                + "programming model."
            )
        if usm_types:
            warn(
                "The usm_types keyword applies only when specializing a "
                + "kernel for a specific signature."
            )
        return _kernel_dispatcher
    elif not sigutils.is_signature(func_or_sig):
        if device:
            warn(
                "The device keyword applies only when specializing a "
                + "kernel for a specific signature. For other cases, the "
                + " device is derived using the compute follows data "
                + "programming model."
            )
        if usm_types:
            warn(
                "The usm_types keyword applies only when specializing a "
                + "kernel for a specific signature."
            )
        func = func_or_sig
        return _kernel_dispatcher(func)
    else:
        # if not device and not usm_types:
        #     raise IncompleteKernelSpecializationError()
        # elif not usm_types:
        #     raise IncompleteKernelSpecializationError(no_device=False)
        # elif not device:
        #     raise IncompleteKernelSpecializationError(no_usm_types=False)

        argtypes, return_type = sigutils.normalize_signature(func_or_sig)
        print(argtypes)
        if return_type and return_type != types.void:
            raise KernelHasReturnValueError(
                kernel_name=None, return_type=return_type, sig=func_or_sig
            )

        def _specialized(pyfunc):
            return SpecializedDispatcher(
                pyfunc=pyfunc,
                argtypes=argtypes,
                device=device,
                usm_types=usm_types,
                debug_flags=debug,
            )

        return _specialized


# def autojit(debug=None, access_types=None):
#     def _kernel_dispatcher(pyfunc):
#         ordered_arg_access_types = get_ordered_arg_access_types(
#             pyfunc, access_types
#         )
#         return Dispatcher(
#             pyfunc=pyfunc,
#             debug_flags=debug,
#             array_access_specifiers=ordered_arg_access_types,
#         )

#     return _kernel_dispatcher


# def _kernel_jit(signature, debug, access_types):
#     argtypes, _ = sigutils.normalize_signature(signature)
#     argtypes = tuple(
#         [
#             npytypes_array_to_dpex_array(ty)
#             if isinstance(ty, types.npytypes.Array)
#             else ty
#             for ty in argtypes
#         ]
#     )

#     def _wrapped(pyfunc):
#         current_queue = dpctl.get_current_queue()
#         ordered_arg_access_types = get_ordered_arg_access_types(
#             pyfunc, access_types
#         )
#         # We create an instance of JitKernel to make sure at call time
#         # we are going through the caching mechanism.
#         kernel = JitKernel(pyfunc, debug, ordered_arg_access_types)
#         # This will make sure we are compiling eagerly.
#         kernel.specialize(argtypes, current_queue)
#         return kernel

#     return _wrapped


def func(signature=None, debug=None):
    if signature is None:
        return _func_autojit_wrapper(debug=debug)
    elif not sigutils.is_signature(signature):
        func = signature
        return _func_autojit(func, debug=debug)
    else:
        return _func_jit(signature, debug=debug)


def _func_jit(signature, debug=None):
    argtypes, restype = sigutils.normalize_signature(signature)
    argtypes = tuple(
        [
            npytypes_array_to_dpex_array(ty)
            if isinstance(ty, types.npytypes.Array)
            else ty
            for ty in argtypes
        ]
    )

    def _wrapped(pyfunc):
        return compile_func(pyfunc, restype, argtypes, debug=debug)

    return _wrapped


def _func_autojit_wrapper(debug=None):
    def _func_autojit(pyfunc, debug=debug):
        return compile_func_template(pyfunc, debug=debug)

    return _func_autojit


def _func_autojit(pyfunc, debug=None):
    return compile_func_template(pyfunc, debug=debug)
