# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0


from numba.core import compiler, dispatcher
from numba.core.target_extension import dispatcher_registry, target_registry

from numba_dpex.core.targets.dpjit_target import DPEX_TARGET_NAME

from .descriptor import dpex_target


class DpjitDispatcher(dispatcher.Dispatcher):
    targetdescr = dpex_target

    def __init__(
        self,
        py_func,
        locals={},
        targetoptions={},
        impl_kind="direct",
        pipeline_class=compiler.Compiler,
    ):
        dispatcher.Dispatcher.__init__(
            self,
            py_func,
            locals=locals,
            targetoptions=targetoptions,
            impl_kind=impl_kind,
            pipeline_class=pipeline_class,
        )


dispatcher_registry[target_registry[DPEX_TARGET_NAME]] = DpjitDispatcher
