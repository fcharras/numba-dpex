// SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
///
/// \file
/// Helper functions for converting between a Python object for a dpnp.ndarray
/// and its corresponding internal Numba representation.
///
//===----------------------------------------------------------------------===//

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#include <numpy/arrayscalars.h>
#include <numpy/ndarrayobject.h>

#include "numba/_arraystruct.h"
#include "numba/_numba_common.h"
#include "numba/_pymodule.h"
#include "numba/core/runtime/nrt.h"
#include "stdbool.h"

#include "dpctl_capi.h"
#include "dpctl_sycl_interface.h"

#define SYCL_USM_ARRAY_INTERFACE "__sycl_usm_array_interface__"

/*
 * The MemInfo structure.
 * NOTE: copy from numba/core/runtime/nrt.c
 */
struct MemInfo
{
    size_t refct;
    NRT_dtor_function dtor;
    void *dtor_info;
    void *data;
    size_t size; /* only used for NRT allocated memory */
    NRT_ExternalAllocator *external_allocator;
};

/*!
 * @brief A wrapper struct to store a MemInfo pointer along with the PyObject
 * that is associated with the MeMinfo.
 *
 * The struct is stored in the dtor_info attribute of a MemInfo object and
 * used by the destructor to free the MemInfo and DecRef the Pyobject.
 *
 */
typedef struct
{
    PyObject *owner;
    NRT_MemInfo *mi;
} MemInfoDtorInfo;

// forward declarations
static struct PyUSMArrayObject *PyUSMNdArray_ARRAYOBJ(PyObject *obj);
static npy_intp UsmNDArray_GetSize(npy_intp *shape, npy_intp ndim);
static void *usm_device_malloc(size_t size, void *opaque_data);
static void *usm_shared_malloc(size_t size, void *opaque_data);
static void *usm_host_malloc(size_t size, void *opaque_data);
static void usm_free(void *data, void *opaque_data);
static NRT_ExternalAllocator *
NRT_ExternalAllocator_new_for_usm(DPCTLSyclQueueRef qref, void *data);
static MemInfoDtorInfo *MemInfoDtorInfo_new(NRT_MemInfo *mi, PyObject *owner);
static NRT_MemInfo *NRT_MemInfo_new_from_usmndarray(PyObject *ndarrobj,
                                                    void *data,
                                                    npy_intp nitems,
                                                    npy_intp itemsize,
                                                    DPCTLSyclQueueRef qref);
static void usmndarray_meminfo_dtor(void *ptr, size_t size, void *info);

/*
 * Debugging printf function used internally
 */
void nrt_debug_print(char *fmt, ...)
{
    va_list args;

    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
}

/** An NRT_external_malloc_func implementation using DPCTLmalloc_device.
 *
 */
static void *usm_device_malloc(size_t size, void *opaque_data)
{
    DPCTLSyclQueueRef qref = NULL;

    qref = (DPCTLSyclQueueRef)opaque_data;
    return DPCTLmalloc_device(size, qref);
}

/** An NRT_external_malloc_func implementation using DPCTLmalloc_shared.
 *
 */
static void *usm_shared_malloc(size_t size, void *opaque_data)
{
    DPCTLSyclQueueRef qref = NULL;

    qref = (DPCTLSyclQueueRef)opaque_data;
    return DPCTLmalloc_shared(size, qref);
}

/** An NRT_external_malloc_func implementation using DPCTLmalloc_host.
 *
 */
static void *usm_host_malloc(size_t size, void *opaque_data)
{
    DPCTLSyclQueueRef qref = NULL;

    qref = (DPCTLSyclQueueRef)opaque_data;
    return DPCTLmalloc_host(size, qref);
}

/** An NRT_external_free_func implementation based on DPCTLfree_with_queue
 *
 */
static void usm_free(void *data, void *opaque_data)
{
    DPCTLSyclQueueRef qref = NULL;
    qref = (DPCTLSyclQueueRef)opaque_data;

    DPCTLfree_with_queue(data, qref);
}

/**
 *
 */
static NRT_ExternalAllocator *
NRT_ExternalAllocator_new_for_usm(DPCTLSyclQueueRef qref, void *data)
{
    DPCTLSyclContextRef cref = NULL;
    const char *usm_type_name = NULL;

    NRT_ExternalAllocator *allocator =
        (NRT_ExternalAllocator *)malloc(sizeof(NRT_ExternalAllocator));
    if (allocator == NULL) {
        nrt_debug_print(
            "Fatal: failed to allocate memory for NRT_ExternalAllocator.\n");
        exit(-1);
    }

    // get the usm pointer type
    cref = DPCTLQueue_GetContext(qref);
    usm_type_name = DPCTLUSM_GetPointerType(data, cref);
    DPCTLContext_Delete(cref);

    if (usm_type_name)
        switch (usm_type_name[0]) {
        case 'd':
            allocator->malloc = usm_device_malloc;
            break;
        case 's':
            allocator->malloc = usm_shared_malloc;
            break;
        case 'h':
            allocator->malloc = usm_host_malloc;
            break;
        default:
            printf(
                "DPEXRT Fatal: Encountered an unknown usm allocation type.\n");
            exit(-1);
        }

    DPCTLCString_Delete(usm_type_name);

    allocator->realloc = NULL;
    allocator->free = usm_free;
    allocator->opaque_data = (void *)qref;

    return allocator;
}

/**
 * Free data
 * decref arrayobj
 * call free of external_allocator
 * Free external_allocator
 * Set external_allocator to NULL
 */
/**
 * @brief Destructor called when a MemInfo object allocated by Dpex RT is freed
 * by Numba using the NRT_MemInfor_release function.
 *
 * The destructor does the following clean up:
 *     - Frees the data associated with the MemInfo object if there was no
 *       parent PyObject that owns the data.
 *     - Frees the DpctlSyclQueueRef pointer stored in the opaque data of the
 *       MemInfo's external_allocator member.
 *     - Frees the external_allocator object associated with the MemInfo object.
 *     - If there was a PyObject associated with the MemInfo, then
 *       the reference count on that object.
 *     - Frees the MemInfoDtorInfo wrapper object that was stored as the
 *       dtor_info member of the MemInfo.
 */
static void usmndarray_meminfo_dtor(void *ptr, size_t size, void *info)
{
    MemInfoDtorInfo *mi_dtor_info = NULL;

    mi_dtor_info = (MemInfoDtorInfo *)info;

    // If there is no owner PythonObject, free the data by calling the
    // external_allocator->free
    if (!(mi_dtor_info->owner))
        mi_dtor_info->mi->external_allocator->free(
            mi_dtor_info->mi->data,
            mi_dtor_info->mi->external_allocator->opaque_data);

    // free the DpctlSyclQueueRef object stored inside the external_allocator
    DPCTLQueue_Delete(mi_dtor_info->mi->external_allocator->opaque_data);

    // free the external_allocator object
    free(mi_dtor_info->mi->external_allocator);

    // Set the pointer to NULL to prevent NRT_dealloc trying to use it free
    // the meminfo object
    mi_dtor_info->mi->external_allocator = NULL;

    if (mi_dtor_info->owner) {
        // Decref the Pyobject from which the MemInfo was created
        PyGILState_STATE gstate;
        PyObject *ownerobj = mi_dtor_info->owner;
        // ensure the GIL
        gstate = PyGILState_Ensure();
        // decref the python object
        Py_DECREF(ownerobj);
        // release the GIL
        PyGILState_Release(gstate);
    }

    // Free the MemInfoDtorInfo object
    free(mi_dtor_info);
}

static MemInfoDtorInfo *MemInfoDtorInfo_new(NRT_MemInfo *mi, PyObject *owner)
{
    MemInfoDtorInfo *mi_dtor_info = NULL;

    if (!(mi_dtor_info = (MemInfoDtorInfo *)malloc(sizeof(MemInfoDtorInfo)))) {
        nrt_debug_print(
            "DPEXRT-FATAL: Could not allocate a new MemInfoDtorInfo object.\n");
        exit(-1);
    }
    mi_dtor_info->mi = mi;
    mi_dtor_info->owner = owner;

    return mi_dtor_info;
}

static NRT_MemInfo *NRT_MemInfo_new_from_usmndarray(PyObject *ndarrobj,
                                                    void *data,
                                                    npy_intp nitems,
                                                    npy_intp itemsize,
                                                    DPCTLSyclQueueRef qref)
{
    NRT_MemInfo *mi = NULL;
    NRT_ExternalAllocator *ext_alloca = NULL;
    MemInfoDtorInfo *midtor_info = NULL;

    Py_IncRef(ndarrobj);

    // Allocate a new NRT_MemInfo object
    if (!(mi = (NRT_MemInfo *)malloc(sizeof(NRT_MemInfo)))) {
        nrt_debug_print(
            "DPEXRT-FATAL: Could not allocate a new NRT_MemInfo object.\n");
        exit(-1);
    }

    // Allocate a new NRT_ExternalAllocator
    ext_alloca = NRT_ExternalAllocator_new_for_usm(qref, data);

    // Allocate a new MemInfoDtorInfo
    midtor_info = MemInfoDtorInfo_new(mi, ndarrobj);

    // Initialize the NRT_MemInfo object
    mi->refct = 1; /* starts with 1 refct */
    mi->dtor = usmndarray_meminfo_dtor;
    mi->dtor_info = midtor_info;
    mi->data = data;
    mi->size = nitems * itemsize;
    mi->external_allocator = ext_alloca;

    nrt_debug_print(
        "DPEXRT-DEBUG: NRT_MemInfo_init mi=%p external_allocator=%p\n", mi,
        ext_alloca);

    return mi;
}

/*--------- Helpers to get attributes out of a dpnp.ndarray PyObject ---------*/

static struct PyUSMArrayObject *PyUSMNdArray_ARRAYOBJ(PyObject *obj)
{
    PyObject *arrayobj = NULL;

    arrayobj = PyObject_GetAttrString(obj, "_array_obj");

    if (!arrayobj)
        return NULL;
    if (!PyObject_TypeCheck(arrayobj, &PyUSMArrayType))
        return NULL;

    Py_INCREF(arrayobj);

    struct PyUSMArrayObject *pyusmarrayobj =
        (struct PyUSMArrayObject *)(arrayobj);

    return pyusmarrayobj;
}

static npy_intp UsmNDArray_GetSize(npy_intp *shape, npy_intp ndim)
{
    npy_intp nelems = 1;

    for (int i = 0; i < ndim; ++i)
        nelems *= shape[i];

    return nelems;
}

static int DPEXRT_sycl_usm_ndarray_from_python(PyObject *obj,
                                               arystruct_t *arystruct)
{
    struct PyUSMArrayObject *arrayobj = NULL;
    int i, ndim;
    npy_intp *shape = NULL, *strides = NULL;
    npy_intp *p = NULL, nitems, itemsize;
    void *data = NULL;
    DPCTLSyclQueueRef qref = NULL;

    nrt_debug_print("DPEXRT-DEBUG: DPEXRT_sycl_usm_ndarray_from_python");

    // Check if the PyObject obj has an _array_obj attribute that is of
    // dpctl.tensor.usm_ndarray type.
    // NOTE: If the PyUSMNdArray_ARRAYOBJ succeeded in extracting the _array_obj
    // attribute, the a PyIncref(obj) was performed by PyUSMNdArray_ARRAYOBJ.
    if (!(arrayobj = PyUSMNdArray_ARRAYOBJ(obj)))
        return -1;

    ndim = UsmNDArray_GetNDim(arrayobj);
    shape = UsmNDArray_GetShape(arrayobj);
    strides = UsmNDArray_GetStrides(arrayobj);
    data = (void *)UsmNDArray_GetData(arrayobj);
    nitems = UsmNDArray_GetSize(shape, ndim);
    itemsize = (npy_intp)UsmNDArray_GetElementSize(arrayobj);
    qref = UsmNDArray_GetQueueRef(arrayobj);

    arystruct->meminfo =
        NRT_MemInfo_new_from_usmndarray(obj, data, nitems, itemsize, qref);

    arystruct->data = data;
    arystruct->nitems = nitems;
    arystruct->itemsize = itemsize;
    arystruct->parent = obj;

    p = arystruct->shape_and_strides;

    for (i = 0; i < ndim; ++i, ++p)
        *p = shape[i];
    for (i = 0; i < ndim; ++i, ++p)
        *p = strides[i];

    // Decref the arrayobj, it was incremented by PyUSMNdArray_ARRAYOBJ
    Py_DECREF(arrayobj);

    return 0;
}

static PyObject *
try_to_return_parent(arystruct_t *arystruct, int ndim, PyArray_Descr *descr)
{
    int i;
    PyObject *array = arystruct->parent;

    // if (!PyArray_Check(arystruct->parent))
    //     /* Parent is a generic buffer-providing object */
    //     goto RETURN_ARRAY_COPY;

    // if (PyArray_DATA(array) != arystruct->data)
    //     goto RETURN_ARRAY_COPY;

    // if (PyArray_NDIM(array) != ndim)
    //     goto RETURN_ARRAY_COPY;

    // if (PyObject_RichCompareBool((PyObject *) PyArray_DESCR(array),
    //                              (PyObject *) descr, Py_EQ) <= 0)
    //     goto RETURN_ARRAY_COPY;

    // for(i = 0; i < ndim; ++i) {
    //     if (PyArray_DIMS(array)[i] != arystruct->shape_and_strides[i])
    //         goto RETURN_ARRAY_COPY;
    //     if (PyArray_STRIDES(array)[i] != arystruct->shape_and_strides[ndim +
    //     i])
    //         goto RETURN_ARRAY_COPY;
    // }

    /* Yes, it is the same array return a new reference */
    Py_INCREF((PyObject *)array);
    return (PyObject *)array;

    // RETURN_ARRAY_COPY:
    //     return NULL;
}

/*!
 * @brief Used to implement the boxing, i.e., conversion from Numba
 * representation to a Python object, for a dpnp.ndarray object.
 *
 * @param arystruct The Numba internal representation of a dpnp.ndarray object.
 * `retty` is the subtype of the NumPy PyArray_Type this function should return.
 *         This is related to `numba.core.types.Array.box_type`.
 * `ndim` is the number of dimension of the array.
 * `writeable` corresponds to the "writable" flag in NumPy ndarray.
 * `descr` is the NumPy data type description.
 *
 */

/**
 * This function is used during the boxing of dpnp.ndarray type.

 * It used to steal the reference of the arystruct.
 */
static PyObject *
DPEXRT_sycl_usm_ndarray_to_python_acqref(arystruct_t *arystruct,
                                         PyTypeObject *retty,
                                         int ndim,
                                         int writeable,
                                         PyArray_Descr *descr)
{
    // PyArrayObject *array;
    PyObject *array;
    //    MemInfoObject *miobj = NULL;
    //    PyObject *args;
    //     npy_intp *shape, *strides;
    //     int flags = 0;

    if (descr == NULL) {
        PyErr_Format(
            PyExc_RuntimeError,
            "In 'DPEXRT_sycl_usm_ndarray_to_python_acqref', 'descr' is NULL");
        return NULL;
    }

    if (!NUMBA_PyArray_DescrCheck(descr)) {
        PyErr_Format(PyExc_TypeError, "expected dtype object, got '%.200s'",
                     Py_TYPE(descr)->tp_name);
        return NULL;
    }

    if (arystruct->parent) {
        PyObject *obj = try_to_return_parent(arystruct, ndim, descr);
        if (obj) {
            return obj;
        }
    }

    //     if (arystruct->meminfo) {
    //         /* wrap into MemInfoObject */
    //         miobj = PyObject_New(MemInfoObject, &MemInfoType);
    //         args = PyTuple_New(1);
    //         /* SETITEM steals reference */
    //         PyTuple_SET_ITEM(args, 0,
    //         PyLong_FromVoidPtr(arystruct->meminfo));
    //         NRT_Debug(nrt_debug_print("NRT_adapt_ndarray_to_python
    //         arystruct->meminfo=%p\n", arystruct->meminfo));
    //         /*  Note: MemInfo_init() does not incref.  This function steals
    //         the
    //          *        NRT reference, which we need to acquire.
    //          */
    //         NRT_Debug(nrt_debug_print("NRT_adapt_ndarray_to_python_acqref
    //         created MemInfo=%p\n", miobj));
    //         NRT_MemInfo_acquire(arystruct->meminfo);
    //         if (MemInfo_init(miobj, args, NULL)) {
    //             NRT_Debug(nrt_debug_print("MemInfo_init failed.\n"));
    //             return NULL;
    //         }
    //         Py_DECREF(args);
    //     }

    //     shape = arystruct->shape_and_strides;
    //     strides = shape + ndim;
    //     Py_INCREF((PyObject *) descr);
    //     array = (PyArrayObject *) PyArray_NewFromDescr(retty, descr, ndim,
    //                                                    shape, strides,
    //                                                    arystruct->data,
    //                                                    flags, (PyObject *)
    //                                                    miobj);

    //     if (array == NULL)
    //         return NULL;

    //     /* Set writable */
    // #if NPY_API_VERSION >= 0x00000007
    //     if (writeable) {
    //         PyArray_ENABLEFLAGS(array, NPY_ARRAY_WRITEABLE);
    //     }
    //     else {
    //         PyArray_CLEARFLAGS(array, NPY_ARRAY_WRITEABLE);
    //     }
    // #else
    //     if (writeable) {
    //         array->flags |= NPY_WRITEABLE;
    //     }
    //     else {
    //         array->flags &= ~NPY_WRITEABLE;
    //     }
    // #endif

    //     if (miobj) {
    //         /* Set the MemInfoObject as the base object */
    // #if NPY_API_VERSION >= 0x00000007
    //         if (-1 == PyArray_SetBaseObject(array,
    //                                         (PyObject *) miobj))
    //         {
    //             Py_DECREF(array);
    //             Py_DECREF(miobj);
    //             return NULL;
    //         }
    // #else
    //         PyArray_BASE(array) = (PyObject *) miobj;
    // #endif

    //     }
    return (PyObject *)array;
}

static PyObject *build_c_helpers_dict(void)
{
    PyObject *dct = PyDict_New();
    if (dct == NULL)
        goto error;

#define _declpointer(name, value)                                              \
    do {                                                                       \
        PyObject *o = PyLong_FromVoidPtr(value);                               \
        if (o == NULL)                                                         \
            goto error;                                                        \
        if (PyDict_SetItemString(dct, name, o)) {                              \
            Py_DECREF(o);                                                      \
            goto error;                                                        \
        }                                                                      \
        Py_DECREF(o);                                                          \
    } while (0)

    _declpointer("DPEXRT_sycl_usm_ndarray_from_python",
                 &DPEXRT_sycl_usm_ndarray_from_python);
    _declpointer("DPEXRT_sycl_usm_ndarray_to_python_acqref",
                 &DPEXRT_sycl_usm_ndarray_to_python_acqref);

#undef _declpointer
    return dct;
error:
    Py_XDECREF(dct);
    return NULL;
}

MOD_INIT(_rt_python)
{
    PyObject *m;
    MOD_DEF(m, "_rt_python", "No docs", NULL)
    if (m == NULL)
        return MOD_ERROR_VAL;

    import_array();
    import_dpctl();

    PyModule_AddObject(
        m, "DPEXRT_sycl_usm_ndarray_from_python",
        PyLong_FromVoidPtr(&DPEXRT_sycl_usm_ndarray_from_python));
    PyModule_AddObject(
        m, "DPEXRT_sycl_usm_ndarray_to_python_acqref",
        PyLong_FromVoidPtr(&DPEXRT_sycl_usm_ndarray_to_python_acqref));

    // PyModule_AddObject(m, "PySyclUsmArray_Check",
    //                    PyLong_FromVoidPtr(&PyUSMNdArray_Check));
    // PyModule_AddObject(m, "PyUSMNdArray_NDIM",
    //                    PyLong_FromVoidPtr(&PyUSMNdArray_NDIM));

    // PyModule_AddObject(m, "itemsize_from_typestr",
    //                    PyLong_FromVoidPtr(&itemsize_from_typestr));

    PyModule_AddObject(m, "c_helpers", build_c_helpers_dict());
    return MOD_SUCCESS_VAL(m);
}
