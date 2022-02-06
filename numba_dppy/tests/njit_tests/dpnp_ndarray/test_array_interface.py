"""Tests for array interfaces.

Like __array_interface__ and __sycl_usm_array_interface__.
"""
import dpctl
import dpnp
import pytest


def test_dpnp_ndarray_sycl_usm_array_interface():
    a = dpnp.ndarray([10])
    assert hasattr(a, "__sycl_usm_array_interface__")

    i = a.__sycl_usm_array_interface__

    assert sorted(list(i.keys())) == [
        "data",
        "offset",
        "shape",
        "strides",
        "syclobj",
        "typestr",
        "version",
    ]

    assert i["data"][1] == True  # noqa
    assert i["offset"] == 0
    assert i["shape"] == (10,)
    assert i["strides"] is None
    assert isinstance(i["syclobj"], dpctl.SyclQueue)
    assert i["typestr"] == "|f8"
    assert i["version"] == 1


def test_numpy_ndarray_array_interface():
    """Test properties of __array_interface__ of NumPy ndarray"""
    import numpy

    a = numpy.ndarray([10])
    assert hasattr(a, "__array_interface__")

    i = a.__array_interface__

    assert sorted(list(i.keys())) == [
        "data",
        "descr",
        "shape",
        "strides",
        "typestr",
        "version",
    ]

    assert isinstance(i["data"], tuple)
    assert len(i["data"]) == 2
    assert isinstance(i["data"][0], int)
    assert i["data"][1] == False  # noqa

    assert i["descr"] == [("", "<f8")]
    assert i["shape"] == (10,)
    assert i["strides"] is None
    assert i["typestr"] == "<f8"
    assert i["version"] == 3


def test_array_interface_and_sycl_usm_array_interface_difference():
    """Test what specific properties __array_interface__ and __sycl_usm_array_interface__ have"""

    import dpnp
    import numpy

    numpy_array = numpy.ndarray([10])
    dpnp_array = dpnp.ndarray([10])

    numpy_keys = numpy_array.__array_interface__.keys()
    dpnp_keys = dpnp_array.__sycl_usm_array_interface__.keys()

    assert sorted(numpy_keys - dpnp_keys) == ["descr"]
    assert sorted(dpnp_keys - numpy_keys) == ["offset", "syclobj"]

    # From https://numpy.org/doc/stable/reference/arrays.interface.html
    # __array_interface__ have:
    #   shape (required)
    #   typestr (required)
    #   descr (optional)
    #   data (optional) or buffer protocol
    #   strides (optional)
    #   mask (optional)
    #   offset (optional)
    #   version (required)


@pytest.mark.parametrize(
    "array, shape",
    [
        (dpnp.ndarray([10]), (10,)),
        (dpnp.ndarray([2, 3]), (2, 3)),
    ],
)
def test_dpnp_ndarray_sycl_usm_array_interface_shape(array, shape):
    i = array.__sycl_usm_array_interface__
    assert i["shape"] == shape
