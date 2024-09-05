"""Polynomials."""

import typing
import numpy as np
import numpy.typing as npt
from ndelement._ndelementrs import lib as _lib, ffi as _ffi
from ndelement.reference_cell import ReferenceCellType


def tabulate_legendre_polynomials(
    cell: ReferenceCellType,
    points: npt.NDArray[np.floating],
    degree: int,
    derivatives: int,
    dtype: typing.Type[np.floating] = np.float64,
) -> npt.NDArray:
    """Tabulate Legendre orthonormal polynomials."""
    shape = np.empty(3, np.uintp)
    _lib.legendre_polynomials_shape(
        cell.value, points.shape[0], degree, derivatives, _ffi.cast("uintptr_t*", shape.ctypes.data)
    )
    data = np.empty(shape[::-1], dtype=dtype)
    if dtype == np.float64:
        _lib.tabulate_legendre_polynomials_f64(
            cell.value,
            _ffi.cast("double*", points.ctypes.data),
            points.shape[0],
            degree,
            derivatives,
            _ffi.cast("double*", data.ctypes.data),
        )
    elif dtype == np.float32:
        _lib.tabulate_legendre_polynomials_f32(
            cell.value,
            _ffi.cast("float*", points.ctypes.data),
            points.shape[0],
            degree,
            derivatives,
            _ffi.cast("float*", data.ctypes.data),
        )
    else:
        raise TypeError(f"Unsupported dtype: {dtype}")
    return data
