"""Quadrature."""

import typing
import numpy as np
import numpy.typing as npt
from ndelement._ndelementrs import lib as _lib, ffi as _ffi
from ndelement.reference_cell import ReferenceCellType, dim


def make_gauss_jacobi_quadrature(
    cell: ReferenceCellType, degree: int, dtype: typing.Type[np.floating] = np.float64
) -> typing.Tuple[npt.NDArray, npt.NDArray]:
    """Get the points and weights of a Gauss-Jacobi quadrature rule."""
    tdim = dim(cell)
    npts = _lib.gauss_jacobi_quadrature_npoints(cell.value, degree)
    points = np.empty((npts, tdim), dtype=dtype)
    weights = np.empty(npts, dtype=dtype)
    if dtype == np.float64:
        _lib.make_gauss_jacobi_quadrature_f64(
            cell.value,
            degree,
            _ffi.cast("double*", points.ctypes.data),
            _ffi.cast("double*", weights.ctypes.data),
        )
    elif dtype == np.float32:
        _lib.make_gauss_jacobi_quadrature_f32(
            cell.value,
            degree,
            _ffi.cast("float*", points.ctypes.data),
            _ffi.cast("float*", weights.ctypes.data),
        )
    else:
        raise TypeError(f"Unsupported dtype: {dtype}")
    return points, weights
