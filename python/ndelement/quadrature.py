"""Quadrature."""

import typing
import numpy as np
import numpy.typing as npt
from ndelement._ndelementrs import lib as _lib, ffi as _ffi
from ndelement.reference_cell import ReferenceCellType, dim


def make_gauss_jacobi_quadrature(
    cell: ReferenceCellType, degree: int
) -> typing.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Get the points and weights of a Gauss-Jacobi quadrature rule."""
    tdim = dim(cell)
    npts = _lib.gauss_jacobi_quadrature_npoints(cell.value, degree)
    points = np.empty((npts, tdim), dtype=np.float64)
    weights = np.empty(npts, dtype=np.float64)
    _lib.make_gauss_jacobi_quadrature(
        cell.value,
        degree,
        _ffi.cast("double*", points.ctypes.data),
        _ffi.cast("double*", weights.ctypes.data),
    )
    return points, weights
