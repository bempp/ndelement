"""Ciarlet elements."""

import typing
import numpy as np
import numpy.typing as npt
from ndelement._ndelementrs import lib as _lib, ffi as _ffi
from ndelement.reference_cell import ReferenceCellType, entity_counts, dim
from enum import Enum
from _cffi_backend import _CDataBase


class Continuity(Enum):
    """Continuity."""

    Standard = _lib.Continuity_Standard
    Discontinuous = _lib.Continuity_Discontinuous


class Family(Enum):
    """Element family."""

    Lagrange = 0
    RaviartThomas = 1
    NedelecFirstKind = 2


_rtypes = {
    np.float32: _lib.DType_F32,
    np.float64: _lib.DType_F64,
    np.complex64: _lib.DType_C32,
    np.complex128: _lib.DType_C64,
}

_dtypes = {j: i for i, j in _rtypes.items()}


class CiarletElement(object):
    """Ciarlet element."""

    def __init__(self, rs_element: _CDataBase, owned: bool = True):
        """Initialise."""
        self._rs_element = rs_element
        self._owned = owned

    def __del__(self):
        """Delete object."""
        if self._owned:
            _lib.ciarlet_element_t_free(self._rs_element)

    @property
    def dtype(self) -> typing.Type[np.floating]:
        """Data type."""
        return _dtypes[_lib.ciarlet_element_dtype(self._rs_element)]

    @property
    def value_size(self) -> int:
        """Value size of the element."""
        return _lib.element_value_size(self._rs_element)

    @property
    def value_shape(self) -> tuple[int, ...]:
        """Value size of the element."""
        shape = np.empty(_lib.ciarlet_element_value_rank(self._rs_element), dtype=np.uintp)
        _lib.ciarlet_element_value_shape(
            self._rs_element, _ffi.cast("uintptr_t*", shape.ctypes.data)
        )
        return tuple(int(i) for i in shape)

    @property
    def degree(self) -> int:
        """Degree of the element."""
        return _lib.ciarlet_element_degree(self._rs_element)

    @property
    def embedded_superdegree(self) -> int:
        """Embedded superdegree of the element."""
        return _lib.ciarlet_element_embedded_superdegree(self._rs_element)

    @property
    def dim(self) -> int:
        """Dimension (number of basis functions) of the element."""
        return _lib.ciarlet_element_dim(self._rs_element)

    @property
    def continuity(self) -> Continuity:
        """Continuity of the element."""
        return Continuity(_lib.ciarlet_element_continuity(self._rs_element))

    @property
    def cell_type(self) -> ReferenceCellType:
        """Cell type of the element."""
        return ReferenceCellType(_lib.ciarlet_element_cell_type(self._rs_element))

    def entity_dofs(self, entity_dim: int, entity_index: int) -> list[int]:
        """Get the DOFs associated with an entity."""
        dofs = np.empty(
            _lib.ciarlet_element_entity_dofs_size(self._rs_element, entity_dim, entity_index),
            dtype=np.uintp,
        )
        _lib.ciarlet_element_entity_dofs(
            self._rs_element, entity_dim, entity_index, _ffi.cast("uintptr_t*", dofs.ctypes.data)
        )
        return [int(i) for i in dofs]

    def entity_closure_dofs(self, entity_dim: int, entity_index: int) -> list[int]:
        """Get the DOFs associated with the closure of an entity."""
        dofs = np.empty(
            _lib.ciarlet_element_entity_closure_dofs_size(
                self._rs_element, entity_dim, entity_index
            ),
            dtype=np.uintp,
        )
        _lib.ciarlet_element_entity_closure_dofs(
            self._rs_element, entity_dim, entity_index, _ffi.cast("uintptr_t*", dofs.ctypes.data)
        )
        return [int(i) for i in dofs]

    def interpolation_points(self) -> list[list[npt.NDArray]]:
        """Interpolation points."""
        points = []
        tdim = dim(self.cell_type)
        for d, n in enumerate(entity_counts(self.cell_type)):
            points_d = []
            for i in range(n):
                shape = (_lib.ciarlet_element_interpolation_npoints(self._rs_element, d, i), tdim)
                points_di = np.empty(shape, dtype=self.dtype(0).real.dtype)
                _lib.ciarlet_element_interpolation_points(
                    self._rs_element, d, i, _ffi.cast("void*", points_di.ctypes.data)
                )
                points_d.append(points_di)
            points.append(points_d)
        return points

    def interpolation_weights(self) -> list[list[npt.NDArray]]:
        """Interpolation weights."""
        weights = []
        for d, n in enumerate(entity_counts(self.cell_type)):
            weights_d = []
            for i in range(n):
                shape = (
                    _lib.ciarlet_element_interpolation_ndofs(self._rs_element, d, i),
                    self.value_size,
                    _lib.ciarlet_element_interpolation_npoints(self._rs_element, d, i),
                )
                weights_di = np.empty(shape, dtype=self.dtype)
                _lib.ciarlet_element_interpolation_weights(
                    self._rs_element, d, i, _ffi.cast("void*", weights_di.ctypes.data)
                )
                weights_d.append(weights_di)
            weights.append(weights_d)
        return weights

    def tabulate(self, points: npt.NDArray[np.floating], nderivs: int) -> npt.NDArray:
        """Tabulate the basis functions at a set of points."""
        if points.dtype != self.dtype(0).real.dtype:
            raise TypeError("points has incorrect type")
        shape = np.empty(4, dtype=np.uintp)
        _lib.ciarlet_element_tabulate_array_shape(
            self._rs_element, nderivs, points.shape[0], _ffi.cast("uintptr_t*", shape.ctypes.data)
        )
        data = np.empty(shape[::-1], dtype=self.dtype)
        _lib.ciarlet_element_tabulate(
            self._rs_element,
            _ffi.cast("void*", points.ctypes.data),
            points.shape[0],
            nderivs,
            _ffi.cast("void*", data.ctypes.data),
        )
        return data

    def physical_value_size(self, gdim: int) -> int:
        """The physical value size of the element on a cell."""
        return _lib.ciarlet_element_physical_value_size(self._rs_element, gdim)

    def physical_value_shape(self, gdim: int) -> tuple[int, ...]:
        """The physical value shape of the element on a cell."""
        shape = np.empty(_lib.ciarlet_element_physical_value_rank(self._rs_element), dtype=np.uintp)
        _lib.ciarlet_element_physical_value_shape(
            self._rs_element, _ffi.cast("uintptr_t*", shape.ctypes.data)
        )
        return tuple(int(i) for i in shape)

    def push_forward(
        self,
        reference_values: npt.NDArray[np.floating],
        nderivs: int,
        jacobians: npt.NDArray[np.floating],
        jacobian_determinants: npt.NDArray[np.floating],
        inverse_jacobians: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Push values forward to a physical cell."""
        if reference_values.dtype != self.dtype(0):
            raise TypeError("reference_values has incorrect type")
        if jacobians.dtype != self.dtype(0).real.dtype:
            raise TypeError("jacobians has incorrect type")
        if jacobian_determinants.dtype != self.dtype(0).real.dtype:
            raise TypeError("jacobian_determinants has incorrect type")
        if inverse_jacobians.dtype != self.dtype(0).real.dtype:
            raise TypeError("inverse_jacobians has incorrect type")

        gdim = jacobians.shape[1]
        npts = reference_values.shape[-2]
        nfuncs = reference_values.shape[-3]
        pvs = self.physical_value_size(gdim)

        shape = (pvs,) + reference_values.shape[1:]
        data = np.empty(shape, dtype=self.dtype)
        _lib.ciarlet_element_push_forward(
            self._rs_element,
            npts,
            nfuncs,
            gdim,
            _ffi.cast("void*", reference_values.ctypes.data),
            nderivs,
            _ffi.cast("void*", jacobians.ctypes.data),
            _ffi.cast("void*", jacobian_determinants.ctypes.data),
            _ffi.cast("void*", inverse_jacobians.ctypes.data),
            _ffi.cast("void*", data.ctypes.data),
        )
        return data

    def pull_back(
        self,
        physical_values: npt.NDArray[np.floating],
        nderivs: int,
        jacobians: npt.NDArray[np.floating],
        jacobian_determinants: npt.NDArray[np.floating],
        inverse_jacobians: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Push values back from a physical cell."""
        if physical_values.dtype != self.dtype(0):
            raise TypeError("physical_values has incorrect type")
        if jacobians.dtype != self.dtype(0).real.dtype:
            raise TypeError("jacobians has incorrect type")
        if jacobian_determinants.dtype != self.dtype(0).real.dtype:
            raise TypeError("jacobian_determinants has incorrect type")
        if inverse_jacobians.dtype != self.dtype(0).real.dtype:
            raise TypeError("inverse_jacobians has incorrect type")

        gdim = jacobians.shape[1]
        npts = physical_values.shape[-2]
        nfuncs = physical_values.shape[-3]
        vs = self.value_size

        shape = (vs,) + physical_values.shape[1:]
        data = np.empty(shape, dtype=self.dtype)
        _lib.ciarlet_element_pull_back(
            self._rs_element,
            npts,
            nfuncs,
            gdim,
            _ffi.cast("void*", physical_values.ctypes.data),
            nderivs,
            _ffi.cast("void*", jacobians.ctypes.data),
            _ffi.cast("void*", jacobian_determinants.ctypes.data),
            _ffi.cast("void*", inverse_jacobians.ctypes.data),
            _ffi.cast("void*", data.ctypes.data),
        )
        return data


class ElementFamily(object):
    """Ciarlet element."""

    def __init__(
        self,
        family: Family,
        degree: int,
        continuity: Continuity,
        rs_family: _CDataBase,
        owned: bool = True,
    ):
        """Initialise."""
        self._rs_family = rs_family
        self._owned = owned
        self._family = family
        self._degree = degree
        self._continuity = continuity

    def __del__(self):
        """Delete object."""
        if self._owned:
            _lib.element_family_t_free(self._rs_family)

    @property
    def dtype(self) -> typing.Type[np.floating]:
        """Data type."""
        return _dtypes[_lib.element_family_dtype(self._rs_family)]

    @property
    def family(self) -> Family:
        """The family."""
        return self._family

    @property
    def degree(self) -> int:
        """The degree."""
        return self._degree

    @property
    def continuity(self) -> Continuity:
        """Continuity."""
        return self._continuity

    def element(self, cell: ReferenceCellType) -> CiarletElement:
        """Create an element."""
        return CiarletElement(_lib.element_family_create_element(self._rs_family, cell.value))


def create_family(
    family: Family,
    degree: int,
    continuity: Continuity = Continuity.Standard,
    dtype: typing.Type[np.floating] = np.float64,
) -> ElementFamily:
    """Create a new element family."""
    rust_type = _rtypes[dtype]
    if family == Family.Lagrange:
        return ElementFamily(
            family,
            degree,
            continuity,
            _lib.create_lagrange_family(degree, continuity.value, rust_type),
        )
    elif family == Family.RaviartThomas:
        return ElementFamily(
            family,
            degree,
            continuity,
            _lib.create_raviart_thomas_family(degree, continuity.value, rust_type),
        )
    elif family == Family.NedelecFirstKind:
        return ElementFamily(
            family,
            degree,
            continuity,
            _lib.create_nedelec_family(degree, continuity.value, rust_type),
        )
    else:
        raise ValueError(f"Unsupported family: {family}")
