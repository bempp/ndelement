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

    Standard = 0
    Discontinuous = 1


class Family(Enum):
    """Element family."""

    Lagrange = 0
    RaviartThomas = 1


class MapType(Enum):
    """Map type."""

    Identity = 0
    CovariantPiola = 1
    ContravariantPiola = 2
    L2Piola = 3


_dtypes = {
    0: np.float32,
    1: np.float64,
}
_ctypes = {
    np.float32: "float",
    np.float64: "double",
}


class CiarletElement(object):
    """Ciarlet element."""

    def __init__(self, rs_element: _CDataBase, owned: bool = True):
        """Initialise."""
        self._rs_element = rs_element
        self._owned = owned

    def __del__(self):
        """Delete object."""
        if self._owned:
            _lib.ciarlet_free_element(self._rs_element)

    @property
    def dtype(self):
        """Data type."""
        return _dtypes[_lib.ciarlet_element_dtype(self._rs_element)]

    @property
    def _ctype(self):
        """C data type."""
        return _ctypes[self.dtype]

    @property
    def value_size(self) -> int:
        """Value size of the element."""
        return _lib.ciarlet_value_size(self._rs_element)

    @property
    def value_shape(self) -> typing.Tuple[int, ...]:
        """Value size of the element."""
        shape = np.empty(_lib.ciarlet_value_rank(self._rs_element), dtype=np.uintp)
        _lib.ciarlet_value_shape(self._rs_element, _ffi.cast("uintptr_t*", shape.ctypes.data))
        return tuple(int(i) for i in shape)

    @property
    def degree(self) -> int:
        """Degree of the element."""
        return _lib.ciarlet_degree(self._rs_element)

    @property
    def embedded_superdegree(self) -> int:
        """Embedded superdegree of the element."""
        return _lib.ciarlet_embedded_superdegree(self._rs_element)

    @property
    def dim(self) -> int:
        """Dimension (number of basis functions) of the element."""
        return _lib.ciarlet_dim(self._rs_element)

    @property
    def continuity(self) -> Continuity:
        """Continuity of the element."""
        return Continuity(_lib.ciarlet_continuity(self._rs_element))

    @property
    def map_type(self) -> MapType:
        """Pullback map type of the element."""
        return MapType(_lib.ciarlet_map_type(self._rs_element))

    @property
    def cell_type(self) -> ReferenceCellType:
        """Cell type of the element."""
        return ReferenceCellType(_lib.ciarlet_cell_type(self._rs_element))

    def entity_dofs(self, entity_dim: int, entity_index: int) -> typing.List[int]:
        """Get the DOFs associated with an entity."""
        dofs = np.empty(
            _lib.ciarlet_entity_dofs_size(self._rs_element, entity_dim, entity_index),
            dtype=np.uintp,
        )
        _lib.ciarlet_entity_dofs(
            self._rs_element, entity_dim, entity_index, _ffi.cast("uintptr_t*", dofs.ctypes.data)
        )
        return [int(i) for i in dofs]

    def entity_closure_dofs(self, entity_dim: int, entity_index: int) -> typing.List[int]:
        """Get the DOFs associated with the closure of an entity."""
        dofs = np.empty(
            _lib.ciarlet_entity_closure_dofs_size(self._rs_element, entity_dim, entity_index),
            dtype=np.uintp,
        )
        _lib.ciarlet_entity_closure_dofs(
            self._rs_element, entity_dim, entity_index, _ffi.cast("uintptr_t*", dofs.ctypes.data)
        )
        return [int(i) for i in dofs]

    def interpolation_points(self) -> typing.List[typing.List[npt.NDArray]]:
        """Interpolation points."""
        points = []
        tdim = dim(self.cell_type)
        for d, n in enumerate(entity_counts(self.cell_type)):
            points_d = []
            for i in range(n):
                shape = (_lib.ciarlet_interpolation_npoints(self._rs_element, d, i), tdim)
                points_di = np.empty(shape, dtype=self.dtype)
                _lib.ciarlet_interpolation_points(
                    self._rs_element, d, i, _ffi.cast("void*", points_di.ctypes.data)
                )
                points_d.append(points_di)
            points.append(points_d)
        return points

    def interpolation_weights(self) -> typing.List[typing.List[npt.NDArray]]:
        """Interpolation weights."""
        weights = []
        for d, n in enumerate(entity_counts(self.cell_type)):
            weights_d = []
            for i in range(n):
                shape = (
                    _lib.ciarlet_interpolation_ndofs(self._rs_element, d, i),
                    self.value_size,
                    _lib.ciarlet_interpolation_npoints(self._rs_element, d, i),
                )
                weights_di = np.empty(shape, dtype=self.dtype)
                _lib.ciarlet_interpolation_weights(
                    self._rs_element, d, i, _ffi.cast("void*", weights_di.ctypes.data)
                )
                weights_d.append(weights_di)
            weights.append(weights_d)
        return weights

    def tabulate(self, points: npt.NDArray[np.floating], nderivs: int) -> npt.NDArray:
        """Tabulate the basis functions at a set of points."""
        shape = np.empty(4, dtype=np.uintp)
        _lib.ciarlet_tabulate_array_shape(
            self._rs_element, nderivs, points.shape[0], _ffi.cast("uintptr_t*", shape.ctypes.data)
        )
        data = np.empty(shape[::-1], dtype=self.dtype)
        _lib.ciarlet_tabulate(
            self._rs_element,
            _ffi.cast("void*", points.ctypes.data),
            points.shape[0],
            nderivs,
            _ffi.cast("void*", data.ctypes.data),
        )
        return data


class ElementFamily(object):
    """Ciarlet element."""

    def __init__(self, rs_family: _CDataBase, owned: bool = True):
        """Initialise."""
        self._rs_family = rs_family
        self._owned = owned

    def __del__(self):
        """Delete object."""
        if self._owned:
            _lib.ciarlet_free_family(self._rs_family)

    def element(self, cell: ReferenceCellType) -> CiarletElement:
        return CiarletElement(_lib.element_family_element(self._rs_family, cell.value))


def create_family(
    family: Family,
    degree: int,
    continuity: Continuity = Continuity.Standard,
    dtype: typing.Type[np.floating] = np.float64,
) -> ElementFamily:
    """Create a new element family."""
    if family == Family.Lagrange:
        if dtype == np.float64:
            return ElementFamily(_lib.lagrange_element_family_new_f64(degree, continuity.value))
        elif dtype == np.float32:
            return ElementFamily(_lib.lagrange_element_family_new_f64(degree, continuity.value))
        else:
            raise TypeError(f"Unsupported dtype: {dtype}")
    elif family == Family.RaviartThomas:
        if dtype == np.float64:
            return ElementFamily(
                _lib.raviart_thomas_element_family_new_f64(degree, continuity.value)
            )
        elif dtype == np.float32:
            return ElementFamily(
                _lib.raviart_thomas_element_family_new_f64(degree, continuity.value)
            )
        else:
            raise TypeError(f"Unsupported dtype: {dtype}")
    else:
        raise ValueError(f"Unsupported family: {family}")
