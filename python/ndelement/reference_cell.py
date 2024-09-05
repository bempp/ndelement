"""Reference cell information."""
import typing
import ctypes
import numpy as np
import numpy.typing as npt
from ._ndelementrs import lib as _lib, ffi as _ffi
from enum import Enum


class ReferenceCellType(Enum):
    """Reference cell type."""
    Point = 0
    Interval = 1
    Triangle = 2
    Quadrilateral = 3
    Tetrahedron = 4
    Hexahedron = 5
    Prism = 6
    Pyramid = 7


def dim(cell: ReferenceCellType) -> int:
    """Get the topological dimension of a reference cell."""
    return _lib.dim(cell.value)


def is_simplex(cell: ReferenceCellType) -> bool:
    """Check if a reference cell is a simplex."""
    return _lib.is_simplex(cell.value)


def vertices(cell: ReferenceCellType, dtype: str = np.float64) -> npt.NDArray:
    """Get the vertices of a reference cell."""
    vertices = np.empty((entity_counts(cell)[0], dim(cell)), dtype=dtype)
    if dtype == np.float64:
        _lib.vertices_f64(cell.value, _ffi.cast("double*", vertices.ctypes.data))
    elif dtype == np.float32:
        _lib.vertices_f32(cell.value, _ffi.cast("float*", vertices.ctypes.data))
    else:
        raise TypeError(f"Unsupported dtype: {dtype}")
    return vertices


def midpoint(cell: ReferenceCellType, dtype: str = np.float64) -> npt.NDArray:
    """Get the midpoint of a reference cell."""
    point = np.empty(dim(cell), dtype=dtype)
    if dtype == np.float64:
        _lib.midpoint_f64(cell.value, _ffi.cast("double*", point.ctypes.data))
    elif dtype == np.float32:
        _lib.midpoint_f32(cell.value, _ffi.cast("float*", point.ctypes.data))
    else:
        raise TypeError(f"Unsupported dtype: {dtype}")
    return point


def edges(cell: ReferenceCellType) -> typing.List[npt.NDArray[int]]:
    """Get the edges of a reference cell."""
    edges = []
    e = np.empty(2 * entity_counts(cell)[1], dtype=int)
    _lib.faces(cell.value, _ffi.cast("uintptr_t* ", e.ctypes.data))
    for i in range(entity_counts(cell)[1]):
        edges.append(e[2*i:2*i+2])
    return edges


def faces(cell: ReferenceCellType) -> typing.List[npt.NDArray[int]]:
    """Get the faces of a reference cell."""
    faces = []
    flen = 0
    for t in entity_types(cell)[2]:
        flen += entity_counts(t)[0]
    f = np.empty(flen, dtype=int)
    _lib.faces(cell.value, _ffi.cast("uintptr_t* ", f.ctypes.data))
    start = 0
    for t in entity_types(cell)[2]:
        n = entity_counts(t)[0]
        faces.append(f[start:start+n])
        start += n
    return faces


def volumes(cell: ReferenceCellType) -> typing.List[npt.NDArray[int]]:
    """Get the volumes of a reference cell."""
    volumes = []
    vlen = 0
    for t in entity_types(cell)[3]:
        vlen += entity_counts(t)[0]
    v = np.empty(vlen, dtype=int)
    _lib.volumes(cell.value, _ffi.cast("uintptr_t* ", v.ctypes.data))
    start = 0
    for t in entity_types(cell)[3]:
        n = entity_counts(t)[0]
        volumes.append(v[start:start+n])
        start += n
    return volumes


def entity_types(cell: ReferenceCellType) -> typing.List[typing.List[ReferenceCellType]]:
    """Get the types of the sub-entities of a reference cell."""
    # TODO: should int be uintptr_t?
    t = np.empty(sum(entity_counts(cell)), dtype=int)
    _lib.entity_types(cell.value, _ffi.cast("uintptr_t* ", t.ctypes.data))
    types = []
    start = 0
    for n in entity_counts(cell):
        types.append([ReferenceCellType(i) for i in t[start:start+n]])
        start += n
    return types


def entity_counts(cell: ReferenceCellType) -> npt.NDArray[int]:
    """Get the number of the sub-entities of each dimension for a reference cell."""
    counts = np.empty(4, dtype=int)
    _lib.entity_counts(cell.value, _ffi.cast("uintptr_t* ", counts.ctypes.data))
    return counts


# TODO: connectivity
