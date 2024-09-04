"""Types."""
from ._ndelementrs import lib as _lib
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
    """The topological dimension of a reference cell."""
    return _lib.dim(cell.value)

