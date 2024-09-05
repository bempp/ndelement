import pytest
import numpy as np
from ndelement.reference_cell import dim, ReferenceCellType, midpoint, vertices


def test_dim():
    assert dim(ReferenceCellType.Point) == 0
    assert dim(ReferenceCellType.Interval) == 1
    assert dim(ReferenceCellType.Triangle) == 2
    assert dim(ReferenceCellType.Quadrilateral) == 2
    assert dim(ReferenceCellType.Tetrahedron) == 3
    assert dim(ReferenceCellType.Hexahedron) == 3


def test_midpoint():
    assert np.allclose(midpoint(ReferenceCellType.Interval), [0.5])
    assert np.allclose(midpoint(ReferenceCellType.Triangle), [1 / 3, 1 / 3])
    assert np.allclose(midpoint(ReferenceCellType.Quadrilateral), [1 / 2, 1 / 2])
    assert np.allclose(midpoint(ReferenceCellType.Tetrahedron), [1 / 4, 1 / 4, 1 / 4])
    assert np.allclose(midpoint(ReferenceCellType.Hexahedron), [1 / 2, 1 / 2, 1 / 2])


@pytest.mark.parametrize("cell", [
    ReferenceCellType.Interval,
    ReferenceCellType.Triangle,
    ReferenceCellType.Quadrilateral,
    ReferenceCellType.Tetrahedron,
    ReferenceCellType.Hexahedron,
])
def test_vertices_and_midpoint(cell):
    v = vertices(cell)
    m = midpoint(cell)

    assert np.allclose(sum(i for i in v) / v.shape[0], m)
