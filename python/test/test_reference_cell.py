from ndelement.reference_cell import dim, ReferenceCellType


def test_dim():
    assert dim(ReferenceCellType.Point) == 0
    assert dim(ReferenceCellType.Interval) == 1
    assert dim(ReferenceCellType.Triangle) == 2
    assert dim(ReferenceCellType.Quadrilateral) == 2
    assert dim(ReferenceCellType.Tetrahedron) == 3
    assert dim(ReferenceCellType.Hexahedron) == 3
