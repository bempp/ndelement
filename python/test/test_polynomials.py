import pytest
import numpy as np
from ndelement.reference_cell import ReferenceCellType
from ndelement.polynomials import tabulate_legendre_polynomials
from ndelement.quadrature import make_gauss_jacobi_quadrature

cells = [
    ReferenceCellType.Interval,
    ReferenceCellType.Triangle,
    ReferenceCellType.Quadrilateral,
    ReferenceCellType.Tetrahedron,
    ReferenceCellType.Hexahedron,
]


@pytest.mark.parametrize("cell", cells)
@pytest.mark.parametrize("degree", range(1, 5))
def test_orthogonal(cell, degree):
    pts, wts = make_gauss_jacobi_quadrature(cell, degree * 2)
    poly = tabulate_legendre_polynomials(cell, pts, degree, 0)

    for i in range(poly.shape[1]):
        for j in range(poly.shape[1]):
            integral = sum(poly[:, i, 0] * poly[:, j, 0] * wts)
            if i == j:
                assert np.isclose(integral, 1.0)
            else:
                assert np.isclose(integral, 0.0)
            pass
