import pytest
import numpy as np
from ndelement.reference_cell import (
    dim,
    ReferenceCellType,
    midpoint,
    vertices,
    entity_counts,
    edges,
    faces,
    volumes,
    is_simplex,
    entity_types,
    connectivity,
)

cells = [
    ReferenceCellType.Interval,
    ReferenceCellType.Triangle,
    ReferenceCellType.Quadrilateral,
    ReferenceCellType.Tetrahedron,
    ReferenceCellType.Hexahedron,
    ReferenceCellType.Prism,
    ReferenceCellType.Pyramid,
]


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


@pytest.mark.parametrize("cell", cells)
def test_vertices_and_midpoint(cell):
    v = vertices(cell)
    m = midpoint(cell)

    assert np.allclose(sum(i for i in v) / v.shape[0], m)


@pytest.mark.parametrize("cell", cells)
def test_entity_counts(cell):
    counts = entity_counts(cell)

    assert len(vertices(cell)) == counts[0]
    assert len(edges(cell)) == counts[1]
    assert len(faces(cell)) == counts[2]
    assert len(volumes(cell)) == counts[3]

    for i, j in zip(counts, entity_types(cell)):
        assert len(j) == i


@pytest.mark.parametrize("cell", cells)
def test_is_simplex(cell):
    is_simplex(cell) == cell in [
        ReferenceCellType.Point,
        ReferenceCellType.Interval,
        ReferenceCellType.Triangle,
        ReferenceCellType.Tetrahedron,
    ]


@pytest.mark.parametrize("cell", cells)
def test_connectivity(cell):
    nvertices = entity_counts(cell)[0]
    entities = [[[i] for i in range(nvertices)], edges(cell), faces(cell), volumes(cell)]

    for d, (c_d, e_d) in enumerate(zip(connectivity(cell), entities)):
        for i, (c_di, e_di) in enumerate(zip(c_d, e_d)):
            for c_dij, e_j in zip(c_di[: d + 1], entities[: d + 1]):
                assert set([j for j, e in enumerate(e_j) if all([k in e_di for k in e])]) == set(
                    c_dij
                )
            for c_dij, e_j in zip(c_di[d + 1 :], entities[d + 1 :]):
                assert set([j for j, e in enumerate(e_j) if all([k in e for k in e_di])]) == set(
                    c_dij
                )
