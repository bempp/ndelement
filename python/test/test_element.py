import pytest
import numpy as np
from ndelement.reference_cell import ReferenceCellType
from ndelement.ciarlet import Continuity, Family, create_family

cells = [
    ReferenceCellType.Interval,
    ReferenceCellType.Triangle,
    ReferenceCellType.Quadrilateral,
    ReferenceCellType.Tetrahedron,
    ReferenceCellType.Hexahedron,
]
dtypes = [
    np.float32,
    np.float64,
    np.complex64,
    np.complex128,
]


@pytest.mark.parametrize("cell", cells)
@pytest.mark.parametrize("degree", range(1, 4))
def test_value_size(cell, degree):
    family = create_family(Family.Lagrange, degree)
    element = family.element(cell)

    assert element.value_size == 1


@pytest.mark.parametrize(
    "dt0,dt1",
    [
        (np.float32, np.float64),
        (np.float64, np.float32),
        (np.complex64, np.float64),
        (np.complex128, np.float32),
    ]
    + [(dt0, dt1) for dt0 in dtypes for dt1 in [np.complex64, np.complex128]],
)
def test_incompatible_types(dt0, dt1):
    family = create_family(Family.Lagrange, 2, dtype=dt0)
    element = family.element(ReferenceCellType.Triangle)
    points = np.array([[0.0, 0.0], [0.2, 0.1], [0.8, 0.05]], dtype=dt1)

    with pytest.raises(TypeError):
        element.tabulate(points, 1)


@pytest.mark.parametrize("dtype", dtypes)
def test_lagrange_2_triangle_tabulate(dtype):
    family = create_family(Family.Lagrange, 2, dtype=dtype)
    element = family.element(ReferenceCellType.Triangle)

    points = np.array([[0.0, 0.0], [0.2, 0.1], [0.8, 0.05]], dtype=dtype(0).real.dtype)

    data = element.tabulate(points, 1)

    data2 = np.empty(data.shape)
    # Basis functions taken from DefElement
    # (https://defelement.org/elements/examples/triangle-lagrange-equispaced-2.html)
    for i, function in enumerate(
        [
            lambda x, y: (x + y - 1) * (2 * x + 2 * y - 1),
            lambda x, y: x * (2 * x - 1),
            lambda x, y: y * (2 * y - 1),
            lambda x, y: 4 * x * y,
            lambda x, y: 4 * (1 - x - y) * y,
            lambda x, y: 4 * x * (1 - x - y),
        ]
    ):
        for j, p in enumerate(points):
            data2[0, i, j, 0] = function(*p)
    # x-derivatives
    for i, function in enumerate(
        [
            lambda x, y: 4 * x + 4 * y - 3,
            lambda x, y: 4 * x - 1,
            lambda x, y: 0,
            lambda x, y: 4 * y,
            lambda x, y: -4 * y,
            lambda x, y: 4 - 8 * x - 4 * y,
        ]
    ):
        for j, p in enumerate(points):
            data2[0, i, j, 1] = function(*p)
    # y-derivatives
    for i, function in enumerate(
        [
            lambda x, y: 4 * x + 4 * y - 3,
            lambda x, y: 0,
            lambda x, y: 4 * y - 1,
            lambda x, y: 4 * x,
            lambda x, y: 4 - 4 * x - 8 * y,
            lambda x, y: -4 * x,
        ]
    ):
        for j, p in enumerate(points):
            data2[0, i, j, 2] = function(*p)

    assert np.allclose(data, data2, atol=np.finfo(dtype).eps * 10)


@pytest.mark.parametrize("continuity", [Continuity.Standard, Continuity.Discontinuous])
def test_lagrange_1_triangle(continuity):
    family = create_family(Family.Lagrange, 1, continuity=continuity)
    element = family.element(ReferenceCellType.Triangle)

    assert element.value_size == 1
    assert element.value_shape == ()
    assert element.degree == 1
    assert element.embedded_superdegree == 1
    assert element.dim == 3
    assert element.continuity == continuity
    assert element.cell_type == ReferenceCellType.Triangle

    if continuity == Continuity.Standard:
        assert element.entity_dofs(0, 0) == [0]
        assert element.entity_dofs(0, 1) == [1]
        assert element.entity_dofs(0, 2) == [2]
        assert element.entity_dofs(1, 0) == []
        assert element.entity_dofs(1, 1) == []
        assert element.entity_dofs(1, 2) == []
        assert element.entity_dofs(2, 0) == []

        assert element.entity_closure_dofs(0, 0) == [0]
        assert element.entity_closure_dofs(0, 1) == [1]
        assert element.entity_closure_dofs(0, 2) == [2]
        assert element.entity_closure_dofs(1, 0) == [1, 2]
        assert element.entity_closure_dofs(1, 1) == [0, 2]
        assert element.entity_closure_dofs(1, 2) == [0, 1]
        assert element.entity_closure_dofs(2, 0) == [0, 1, 2]

        ip = element.interpolation_points()
        assert np.allclose(ip[0][0], np.array([[0.0, 0.0]]))
        assert np.allclose(ip[0][1], np.array([[1.0, 0.0]]))
        assert np.allclose(ip[0][2], np.array([[0.0, 1.0]]))
        assert ip[1][0].shape == (0, 2)
        assert ip[1][1].shape == (0, 2)
        assert ip[1][2].shape == (0, 2)
        assert ip[2][0].shape == (0, 2)

        iw = element.interpolation_weights()
        assert np.allclose(iw[0][0], np.array([[[1.0]]]))
        assert np.allclose(iw[0][1], np.array([[[1.0]]]))
        assert np.allclose(iw[0][2], np.array([[[1.0]]]))
        assert iw[1][0].shape == (0, 1, 0)
        assert iw[1][1].shape == (0, 1, 0)
        assert iw[1][2].shape == (0, 1, 0)
        assert iw[2][0].shape == (0, 1, 0)
    else:
        for i in range(3):
            assert element.entity_dofs(0, i) == []
            assert element.entity_dofs(1, i) == []
            assert element.entity_closure_dofs(0, i) == []
            assert element.entity_closure_dofs(1, i) == []
        assert element.entity_dofs(2, 0) == [0, 1, 2]
        assert element.entity_closure_dofs(2, 0) == [0, 1, 2]

        ip = element.interpolation_points()
        assert ip[0][0].shape == (0, 2)
        assert ip[0][1].shape == (0, 2)
        assert ip[0][2].shape == (0, 2)
        assert ip[1][0].shape == (0, 2)
        assert ip[1][1].shape == (0, 2)
        assert ip[1][2].shape == (0, 2)
        assert np.allclose(ip[2][0], np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))

        iw = element.interpolation_weights()
        assert iw[0][0].shape == (0, 1, 0)
        assert iw[0][1].shape == (0, 1, 0)
        assert iw[0][2].shape == (0, 1, 0)
        assert iw[1][0].shape == (0, 1, 0)
        assert iw[1][1].shape == (0, 1, 0)
        assert iw[1][2].shape == (0, 1, 0)
        assert np.allclose(
            iw[2][0], np.array([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0]]])
        )


@pytest.mark.parametrize("continuity", [Continuity.Standard, Continuity.Discontinuous])
def test_raviart_thomas_1_triangle(continuity):
    family = create_family(Family.RaviartThomas, 1, continuity=continuity)
    element = family.element(ReferenceCellType.Triangle)

    assert element.value_size == 2
    assert element.value_shape == (2,)
    assert element.degree == 1
    assert element.embedded_superdegree == 1
    assert element.dim == 3
    assert element.continuity == continuity
    assert element.cell_type == ReferenceCellType.Triangle

    if continuity == Continuity.Standard:
        assert element.entity_dofs(0, 0) == []
        assert element.entity_dofs(0, 1) == []
        assert element.entity_dofs(0, 2) == []
        assert element.entity_dofs(1, 0) == [0]
        assert element.entity_dofs(1, 1) == [1]
        assert element.entity_dofs(1, 2) == [2]
        assert element.entity_dofs(2, 0) == []

        assert element.entity_closure_dofs(0, 0) == []
        assert element.entity_closure_dofs(0, 1) == []
        assert element.entity_closure_dofs(0, 2) == []
        assert element.entity_closure_dofs(1, 0) == [0]
        assert element.entity_closure_dofs(1, 1) == [1]
        assert element.entity_closure_dofs(1, 2) == [2]
        assert element.entity_closure_dofs(2, 0) == [0, 1, 2]

        ip = element.interpolation_points()
        assert ip[0][0].shape == (0, 2)
        assert ip[0][1].shape == (0, 2)
        assert ip[0][2].shape == (0, 2)
        assert ip[2][0].shape == (0, 2)
    else:
        for i in range(3):
            assert element.entity_dofs(0, i) == []
            assert element.entity_dofs(1, i) == []
            assert element.entity_closure_dofs(0, i) == []
            assert element.entity_closure_dofs(1, i) == []
        assert element.entity_dofs(2, 0) == [0, 1, 2]
        assert element.entity_closure_dofs(2, 0) == [0, 1, 2]


@pytest.mark.parametrize("continuity", [Continuity.Standard, Continuity.Discontinuous])
def test_nedelec_1_triangle(continuity):
    family = create_family(Family.NedelecFirstKind, 1, continuity=continuity)
    element = family.element(ReferenceCellType.Triangle)

    assert element.value_size == 2
    assert element.value_shape == (2,)
    assert element.degree == 1
    assert element.embedded_superdegree == 1
    assert element.dim == 3
    assert element.continuity == continuity
    assert element.cell_type == ReferenceCellType.Triangle

    if continuity == Continuity.Standard:
        assert element.entity_dofs(0, 0) == []
        assert element.entity_dofs(0, 1) == []
        assert element.entity_dofs(0, 2) == []
        assert element.entity_dofs(1, 0) == [0]
        assert element.entity_dofs(1, 1) == [1]
        assert element.entity_dofs(1, 2) == [2]
        assert element.entity_dofs(2, 0) == []

        assert element.entity_closure_dofs(0, 0) == []
        assert element.entity_closure_dofs(0, 1) == []
        assert element.entity_closure_dofs(0, 2) == []
        assert element.entity_closure_dofs(1, 0) == [0]
        assert element.entity_closure_dofs(1, 1) == [1]
        assert element.entity_closure_dofs(1, 2) == [2]
        assert element.entity_closure_dofs(2, 0) == [0, 1, 2]

        ip = element.interpolation_points()
        assert ip[0][0].shape == (0, 2)
        assert ip[0][1].shape == (0, 2)
        assert ip[0][2].shape == (0, 2)
        assert ip[2][0].shape == (0, 2)
    else:
        for i in range(3):
            assert element.entity_dofs(0, i) == []
            assert element.entity_dofs(1, i) == []
            assert element.entity_closure_dofs(0, i) == []
            assert element.entity_closure_dofs(1, i) == []
        assert element.entity_dofs(2, 0) == [0, 1, 2]
        assert element.entity_closure_dofs(2, 0) == [0, 1, 2]


@pytest.mark.parametrize(
    ("ftype", "reference_values", "physical_values"),
    [
        (
            Family.Lagrange,
            np.array([[[[1.0], [0.5]], [[0.0], [2.0]]]]),
            np.array([[[[1.0], [0.5]], [[0.0], [2.0]]]]),
        ),
        (
            Family.NedelecFirstKind,
            np.array([[[[1.0], [0.0]], [[0.5], [2.0]]], [[[0.0], [1.0]], [[1.5], [2.0]]]]),
            np.array(
                [[[[1.0], [0.0]], [[0.5], [1.0]]], [[[-1.0], [1.0 / 3.0]], [[1.0], [2.0 / 3.0]]]]
            ),
        ),
        (
            Family.RaviartThomas,
            np.array([[[[1.0], [0.0]], [[0.5], [2.0]]], [[[0.0], [1.0]], [[1.5], [2.0]]]]),
            np.array([[[[1.0], [0.0]], [[2.0], [2.0 / 3.0]]], [[[0.0], [0.5]], [[1.5], [1.0]]]]),
        ),
    ],
)
def test_push_forward_pull_back(ftype, reference_values, physical_values):
    family = create_family(ftype, 1, continuity=Continuity.Standard)
    element = family.element(ReferenceCellType.Triangle)
    j = np.array([[[1.0, 2.0], [0.0, 0.0]], [[1.0, 0.0], [1.0, 3.0]]])
    jdet = np.array([1.0, 6.0])
    jinv = np.array(
        [
            [
                [1.0, 0.5],
                [0.0, 0.0],
            ],
            [
                [-1.0, 0.0],
                [1.0, 1.0 / 3.0],
            ],
        ]
    )

    assert np.allclose(physical_values, element.push_forward(reference_values, 0, j, jdet, jinv))
    assert np.allclose(reference_values, element.pull_back(physical_values, 0, j, jdet, jinv))
