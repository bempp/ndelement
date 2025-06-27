//! Nedelec elements

use super::CiarletElement;
use crate::map::CovariantPiolaMap;
use crate::math::orthogonalise_3;
use crate::polynomials::{legendre_shape, polynomial_count, tabulate_legendre_polynomials};
use crate::quadrature::gauss_jacobi_rule;
use crate::reference_cell;
use crate::traits::ElementFamily;
use crate::types::{Continuity, ReferenceCellType};
use itertools::izip;
use rlst::{
    rlst_array_from_slice2, rlst_dynamic_array2, rlst_dynamic_array3, MatrixInverse,
    RandomAccessMut, RlstScalar, Shape,
};
use std::marker::PhantomData;

fn create_simplex<TReal: RlstScalar<Real = TReal>, T: RlstScalar<Real = TReal> + MatrixInverse>(
    cell_type: ReferenceCellType,
    degree: usize,
    continuity: Continuity,
) -> CiarletElement<T, CovariantPiolaMap> {
    if cell_type != ReferenceCellType::Triangle && cell_type != ReferenceCellType::Tetrahedron {
        panic!("Invalid cell: {cell_type:?}");
    }

    if degree < 1 {
        panic!("Degree must be at least 1");
    }

    let tdim = reference_cell::dim(cell_type);
    let cell_q = gauss_jacobi_rule(cell_type, 2 * degree).unwrap();
    let pts_t = cell_q
        .points
        .iter()
        .map(|i| TReal::from(*i).unwrap())
        .collect::<Vec<_>>();
    let pts = rlst_array_from_slice2!(&pts_t, [tdim, cell_q.npoints]);

    let mut phi = rlst_dynamic_array3![T, legendre_shape(cell_type, &pts, degree, 0)];
    tabulate_legendre_polynomials(cell_type, &pts, degree, 0, &mut phi);

    let pdim = phi.shape()[1];

    let pdim_facet_minus1 = polynomial_count(
        if tdim == 3 {
            ReferenceCellType::Triangle
        } else {
            ReferenceCellType::Interval
        },
        degree - 1,
    );
    let pdim_edge_minus1 = polynomial_count(ReferenceCellType::Interval, degree - 1);
    let pdim_face_minus2 = if degree < 2 {
        0
    } else {
        polynomial_count(ReferenceCellType::Triangle, degree - 2)
    };
    let pdim_minus1 = polynomial_count(cell_type, degree - 1);
    let pdim_minus2 = if degree < 2 {
        0
    } else {
        polynomial_count(cell_type, degree - 2)
    };
    let pdim_minus3 = if degree < 3 {
        0
    } else {
        polynomial_count(cell_type, degree - 3)
    };

    let mut wcoeffs = rlst_dynamic_array3!(
        T,
        [
            pdim_minus1 * tdim + pdim_facet_minus1 * (tdim - 1) + pdim_edge_minus1 * (tdim - 2),
            tdim,
            pdim
        ]
    );

    // vector polynomials of degree <= n-1
    for i in 0..tdim {
        for j in 0..pdim_minus1 {
            *wcoeffs.get_mut([i * pdim_minus1 + j, i, j]).unwrap() = T::from(1.0).unwrap();
        }
    }

    if cell_type == ReferenceCellType::Triangle {
        // Additional Nedelec polynomials
        for i in 0..pdim_facet_minus1 {
            for j in pdim_minus1..pdim {
                let w0 = cell_q
                    .weights
                    .iter()
                    .enumerate()
                    .map(|(w_i, wt)| {
                        T::from(*wt).unwrap()
                            * phi[[0, pdim_minus2 + i, w_i]]
                            * T::from(pts[[0, w_i]]).unwrap()
                            * phi[[0, j, w_i]]
                    })
                    .sum::<T>();
                let w1 = cell_q
                    .weights
                    .iter()
                    .enumerate()
                    .map(|(w_i, wt)| {
                        T::from(*wt).unwrap()
                            * phi[[0, pdim_minus2 + i, w_i]]
                            * T::from(pts[[1, w_i]]).unwrap()
                            * phi[[0, j, w_i]]
                    })
                    .sum::<T>();
                wcoeffs[[2 * pdim_minus1 + i, 0, j]] = w1;
                wcoeffs[[2 * pdim_minus1 + i, 1, j]] = -w0;
            }
        }
    } else {
        // Additional Nedelec polynomials
        for i in 0..pdim_facet_minus1 {
            for j in pdim_minus1..pdim {
                let w0 = cell_q
                    .weights
                    .iter()
                    .enumerate()
                    .map(|(w_i, wt)| {
                        T::from(*wt).unwrap()
                            * phi[[0, pdim_minus2 + i, w_i]]
                            * T::from(pts[[0, w_i]]).unwrap()
                            * phi[[0, j, w_i]]
                    })
                    .sum::<T>();
                let w1 = cell_q
                    .weights
                    .iter()
                    .enumerate()
                    .map(|(w_i, wt)| {
                        T::from(*wt).unwrap()
                            * phi[[0, pdim_minus2 + i, w_i]]
                            * T::from(pts[[1, w_i]]).unwrap()
                            * phi[[0, j, w_i]]
                    })
                    .sum::<T>();
                let w2 = cell_q
                    .weights
                    .iter()
                    .enumerate()
                    .map(|(w_i, wt)| {
                        T::from(*wt).unwrap()
                            * phi[[0, pdim_minus2 + i, w_i]]
                            * T::from(pts[[2, w_i]]).unwrap()
                            * phi[[0, j, w_i]]
                    })
                    .sum::<T>();

                if i >= pdim_face_minus2 {
                    wcoeffs[[tdim * pdim_minus1 + i - pdim_face_minus2, 2, j]] = w1;
                    wcoeffs[[tdim * pdim_minus1 + i - pdim_face_minus2, 1, j]] = -w2;
                }
                wcoeffs[[
                    tdim * pdim_minus1 + i + pdim_facet_minus1 - pdim_face_minus2,
                    0,
                    j,
                ]] = w2;
                wcoeffs[[
                    tdim * pdim_minus1 + i + pdim_facet_minus1 * 2 - pdim_face_minus2,
                    0,
                    j,
                ]] = -w1;
                wcoeffs[[
                    tdim * pdim_minus1 + i + pdim_facet_minus1 - pdim_face_minus2,
                    2,
                    j,
                ]] = -w0;
                wcoeffs[[
                    tdim * pdim_minus1 + i + pdim_facet_minus1 * 2 - pdim_face_minus2,
                    1,
                    j,
                ]] = w0;
            }
        }
    };

    orthogonalise_3(&mut wcoeffs, pdim_minus1 * tdim);

    let mut x = [vec![], vec![], vec![], vec![]];
    let mut m = [vec![], vec![], vec![], vec![]];

    let entity_counts = reference_cell::entity_counts(cell_type);
    let vertices = reference_cell::vertices::<T::Real>(cell_type);

    for _ in 0..entity_counts[0] {
        x[0].push(rlst_dynamic_array2!(T::Real, [tdim, 0]));
        m[0].push(rlst_dynamic_array3!(T, [0, tdim, 0]));
    }

    // DOFs on edges
    let edge_q = gauss_jacobi_rule(ReferenceCellType::Interval, 2 * degree - 1).unwrap();
    let edge_pts_t = edge_q
        .points
        .iter()
        .map(|i| TReal::from(*i).unwrap())
        .collect::<Vec<_>>();
    let edge_pts = rlst_array_from_slice2!(&edge_pts_t, [1, edge_q.npoints]);

    let mut edge_phi = rlst_dynamic_array3![
        T,
        legendre_shape(ReferenceCellType::Interval, &edge_pts, degree - 1, 0)
    ];
    tabulate_legendre_polynomials(
        ReferenceCellType::Interval,
        &edge_pts,
        degree - 1,
        0,
        &mut edge_phi,
    );

    for edge in reference_cell::edges(cell_type) {
        let mut pts = rlst_dynamic_array2!(T::Real, [tdim, edge_q.npoints]);
        let mut mat = rlst_dynamic_array3!(T, [pdim_edge_minus1, tdim, edge_q.npoints]);

        for (w_i, (pt, wt)) in izip!(&edge_pts_t, &edge_q.weights).enumerate() {
            for i in 0..tdim {
                pts[[i, w_i]] =
                    vertices[edge[0]][i] + (vertices[edge[1]][i] - vertices[edge[0]][i]) * *pt;

                for j in 0..pdim_edge_minus1 {
                    mat[[j, i, w_i]] = T::from(*wt).unwrap()
                        * edge_phi[[0, j, w_i]]
                        * T::from(vertices[edge[1]][i] - vertices[edge[0]][i]).unwrap();
                }
            }
        }

        x[1].push(pts);
        m[1].push(mat);
    }

    // DOFs on faces
    if degree == 1 {
        for _ in 0..entity_counts[2] {
            x[2].push(rlst_dynamic_array2!(T::Real, [tdim, 0]));
            m[2].push(rlst_dynamic_array3!(T, [0, tdim, 0]))
        }
    } else {
        let face_q = gauss_jacobi_rule(ReferenceCellType::Triangle, 2 * degree - 2).unwrap();
        let face_pts_t = face_q
            .points
            .iter()
            .map(|i| TReal::from(*i).unwrap())
            .collect::<Vec<_>>();
        let face_pts = rlst_array_from_slice2!(&face_pts_t, [2, face_q.npoints]);

        let mut face_phi = rlst_dynamic_array3![
            T,
            legendre_shape(ReferenceCellType::Triangle, &face_pts, degree - 2, 0)
        ];
        tabulate_legendre_polynomials(
            ReferenceCellType::Triangle,
            &face_pts,
            degree - 2,
            0,
            &mut face_phi,
        );

        for face in reference_cell::faces(cell_type) {
            let mut pts = rlst_dynamic_array2!(T::Real, [tdim, face_q.npoints]);
            let mut mat = rlst_dynamic_array3!(T, [2 * pdim_face_minus2, tdim, face_q.npoints]);

            for (w_i, wt) in face_q.weights.iter().enumerate() {
                for i in 0..tdim {
                    pts[[i, w_i]] = vertices[face[0]][i]
                        + (vertices[face[1]][i] - vertices[face[0]][i]) * face_pts[[0, w_i]]
                        + (vertices[face[2]][i] - vertices[face[0]][i]) * face_pts[[1, w_i]];

                    for tangent in 0..2 {
                        for j in 0..pdim_face_minus2 {
                            mat[[tangent * pdim_face_minus2 + j, i, w_i]] = T::from(*wt).unwrap()
                                * face_phi[[0, j, w_i]]
                                * T::from(vertices[face[tangent + 1]][i] - vertices[face[0]][i])
                                    .unwrap();
                        }
                    }
                }
            }
            x[2].push(pts);
            m[2].push(mat);
        }
    }
    if tdim == 3 {
        if degree <= 2 {
            x[3].push(rlst_dynamic_array2!(T::Real, [tdim, 0]));
            m[3].push(rlst_dynamic_array3!(T, [0, tdim, 0]))
        } else {
            let internal_q = gauss_jacobi_rule(cell_type, 2 * degree - 3).unwrap();
            let mut pts = rlst_dynamic_array2!(T::Real, [tdim, internal_q.npoints]);
            for p in 0..internal_q.npoints {
                for d in 0..3 {
                    pts[[d, p]] = TReal::from(internal_q.points[3 * p + d]).unwrap()
                }
            }

            let mut internal_phi =
                rlst_dynamic_array3![T, legendre_shape(cell_type, &pts, degree - 3, 0)];
            tabulate_legendre_polynomials(cell_type, &pts, degree - 3, 0, &mut internal_phi);

            let mut mat = rlst_dynamic_array3!(T, [tdim * pdim_minus3, tdim, internal_q.npoints]);
            for (w_i, wt) in internal_q.weights.iter().enumerate() {
                for i in 0..tdim {
                    for j in 0..pdim_minus3 {
                        mat[[j + pdim_minus3 * i, i, w_i]] =
                            T::from(*wt).unwrap() * internal_phi[[0, j, w_i]];
                    }
                }
            }

            x[tdim].push(pts);
            m[tdim].push(mat);
        }
    }

    CiarletElement::create(
        "Nedelec (first kind)".to_string(),
        cell_type,
        degree,
        vec![tdim],
        wcoeffs,
        x,
        m,
        continuity,
        degree,
        CovariantPiolaMap {}, // TODO
    )
}

fn create_tp<TReal: RlstScalar<Real = TReal>, T: RlstScalar<Real = TReal> + MatrixInverse>(
    cell_type: ReferenceCellType,
    degree: usize,
    continuity: Continuity,
) -> CiarletElement<T, CovariantPiolaMap> {
    if cell_type != ReferenceCellType::Quadrilateral && cell_type != ReferenceCellType::Hexahedron {
        panic!("Invalid cell: {cell_type:?}");
    }

    if degree < 1 {
        panic!("Degree must be at least 1");
    }

    let tdim = reference_cell::dim(cell_type);
    let cell_q = gauss_jacobi_rule(cell_type, 2 * degree).unwrap();
    let pts_t = cell_q
        .points
        .iter()
        .map(|i| TReal::from(*i).unwrap())
        .collect::<Vec<_>>();
    let pts = rlst_array_from_slice2!(&pts_t, [tdim, cell_q.npoints]);

    let mut phi = rlst_dynamic_array3![T, legendre_shape(cell_type, &pts, degree, 0)];
    tabulate_legendre_polynomials(cell_type, &pts, degree, 0, &mut phi);

    let pdim = phi.shape()[1];

    let pdim_edge = polynomial_count(ReferenceCellType::Interval, degree);
    let pdim_edge_minus1 = polynomial_count(ReferenceCellType::Interval, degree - 1);
    let pdim_edge_minus2 = if degree < 2 {
        0
    } else {
        polynomial_count(ReferenceCellType::Interval, degree - 2)
    };

    let entity_counts = reference_cell::entity_counts(cell_type);

    let mut wcoeffs = rlst_dynamic_array3!(
        T,
        [
            entity_counts[1] * pdim_edge_minus1
                + entity_counts[2] * 2 * pdim_edge_minus1 * pdim_edge_minus2
                + entity_counts[3] * 3 * pdim_edge_minus1 * pdim_edge_minus2 * pdim_edge_minus2,
            tdim,
            pdim
        ]
    );

    // vector polynomials of degree <= n-1
    if tdim == 2 {
        for i in 0..pdim_edge_minus1 {
            for j in 0..pdim_edge {
                *wcoeffs
                    .get_mut([i * pdim_edge + j, 0, i * pdim_edge + j])
                    .unwrap() = T::from(1.0).unwrap();
                *wcoeffs
                    .get_mut([
                        pdim_edge_minus1 * pdim_edge + j * pdim_edge_minus1 + i,
                        1,
                        j * pdim_edge + i,
                    ])
                    .unwrap() = T::from(1.0).unwrap();
            }
        }
    } else {
        for i in 0..pdim_edge_minus1 {
            for j in 0..pdim_edge {
                for k in 0..pdim_edge {
                    *wcoeffs
                        .get_mut([
                            i * pdim_edge.pow(2) + j * pdim_edge + k,
                            0,
                            i * pdim_edge.pow(2) + j * pdim_edge + k,
                        ])
                        .unwrap() = T::from(1.0).unwrap();
                    *wcoeffs
                        .get_mut([
                            pdim_edge.pow(2) * pdim_edge_minus1
                                + k * pdim_edge * pdim_edge_minus1
                                + i * pdim_edge
                                + j,
                            1,
                            k * pdim_edge.pow(2) + i * pdim_edge + j,
                        ])
                        .unwrap() = T::from(1.0).unwrap();
                    *wcoeffs
                        .get_mut([
                            pdim_edge.pow(2) * pdim_edge_minus1 * 2
                                + j * pdim_edge * pdim_edge_minus1
                                + k * pdim_edge_minus1
                                + i,
                            2,
                            j * pdim_edge.pow(2) + k * pdim_edge + i,
                        ])
                        .unwrap() = T::from(1.0).unwrap();
                }
            }
        }
    }

    let mut x = [vec![], vec![], vec![], vec![]];
    let mut m = [vec![], vec![], vec![], vec![]];

    let vertices = reference_cell::vertices::<T::Real>(cell_type);

    for _ in 0..entity_counts[0] {
        x[0].push(rlst_dynamic_array2!(T::Real, [tdim, 0]));
        m[0].push(rlst_dynamic_array3!(T, [0, tdim, 0]));
    }

    // DOFs on edges
    let edge_q = gauss_jacobi_rule(ReferenceCellType::Interval, 2 * degree - 1).unwrap();
    let edge_pts_t = edge_q
        .points
        .iter()
        .map(|i| TReal::from(*i).unwrap())
        .collect::<Vec<_>>();
    let edge_pts = rlst_array_from_slice2!(&edge_pts_t, [1, edge_q.npoints]);

    let mut edge_phi = rlst_dynamic_array3![
        T,
        legendre_shape(ReferenceCellType::Interval, &edge_pts, degree - 1, 0)
    ];
    tabulate_legendre_polynomials(
        ReferenceCellType::Interval,
        &edge_pts,
        degree - 1,
        0,
        &mut edge_phi,
    );

    for edge in reference_cell::edges(cell_type) {
        let mut pts = rlst_dynamic_array2!(T::Real, [tdim, edge_q.npoints]);
        let mut mat = rlst_dynamic_array3!(T, [pdim_edge_minus1, tdim, edge_q.npoints]);

        for (w_i, (pt, wt)) in izip!(&edge_pts_t, &edge_q.weights).enumerate() {
            for i in 0..tdim {
                pts[[i, w_i]] =
                    vertices[edge[0]][i] + (vertices[edge[1]][i] - vertices[edge[0]][i]) * *pt;

                for j in 0..pdim_edge_minus1 {
                    mat[[j, i, w_i]] = T::from(*wt).unwrap()
                        * edge_phi[[0, j, w_i]]
                        * T::from(vertices[edge[1]][i] - vertices[edge[0]][i]).unwrap();
                }
            }
        }

        x[1].push(pts);
        m[1].push(mat);
    }

    // DOFs on faces
    if degree == 1 {
        for _ in 0..entity_counts[2] {
            x[2].push(rlst_dynamic_array2!(T::Real, [tdim, 0]));
            m[2].push(rlst_dynamic_array3!(T, [0, tdim, 0]))
        }
    } else {
        let face_q = gauss_jacobi_rule(ReferenceCellType::Quadrilateral, 2 * degree - 1).unwrap();
        let face_pts_t = face_q
            .points
            .iter()
            .map(|i| TReal::from(*i).unwrap())
            .collect::<Vec<_>>();
        let face_pts = rlst_array_from_slice2!(&face_pts_t, [2, face_q.npoints]);

        let mut face_phi = rlst_dynamic_array3![
            T,
            legendre_shape(ReferenceCellType::Quadrilateral, &face_pts, degree - 1, 0)
        ];
        tabulate_legendre_polynomials(
            ReferenceCellType::Quadrilateral,
            &face_pts,
            degree - 1,
            0,
            &mut face_phi,
        );

        for face in reference_cell::faces(cell_type) {
            let mut pts = rlst_dynamic_array2!(T::Real, [tdim, face_q.npoints]);
            let mut mat = rlst_dynamic_array3!(
                T,
                [
                    2 * pdim_edge_minus2 * pdim_edge_minus1,
                    tdim,
                    face_q.npoints
                ]
            );

            for (w_i, wt) in face_q.weights.iter().enumerate() {
                for i in 0..tdim {
                    pts[[i, w_i]] = vertices[face[0]][i]
                        + (vertices[face[1]][i] - vertices[face[0]][i]) * face_pts[[0, w_i]]
                        + (vertices[face[2]][i] - vertices[face[0]][i]) * face_pts[[1, w_i]];
                }
                for i in 0..pdim_edge_minus2 {
                    for j in 0..pdim_edge_minus1 {
                        let index = 2 * (i * pdim_edge_minus1 + j);
                        let entry0 =
                            T::from(*wt).unwrap() * face_phi[[0, j * pdim_edge_minus1 + i, w_i]];
                        let entry1 =
                            T::from(*wt).unwrap() * face_phi[[0, i * pdim_edge_minus1 + j, w_i]];
                        for d in 0..tdim {
                            mat[[index, d, w_i]] = entry0
                                * T::from(vertices[face[1]][d] - vertices[face[0]][d]).unwrap();
                            mat[[index + 1, d, w_i]] = entry1
                                * T::from(vertices[face[2]][d] - vertices[face[0]][d]).unwrap();
                        }
                    }
                }
            }
            x[2].push(pts);
            m[2].push(mat);
        }
    }
    // DOFs on volume
    if tdim == 3 {
        if degree == 1 {
            x[3].push(rlst_dynamic_array2!(T::Real, [tdim, 0]));
            m[3].push(rlst_dynamic_array3!(T, [0, tdim, 0]))
        } else {
            let interior_q =
                gauss_jacobi_rule(ReferenceCellType::Hexahedron, 2 * degree - 1).unwrap();
            let interior_pts_t = interior_q
                .points
                .iter()
                .map(|i| TReal::from(*i).unwrap())
                .collect::<Vec<_>>();
            let interior_pts = rlst_array_from_slice2!(&interior_pts_t, [3, interior_q.npoints]);

            let mut interior_phi = rlst_dynamic_array3![
                T,
                legendre_shape(ReferenceCellType::Hexahedron, &interior_pts, degree - 1, 0)
            ];
            tabulate_legendre_polynomials(
                ReferenceCellType::Hexahedron,
                &interior_pts,
                degree - 1,
                0,
                &mut interior_phi,
            );

            let mut pts = rlst_dynamic_array2!(T::Real, [tdim, interior_q.npoints]);
            let mut mat = rlst_dynamic_array3!(
                T,
                [
                    3 * pdim_edge_minus2.pow(2) * pdim_edge_minus1,
                    tdim,
                    interior_q.npoints
                ]
            );

            for (w_i, wt) in interior_q.weights.iter().enumerate() {
                for i in 0..tdim {
                    pts[[i, w_i]] = interior_pts[[i, w_i]];
                }
                for i in 0..pdim_edge_minus2 {
                    for j in 0..pdim_edge_minus2 {
                        for k in 0..pdim_edge_minus1 {
                            let index = 3
                                * (i * pdim_edge_minus1 * pdim_edge_minus2
                                    + j * pdim_edge_minus1
                                    + k);
                            mat[[index, 0, w_i]] = T::from(*wt).unwrap()
                                * interior_phi[[
                                    0,
                                    k * pdim_edge_minus1.pow(2) + j * pdim_edge_minus1 + i,
                                    w_i,
                                ]];
                            mat[[index + 1, 1, w_i]] = T::from(*wt).unwrap()
                                * interior_phi[[
                                    0,
                                    i * pdim_edge_minus1.pow(2) + k * pdim_edge_minus1 + j,
                                    w_i,
                                ]];
                            mat[[index + 2, 2, w_i]] = T::from(*wt).unwrap()
                                * interior_phi[[
                                    0,
                                    j * pdim_edge_minus1.pow(2) + i * pdim_edge_minus1 + k,
                                    w_i,
                                ]];
                        }
                    }
                }
            }
            x[3].push(pts);
            m[3].push(mat);
        }
    }

    CiarletElement::create(
        "Nedelec (first kind)".to_string(),
        cell_type,
        degree,
        vec![tdim],
        wcoeffs,
        x,
        m,
        continuity,
        degree,
        CovariantPiolaMap {}, // TODO
    )
}

/// Create a Nedelec (first kind) element
pub fn create<T: RlstScalar + MatrixInverse>(
    cell_type: ReferenceCellType,
    degree: usize,
    continuity: Continuity,
) -> CiarletElement<T, CovariantPiolaMap> {
    if cell_type == ReferenceCellType::Triangle || cell_type == ReferenceCellType::Tetrahedron {
        create_simplex(cell_type, degree, continuity)
    } else if cell_type == ReferenceCellType::Quadrilateral
        || cell_type == ReferenceCellType::Hexahedron
    {
        create_tp(cell_type, degree, continuity)
    } else {
        panic!("Invalid cell: {cell_type:?}");
    }
}

/// Nedelec (first kind) element family
pub struct NedelecFirstKindElementFamily<T: RlstScalar + MatrixInverse> {
    degree: usize,
    continuity: Continuity,
    _t: PhantomData<T>,
}

impl<T: RlstScalar + MatrixInverse> NedelecFirstKindElementFamily<T> {
    /// Create new family
    pub fn new(degree: usize, continuity: Continuity) -> Self {
        Self {
            degree,
            continuity,
            _t: PhantomData,
        }
    }
}

impl<T: RlstScalar + MatrixInverse> ElementFamily for NedelecFirstKindElementFamily<T> {
    type T = T;
    type CellType = ReferenceCellType;
    type FiniteElement = CiarletElement<T, CovariantPiolaMap>;
    fn element(&self, cell_type: ReferenceCellType) -> CiarletElement<T, CovariantPiolaMap> {
        create::<T>(cell_type, self.degree, self.continuity)
    }
}
