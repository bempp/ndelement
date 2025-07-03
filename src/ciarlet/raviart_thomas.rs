//! Raviart-Thomas elements

use super::CiarletElement;
use crate::map::ContravariantPiolaMap;
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
) -> CiarletElement<T, ContravariantPiolaMap> {
    if cell_type != ReferenceCellType::Triangle && cell_type != ReferenceCellType::Tetrahedron {
        panic!("Invalid cell: {cell_type:?}");
    }

    if degree < 1 {
        panic!("Degree must be at least 1");
    }

    let tdim = reference_cell::dim(cell_type);
    let facet_type = if tdim == 2 {
        ReferenceCellType::Interval
    } else {
        ReferenceCellType::Triangle
    };
    let pdim_minus1 = polynomial_count(cell_type, degree - 1);
    let pdim_facet_minus1 = polynomial_count(facet_type, degree - 1);
    let pdim_minus2 = if degree < 2 {
        0
    } else {
        polynomial_count(cell_type, degree - 2)
    };

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

    let mut wcoeffs = rlst_dynamic_array3!(T, [pdim_minus1 * tdim + pdim_facet_minus1, tdim, pdim]);

    // vector polynomials of degree <= n-1
    for i in 0..tdim {
        for j in 0..pdim_minus1 {
            *wcoeffs.get_mut([i * pdim_minus1 + j, i, j]).unwrap() = T::from(1.0).unwrap();
        }
    }

    // (px, py, pz) , where p = scalar polynomial of degree = n-1
    for i in 0..pdim_facet_minus1 {
        for k in pdim_minus1..pdim {
            for j in 0..tdim {
                *wcoeffs.get_mut([pdim_minus1 * tdim + i, j, k]).unwrap() = cell_q
                    .weights
                    .iter()
                    .enumerate()
                    .map(|(w_i, wt)| {
                        T::from(*wt).unwrap()
                            * phi[[0, pdim_minus2 + i, w_i]]
                            * T::from(pts[[j, w_i]]).unwrap()
                            * phi[[0, k, w_i]]
                    })
                    .sum();
            }
        }
    }

    orthogonalise_3(&mut wcoeffs, pdim_minus1 * tdim);

    let mut x = [vec![], vec![], vec![], vec![]];
    let mut m = [vec![], vec![], vec![], vec![]];

    let entity_counts = reference_cell::entity_counts(cell_type);
    let vertices = reference_cell::vertices::<T::Real>(cell_type);

    for d in 0..tdim - 1 {
        for _ in 0..entity_counts[d] {
            x[d].push(rlst_dynamic_array2!(T::Real, [tdim, 0]));
            m[d].push(rlst_dynamic_array3!(T, [0, tdim, 0]));
        }
    }

    // Integral moments on facets
    let facet_q = gauss_jacobi_rule(facet_type, 2 * degree - 1).unwrap();
    let facet_pts_t = facet_q
        .points
        .iter()
        .map(|i| TReal::from(*i).unwrap())
        .collect::<Vec<_>>();
    let facet_pts = rlst_array_from_slice2!(&facet_pts_t, [tdim - 1, facet_q.npoints]);

    let mut facet_phi =
        rlst_dynamic_array3![T, legendre_shape(facet_type, &facet_pts, degree - 1, 0)];
    tabulate_legendre_polynomials(facet_type, &facet_pts, degree - 1, 0, &mut facet_phi);

    for (facet, normal) in izip!(
        if tdim == 2 {
            reference_cell::edges(cell_type)
        } else {
            reference_cell::faces(cell_type)
        },
        reference_cell::facet_normals::<TReal>(cell_type),
    ) {
        let mut pts = rlst_dynamic_array2!(T::Real, [tdim, facet_q.npoints]);
        let mut mat = rlst_dynamic_array3!(T, [pdim_facet_minus1, tdim, facet_q.npoints]);

        for (w_i, wt) in facet_q.weights.iter().enumerate() {
            for i in 0..tdim {
                pts[[i, w_i]] = vertices[facet[0]][i]
                    + izip!(
                        facet.iter().skip(1),
                        &facet_q.points[w_i * (tdim - 1)..(w_i + 1) * (tdim - 1)]
                    )
                    .map(|(v_i, pt)| {
                        (vertices[*v_i][i] - vertices[facet[0]][i]) * TReal::from(*pt).unwrap()
                    })
                    .sum();

                for j in 0..pdim_facet_minus1 {
                    mat[[j, i, w_i]] = T::from(*wt).unwrap()
                        * facet_phi[[0, j, w_i]]
                        * T::from(normal[i]).unwrap();
                }
            }
        }

        x[tdim - 1].push(pts);
        m[tdim - 1].push(mat);
    }

    if degree == 1 {
        for _ in 0..entity_counts[tdim] {
            x[tdim].push(rlst_dynamic_array2!(T::Real, [tdim, 0]));
            m[tdim].push(rlst_dynamic_array3!(T, [0, tdim, 0]))
        }
    } else {
        let internal_q = gauss_jacobi_rule(cell_type, 2 * degree - 2).unwrap();
        let mut pts = rlst_dynamic_array2!(T::Real, [tdim, internal_q.npoints]);
        for p in 0..internal_q.npoints {
            for d in 0..tdim {
                pts[[d, p]] = TReal::from(internal_q.points[tdim * p + d]).unwrap()
            }
        }

        let mut internal_phi =
            rlst_dynamic_array3![T, legendre_shape(cell_type, &pts, degree - 2, 0)];
        tabulate_legendre_polynomials(cell_type, &pts, degree - 2, 0, &mut internal_phi);

        let mut mat = rlst_dynamic_array3!(T, [tdim * pdim_minus2, tdim, internal_q.npoints]);
        for (w_i, wt) in internal_q.weights.iter().enumerate() {
            for i in 0..tdim {
                for j in 0..pdim_minus2 {
                    mat[[j + pdim_minus2 * i, i, w_i]] =
                        T::from(*wt).unwrap() * internal_phi[[0, j, w_i]];
                }
            }
        }

        x[tdim].push(pts);
        m[tdim].push(mat);
    }

    CiarletElement::create(
        "Raviart-Thomas".to_string(),
        cell_type,
        degree,
        vec![tdim],
        wcoeffs,
        x,
        m,
        continuity,
        degree,
        ContravariantPiolaMap {}, // TODO
    )
}

fn create_tp<TReal: RlstScalar<Real = TReal>, T: RlstScalar<Real = TReal> + MatrixInverse>(
    cell_type: ReferenceCellType,
    degree: usize,
    continuity: Continuity,
) -> CiarletElement<T, ContravariantPiolaMap> {
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

    let facet_type = if tdim == 2 {
        ReferenceCellType::Interval
    } else {
        ReferenceCellType::Quadrilateral
    };
    let pdim_facet_minus1 = polynomial_count(facet_type, degree - 1);
    let pdim_internal = tdim * pdim_edge_minus2 * pdim_edge_minus1.pow(tdim as u32 - 1);

    let entity_counts = reference_cell::entity_counts(cell_type);

    let mut wcoeffs = rlst_dynamic_array3!(
        T,
        [
            entity_counts[tdim - 1] * pdim_facet_minus1 + entity_counts[tdim] * pdim_internal,
            tdim,
            pdim,
        ]
    );

    // vector polynomials of degree <= n-1
    if tdim == 2 {
        for i in 0..pdim_edge_minus1 {
            for j in 0..pdim_edge {
                *wcoeffs
                    .get_mut([j * pdim_edge_minus1 + i, 0, j * pdim_edge + i])
                    .unwrap() = T::from(1.0).unwrap();
                *wcoeffs
                    .get_mut([
                        pdim_edge_minus1 * pdim_edge + i * pdim_edge + j,
                        1,
                        i * pdim_edge + j,
                    ])
                    .unwrap() = T::from(1.0).unwrap();
            }
        }
    } else {
        for i in 0..pdim_edge_minus1 {
            for j in 0..pdim_edge_minus1 {
                for k in 0..pdim_edge {
                    *wcoeffs
                        .get_mut([
                            k * pdim_edge_minus1.pow(2) + j * pdim_edge_minus1 + i,
                            0,
                            k * pdim_edge.pow(2) + j * pdim_edge + i,
                        ])
                        .unwrap() = T::from(1.0).unwrap();
                    *wcoeffs
                        .get_mut([
                            pdim_edge * pdim_edge_minus1.pow(2)
                                + i * pdim_edge * pdim_edge_minus1
                                + k * pdim_edge_minus1
                                + j,
                            1,
                            i * pdim_edge.pow(2) + k * pdim_edge + j,
                        ])
                        .unwrap() = T::from(1.0).unwrap();
                    *wcoeffs
                        .get_mut([
                            pdim_edge * pdim_edge_minus1.pow(2) * 2
                                + j * pdim_edge * pdim_edge_minus1
                                + i * pdim_edge
                                + k,
                            2,
                            j * pdim_edge.pow(2) + i * pdim_edge + k,
                        ])
                        .unwrap() = T::from(1.0).unwrap();
                }
            }
        }
    }

    let mut x = [vec![], vec![], vec![], vec![]];
    let mut m = [vec![], vec![], vec![], vec![]];

    let vertices = reference_cell::vertices::<T::Real>(cell_type);

    for d in 0..tdim - 1 {
        for _ in 0..entity_counts[d] {
            x[d].push(rlst_dynamic_array2!(T::Real, [tdim, 0]));
            m[d].push(rlst_dynamic_array3!(T, [0, tdim, 0]));
        }
    }
    // DOFs on facets
    let facet_q = gauss_jacobi_rule(facet_type, 2 * degree - 1).unwrap();
    let facet_pts_t = facet_q
        .points
        .iter()
        .map(|i| TReal::from(*i).unwrap())
        .collect::<Vec<_>>();
    let facet_pts = rlst_array_from_slice2!(&facet_pts_t, [tdim - 1, facet_q.npoints]);

    let mut facet_phi =
        rlst_dynamic_array3![T, legendre_shape(facet_type, &facet_pts, degree - 1, 0)];
    tabulate_legendre_polynomials(facet_type, &facet_pts, degree - 1, 0, &mut facet_phi);

    for (facet, normal) in izip!(
        reference_cell::facets(cell_type),
        reference_cell::facet_normals::<T::Real>(cell_type),
    ) {
        let mut pts = rlst_dynamic_array2!(T::Real, [tdim, facet_q.npoints]);
        let mut mat = rlst_dynamic_array3!(T, [pdim_facet_minus1, tdim, facet_q.npoints]);

        for (w_i, wt) in facet_q.weights.iter().enumerate() {
            for d in 0..tdim {
                pts[[d, w_i]] = vertices[facet[0]][d]
                    + (0..tdim - 1)
                        .map(|x| {
                            (vertices[facet[usize::pow(2, x as u32)]][d] - vertices[facet[0]][d])
                                * facet_pts[[x, w_i]]
                        })
                        .sum::<T::Real>();
                for j in 0..facet_phi.shape()[1] {
                    mat[[j, d, w_i]] = T::from(*wt).unwrap()
                        * facet_phi[[0, j, w_i]]
                        * T::from(normal[d]).unwrap();
                }
            }
        }

        x[tdim - 1].push(pts);
        m[tdim - 1].push(mat);
    }

    // DOFs on interior
    if degree == 1 {
        x[tdim].push(rlst_dynamic_array2!(T::Real, [tdim, 0]));
        m[tdim].push(rlst_dynamic_array3!(T, [0, tdim, 0]))
    } else if tdim == 2 {
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
                pts[[i, w_i]] = face_pts[[i, w_i]];
            }
            for i in 0..pdim_edge_minus2 {
                for j in 0..pdim_edge_minus1 {
                    let index = 2 * (i * pdim_edge_minus1 + j);
                    mat[[index, 0, w_i]] =
                        T::from(*wt).unwrap() * face_phi[[0, i * pdim_edge_minus1 + j, w_i]];
                    mat[[index + 1, 1, w_i]] =
                        T::from(*wt).unwrap() * face_phi[[0, j * pdim_edge_minus1 + i, w_i]];
                }
            }
        }
        x[2].push(pts);
        m[2].push(mat);
    } else {
        let interior_q = gauss_jacobi_rule(ReferenceCellType::Hexahedron, 2 * degree - 1).unwrap();
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
                3 * pdim_edge_minus1.pow(2) * pdim_edge_minus2,
                tdim,
                interior_q.npoints
            ]
        );

        for (w_i, wt) in interior_q.weights.iter().enumerate() {
            for i in 0..tdim {
                pts[[i, w_i]] = interior_pts[[i, w_i]];
            }
            for i in 0..pdim_edge_minus2 {
                for j in 0..pdim_edge_minus1 {
                    for k in 0..pdim_edge_minus1 {
                        let index = 3 * (i * pdim_edge_minus1.pow(2) + j * pdim_edge_minus1 + k);
                        mat[[index, 0, w_i]] = T::from(*wt).unwrap()
                            * interior_phi[[
                                0,
                                i * pdim_edge_minus1.pow(2) + j * pdim_edge_minus1 + k,
                                w_i,
                            ]];
                        mat[[index + 1, 1, w_i]] = T::from(*wt).unwrap()
                            * interior_phi[[
                                0,
                                k * pdim_edge_minus1.pow(2) + i * pdim_edge_minus1 + j,
                                w_i,
                            ]];
                        mat[[index + 2, 2, w_i]] = T::from(*wt).unwrap()
                            * interior_phi[[
                                0,
                                j * pdim_edge_minus1.pow(2) + k * pdim_edge_minus1 + i,
                                w_i,
                            ]];
                    }
                }
            }
        }
        x[3].push(pts);
        m[3].push(mat);
    }

    CiarletElement::create(
        "Raviart-Thomas".to_string(),
        cell_type,
        degree,
        vec![tdim],
        wcoeffs,
        x,
        m,
        continuity,
        degree,
        ContravariantPiolaMap {}, // TODO
    )
}

/// Create a Raviart-Thomas element
pub fn create<T: RlstScalar + MatrixInverse>(
    cell_type: ReferenceCellType,
    degree: usize,
    continuity: Continuity,
) -> CiarletElement<T, ContravariantPiolaMap> {
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

/// Raviart-Thomas element family
pub struct RaviartThomasElementFamily<T: RlstScalar + MatrixInverse> {
    degree: usize,
    continuity: Continuity,
    _t: PhantomData<T>,
}

impl<T: RlstScalar + MatrixInverse> RaviartThomasElementFamily<T> {
    /// Create new family
    pub fn new(degree: usize, continuity: Continuity) -> Self {
        Self {
            degree,
            continuity,
            _t: PhantomData,
        }
    }
}

impl<T: RlstScalar + MatrixInverse> ElementFamily for RaviartThomasElementFamily<T> {
    type T = T;
    type CellType = ReferenceCellType;
    type FiniteElement = CiarletElement<T, ContravariantPiolaMap>;
    fn element(&self, cell_type: ReferenceCellType) -> CiarletElement<T, ContravariantPiolaMap> {
        create::<T>(cell_type, self.degree, self.continuity)
    }
}
