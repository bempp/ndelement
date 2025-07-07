//! Finite element definitions

use crate::math;
use crate::polynomials::{legendre_shape, polynomial_count, tabulate_legendre_polynomials};
use crate::reference_cell;
use crate::traits::{FiniteElement, Map};
use crate::types::{Continuity, DofTransformation, ReferenceCellType, Transformation};
use itertools::izip;
use num::{One, Zero};
use rlst::{
    rlst_dynamic_array2, rlst_dynamic_array3, rlst_dynamic_array4, Array, BaseArray, MatrixInverse,
    RandomAccessByRef, RandomAccessMut, RawAccess, RlstScalar, Shape, VectorContainer,
};
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};

pub mod lagrange;
pub mod nedelec;
pub mod raviart_thomas;
pub use lagrange::LagrangeElementFamily;
pub use nedelec::NedelecFirstKindElementFamily;
pub use raviart_thomas::RaviartThomasElementFamily;

type EntityPoints<T> = [Vec<Array<T, BaseArray<T, VectorContainer<T>, 2>, 2>>; 4];
type EntityWeights<T> = [Vec<Array<T, BaseArray<T, VectorContainer<T>, 3>, 3>>; 4];

/// Compute the number of derivatives for a cell
fn compute_derivative_count(nderivs: usize, cell_type: ReferenceCellType) -> usize {
    match cell_type {
        ReferenceCellType::Point => 0,
        ReferenceCellType::Interval => nderivs + 1,
        ReferenceCellType::Triangle => (nderivs + 1) * (nderivs + 2) / 2,
        ReferenceCellType::Quadrilateral => (nderivs + 1) * (nderivs + 2) / 2,
        ReferenceCellType::Tetrahedron => (nderivs + 1) * (nderivs + 2) * (nderivs + 3) / 6,
        ReferenceCellType::Hexahedron => (nderivs + 1) * (nderivs + 2) * (nderivs + 3) / 6,
        ReferenceCellType::Prism => (nderivs + 1) * (nderivs + 2) * (nderivs + 3) / 6,
        ReferenceCellType::Pyramid => (nderivs + 1) * (nderivs + 2) * (nderivs + 3) / 6,
    }
}

/// A Ciarlet element
pub struct CiarletElement<T: RlstScalar + MatrixInverse, M: Map> {
    family_name: String,
    cell_type: ReferenceCellType,
    degree: usize,
    embedded_superdegree: usize,
    value_shape: Vec<usize>,
    value_size: usize,
    continuity: Continuity,
    dim: usize,
    coefficients: Array<T, BaseArray<T, VectorContainer<T>, 3>, 3>,
    entity_dofs: [Vec<Vec<usize>>; 4],
    entity_closure_dofs: [Vec<Vec<usize>>; 4],
    interpolation_points: EntityPoints<T::Real>,
    interpolation_weights: EntityWeights<T>,
    map: M,
    dof_transformations: HashMap<(ReferenceCellType, Transformation), DofTransformation<T>>,
}

impl<T: RlstScalar + MatrixInverse, M: Map> Debug for CiarletElement<T, M> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        f.debug_tuple("CiarletElement")
            .field(&self.family_name)
            .field(&self.cell_type)
            .field(&self.degree)
            .finish()
    }
}

impl<T: RlstScalar + MatrixInverse, M: Map> CiarletElement<T, M> {
    /// Create a Ciarlet element
    #[allow(clippy::too_many_arguments)]
    pub fn create(
        family_name: String,
        cell_type: ReferenceCellType,
        degree: usize,
        value_shape: Vec<usize>,
        polynomial_coeffs: Array<T, BaseArray<T, VectorContainer<T>, 3>, 3>,
        interpolation_points: EntityPoints<T::Real>,
        interpolation_weights: EntityWeights<T>,
        continuity: Continuity,
        embedded_superdegree: usize,
        map: M,
    ) -> Self {
        let mut dim = 0;
        let mut npts = 0;

        for emats in &interpolation_weights {
            for mat in emats {
                dim += mat.shape()[0];
                npts += mat.shape()[2];
            }
        }
        let tdim = reference_cell::dim(cell_type);

        let mut value_size = 1;
        for i in &value_shape {
            value_size *= *i;
        }

        for matrices in &interpolation_weights {
            for mat in matrices {
                if mat.shape()[1] != value_size {
                    panic!("Incompatible value size");
                }
            }
        }

        // Format the interpolation points and weights
        let new_pts = if continuity == Continuity::Discontinuous {
            let mut new_pts: EntityPoints<T::Real> = [vec![], vec![], vec![], vec![]];
            let mut all_pts = rlst_dynamic_array2![T::Real, [tdim, npts]];
            for (i, pts_i) in interpolation_points.iter().take(tdim).enumerate() {
                for _pts in pts_i {
                    new_pts[i].push(rlst_dynamic_array2![T::Real, [tdim, 0]]);
                }
            }
            let mut col = 0;
            for pts_i in interpolation_points.iter() {
                for pts in pts_i {
                    let ncols = pts.shape()[1];
                    all_pts
                        .view_mut()
                        .into_subview([0, col], [tdim, ncols])
                        .fill_from(pts.view());
                    col += ncols;
                }
            }
            new_pts[tdim].push(all_pts);
            new_pts
        } else {
            interpolation_points
        };
        let new_wts = if continuity == Continuity::Discontinuous {
            let mut new_wts = [vec![], vec![], vec![], vec![]];
            let mut pn = 0;
            let mut dn = 0;
            let mut all_mat = rlst_dynamic_array3!(T, [dim, value_size, npts]);
            for (i, mi) in interpolation_weights.iter().take(tdim).enumerate() {
                for _mat in mi {
                    new_wts[i].push(rlst_dynamic_array3!(T, [0, value_size, 0]));
                }
            }
            for mi in interpolation_weights.iter() {
                for mat in mi {
                    let dim0 = mat.shape()[0];
                    let dim2 = mat.shape()[2];
                    all_mat
                        .view_mut()
                        .into_subview([dn, 0, pn], [dim0, value_size, dim2])
                        .fill_from(mat.view());
                    dn += dim0;
                    pn += dim2;
                }
            }
            new_wts[tdim].push(all_mat);
            new_wts
        } else {
            interpolation_weights
        };

        // Compute the dual matrix
        let pdim = polynomial_count(cell_type, embedded_superdegree);
        let mut d_matrix = rlst_dynamic_array3!(T, [value_size, pdim, dim]);

        let mut dof = 0;
        for d in 0..4 {
            for (e, pts) in new_pts[d].iter().enumerate() {
                if pts.shape()[1] > 0 {
                    let mut table = rlst_dynamic_array3!(T, [1, pdim, pts.shape()[1]]);
                    tabulate_legendre_polynomials(
                        cell_type,
                        pts,
                        embedded_superdegree,
                        0,
                        &mut table,
                    );
                    let mat = &new_wts[d][e];
                    for i in 0..mat.shape()[0] {
                        for l in 0..pdim {
                            for j in 0..value_size {
                                // d_matrix[j, l, dof + i] = inner(mat[i, j, :], table[0, l, :])
                                *d_matrix.get_mut([j, l, dof + i]).unwrap() = mat
                                    .view()
                                    .slice(0, i)
                                    .slice(0, j)
                                    .inner(table.view().slice(0, 0).slice(0, l));
                            }
                        }
                    }
                    dof += mat.shape()[0];
                }
            }
        }

        // Compute the coefficients that define the basis functions
        let mut inverse = rlst::rlst_dynamic_array2!(T, [dim, dim]);

        for i in 0..dim {
            for j in 0..dim {
                *inverse.get_mut([i, j]).unwrap() = (0..value_size)
                    .map(|k| {
                        (0..pdim)
                            .map(|l| {
                                *polynomial_coeffs.get([i, k, l]).unwrap()
                                    * *d_matrix.get([k, l, j]).unwrap()
                            })
                            .sum::<T>()
                    })
                    .sum::<T>();
            }
        }

        inverse.view_mut().into_inverse_alloc().unwrap();

        let mut coefficients = rlst_dynamic_array3!(T, [dim, value_size, pdim]);
        for i in 0..dim {
            for j in 0..value_size {
                for k in 0..pdim {
                    // coefficients[i, j, k] = inner(inverse[i, :], polynomial_coeffs[:, j, k])
                    *coefficients.get_mut([i, j, k]).unwrap() = inverse
                        .view()
                        .slice(0, i)
                        .inner(polynomial_coeffs.view().slice(1, j).slice(1, k));
                }
            }
        }

        // Compute entity DOFs and entity closure DOFs
        let mut entity_dofs = [vec![], vec![], vec![], vec![]];
        let mut dof = 0;
        for i in 0..4 {
            for wts in &new_wts[i] {
                let dofs = (dof..dof + wts.shape()[0]).collect::<Vec<_>>();
                entity_dofs[i].push(dofs);
                dof += wts.shape()[0];
            }
        }
        let connectivity = reference_cell::connectivity(cell_type);
        let mut entity_closure_dofs = [vec![], vec![], vec![], vec![]];
        for (edim, (ecdofs, connectivity_edim)) in
            izip!(entity_closure_dofs.iter_mut(), &connectivity).enumerate()
        {
            for connectivity_edim_eindex in connectivity_edim {
                let mut cdofs = vec![];
                for (edim2, connectivity_edim_eindex_edim2) in
                    connectivity_edim_eindex.iter().take(edim + 1).enumerate()
                {
                    for index in connectivity_edim_eindex_edim2 {
                        for i in &entity_dofs[edim2][*index] {
                            cdofs.push(*i)
                        }
                    }
                }
                ecdofs.push(cdofs);
            }
        }

        // Compute DOF transformations
        let mut dof_transformations = HashMap::new();
        for (entity, entity_id, transform, f) in match cell_type {
            ReferenceCellType::Point => vec![],
            ReferenceCellType::Interval => vec![],
            ReferenceCellType::Triangle => vec![(
                ReferenceCellType::Interval,
                0,
                Transformation::Reflection,
                (|x| vec![x[1], x[0]]) as fn(&[T::Real]) -> Vec<T::Real>,
            )],
            ReferenceCellType::Quadrilateral => vec![(
                ReferenceCellType::Interval,
                0,
                Transformation::Reflection,
                (|x| vec![T::Real::one() - x[0], x[1]]) as fn(&[T::Real]) -> Vec<T::Real>,
            )],
            ReferenceCellType::Tetrahedron => vec![
                (
                    ReferenceCellType::Interval,
                    0,
                    Transformation::Reflection,
                    (|x| vec![x[0], x[2], x[1]]) as fn(&[T::Real]) -> Vec<T::Real>,
                ),
                (
                    ReferenceCellType::Triangle,
                    0,
                    Transformation::Rotation,
                    (|x| vec![x[1], x[2], x[0]]) as fn(&[T::Real]) -> Vec<T::Real>,
                ),
                (
                    ReferenceCellType::Triangle,
                    0,
                    Transformation::Reflection,
                    (|x| vec![x[0], x[2], x[1]]) as fn(&[T::Real]) -> Vec<T::Real>,
                ),
            ],
            ReferenceCellType::Hexahedron => vec![
                (
                    ReferenceCellType::Interval,
                    0,
                    Transformation::Reflection,
                    (|x| vec![T::Real::one() - x[0], x[1], x[2]]) as fn(&[T::Real]) -> Vec<T::Real>,
                ),
                (
                    ReferenceCellType::Quadrilateral,
                    0,
                    Transformation::Rotation,
                    (|x| vec![x[1], T::Real::one() - x[0], x[2]]) as fn(&[T::Real]) -> Vec<T::Real>,
                ),
                (
                    ReferenceCellType::Quadrilateral,
                    0,
                    Transformation::Reflection,
                    (|x| vec![x[1], x[0], x[2]]) as fn(&[T::Real]) -> Vec<T::Real>,
                ),
            ],
            ReferenceCellType::Prism => vec![
                (
                    ReferenceCellType::Interval,
                    0,
                    Transformation::Reflection,
                    (|x| vec![T::Real::one() - x[0], x[1], x[2]]) as fn(&[T::Real]) -> Vec<T::Real>,
                ),
                (
                    ReferenceCellType::Triangle,
                    0,
                    Transformation::Rotation,
                    (|x| vec![x[1], T::Real::one() - x[1] - x[0], x[2]])
                        as fn(&[T::Real]) -> Vec<T::Real>,
                ),
                (
                    ReferenceCellType::Triangle,
                    0,
                    Transformation::Reflection,
                    (|x| vec![x[1], x[0], x[2]]) as fn(&[T::Real]) -> Vec<T::Real>,
                ),
                (
                    ReferenceCellType::Quadrilateral,
                    1,
                    Transformation::Rotation,
                    (|x| vec![x[2], T::Real::one() - x[1], x[0]]) as fn(&[T::Real]) -> Vec<T::Real>,
                ),
                (
                    ReferenceCellType::Quadrilateral,
                    1,
                    Transformation::Reflection,
                    (|x| vec![x[2], x[1], x[0]]) as fn(&[T::Real]) -> Vec<T::Real>,
                ),
            ],
            ReferenceCellType::Pyramid => vec![
                (
                    ReferenceCellType::Interval,
                    0,
                    Transformation::Reflection,
                    (|x| vec![T::Real::one() - x[0], x[1], x[2]]) as fn(&[T::Real]) -> Vec<T::Real>,
                ),
                (
                    ReferenceCellType::Triangle,
                    1,
                    Transformation::Rotation,
                    (|x| vec![x[2], x[1], T::Real::one() - x[2] - x[0]])
                        as fn(&[T::Real]) -> Vec<T::Real>,
                ),
                (
                    ReferenceCellType::Triangle,
                    1,
                    Transformation::Reflection,
                    (|x| vec![x[2], x[1], x[0]]) as fn(&[T::Real]) -> Vec<T::Real>,
                ),
                (
                    ReferenceCellType::Quadrilateral,
                    0,
                    Transformation::Rotation,
                    (|x| vec![x[1], T::Real::one() - x[0], x[2]]) as fn(&[T::Real]) -> Vec<T::Real>,
                ),
                (
                    ReferenceCellType::Quadrilateral,
                    0,
                    Transformation::Reflection,
                    (|x| vec![x[1], x[0], x[2]]) as fn(&[T::Real]) -> Vec<T::Real>,
                ),
            ],
        } {
            let edim = reference_cell::dim(entity);
            let ref_pts = &new_pts[edim][entity_id];
            let npts = ref_pts.shape()[1];

            let finv = match transform {
                Transformation::Reflection => {
                    (|x, f| f(x)) as fn(&[T::Real], fn(&[T::Real]) -> Vec<T::Real>) -> Vec<T::Real>
                }
                Transformation::Rotation => match entity {
                    ReferenceCellType::Interval => {
                        (|x, f| f(x))
                            as fn(&[T::Real], fn(&[T::Real]) -> Vec<T::Real>) -> Vec<T::Real>
                    }
                    ReferenceCellType::Triangle => {
                        (|x, f| f(&f(x)))
                            as fn(&[T::Real], fn(&[T::Real]) -> Vec<T::Real>) -> Vec<T::Real>
                    }
                    ReferenceCellType::Quadrilateral => {
                        (|x, f| f(&f(&f(x))))
                            as fn(&[T::Real], fn(&[T::Real]) -> Vec<T::Real>) -> Vec<T::Real>
                    }
                    _ => panic!("Unsupported entity: {entity:?}"),
                },
            };

            let mut pts = rlst_dynamic_array2!(T::Real, ref_pts.shape());
            for p in 0..npts {
                for (i, c) in finv(ref_pts.view().slice(1, p).data(), f)
                    .iter()
                    .enumerate()
                {
                    *pts.get_mut([i, p]).unwrap() = *c
                }
            }

            let wts = &new_wts[edim][entity_id];
            let edofs = &entity_dofs[edim][entity_id];
            let endofs = edofs.len();
            let mut j = rlst_dynamic_array3![T::Real, [npts, tdim, tdim]];
            for t_in in 0..tdim {
                for (t_out, (a, b)) in izip!(
                    f(&vec![T::Real::zero(); tdim]),
                    f(&{
                        let mut axis = vec![T::Real::zero(); tdim];
                        axis[t_in] = T::Real::one();
                        axis
                    })
                )
                .enumerate()
                {
                    for p in 0..npts {
                        *j.get_mut([p, t_out, t_in]).unwrap() = b - a;
                    }
                }
            }
            // f is linear. Use this to compute determinants
            let jdet = vec![
                match transform {
                    Transformation::Reflection => -T::Real::one(),
                    Transformation::Rotation => T::Real::one(),
                };
                npts
            ];
            let mut jinv = rlst_dynamic_array3![T::Real, [npts, tdim, tdim]];
            for t_in in 0..tdim {
                for (t_out, (a, b)) in izip!(
                    finv(&vec![T::Real::zero(); tdim], f),
                    finv(
                        &{
                            let mut axis = vec![T::Real::zero(); tdim];
                            axis[t_in] = T::Real::one();
                            axis
                        },
                        f
                    )
                )
                .enumerate()
                {
                    for p in 0..npts {
                        *jinv.get_mut([p, t_out, t_in]).unwrap() = (b - a).re();
                    }
                }
            }

            let mut table =
                rlst_dynamic_array3!(T, legendre_shape(cell_type, &pts, embedded_superdegree, 0));
            tabulate_legendre_polynomials(cell_type, &pts, embedded_superdegree, 0, &mut table);

            let mut data = rlst_dynamic_array4!(T, [1, npts, endofs, value_size]);
            for p in 0..npts {
                for j in 0..value_size {
                    for (b, dof) in edofs.iter().enumerate() {
                        // data[0, p, b, j] = inner(self.coefficients[b, j, :], table[0, :, p])
                        *data.get_mut([0, p, b, j]).unwrap() = coefficients
                            .view()
                            .slice(0, *dof)
                            .slice(0, j)
                            .inner(table.view().slice(0, 0).slice(1, p));
                    }
                }
            }

            let mut pushed_data = rlst_dynamic_array4!(T, [1, npts, endofs, value_size]);
            map.push_forward(&data, 0, &j, &jdet, &jinv, &mut pushed_data);

            let mut dof_transform = rlst_dynamic_array2!(T, [edofs.len(), edofs.len()]);
            for i in 0..edofs.len() {
                for j in 0..edofs.len() {
                    *dof_transform.get_mut([i, j]).unwrap() = (0..value_size)
                        .map(|l| {
                            (0..npts)
                                .map(|m| {
                                    *wts.get([j, l, m]).unwrap()
                                        * *pushed_data.get([0, m, i, l]).unwrap()
                                })
                                .sum::<T>()
                        })
                        .sum::<T>();
                }
            }

            let perm = math::prepare_matrix(&mut dof_transform);

            // Check if transformation is the identity or a permutation
            let mut is_identity = true;
            'outer: for j in 0..edofs.len() {
                for i in 0..edofs.len() {
                    if (*dof_transform.get([i, j]).unwrap()
                        - if i == j { T::one() } else { T::zero() })
                    .abs()
                        > T::from(1e-8).unwrap().re()
                    {
                        is_identity = false;
                        break 'outer;
                    }
                }
            }

            if is_identity {
                let mut is_unpermuted = true;
                for (i, p) in perm.iter().enumerate() {
                    if i != *p {
                        is_unpermuted = false;
                        break;
                    }
                }
                if is_unpermuted {
                    dof_transformations.insert((entity, transform), DofTransformation::Identity);
                } else {
                    dof_transformations
                        .insert((entity, transform), DofTransformation::Permutation(perm));
                }
            } else {
                dof_transformations.insert(
                    (entity, transform),
                    DofTransformation::Transformation(dof_transform, perm),
                );
            }
        }
        CiarletElement::<T, M> {
            family_name,
            cell_type,
            degree,
            embedded_superdegree,
            value_shape,
            value_size,
            continuity,
            dim,
            coefficients,
            entity_dofs,
            entity_closure_dofs,
            interpolation_points: new_pts,
            interpolation_weights: new_wts,
            map,
            dof_transformations,
        }
    }

    /// The polynomial degree
    pub fn degree(&self) -> usize {
        self.degree
    }
    /// The continuity of the element
    pub fn continuity(&self) -> Continuity {
        self.continuity
    }
    /// The interpolation points
    pub fn interpolation_points(&self) -> &EntityPoints<T::Real> {
        &self.interpolation_points
    }
    /// The interpolation weights
    pub fn interpolation_weights(&self) -> &EntityWeights<T> {
        &self.interpolation_weights
    }
}
impl<T: RlstScalar + MatrixInverse, M: Map> FiniteElement for CiarletElement<T, M> {
    type CellType = ReferenceCellType;
    type TransformationType = Transformation;
    type T = T;
    fn value_shape(&self) -> &[usize] {
        &self.value_shape
    }
    fn value_size(&self) -> usize {
        self.value_size
    }

    fn cell_type(&self) -> ReferenceCellType {
        self.cell_type
    }
    fn embedded_superdegree(&self) -> usize {
        self.embedded_superdegree
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn tabulate<Array2: RandomAccessByRef<2, Item = T::Real> + Shape<2>>(
        &self,
        points: &Array2,
        nderivs: usize,
        data: &mut impl RandomAccessMut<4, Item = T>,
    ) {
        let mut table = rlst_dynamic_array3!(
            T,
            legendre_shape(self.cell_type, points, self.embedded_superdegree, nderivs)
        );
        tabulate_legendre_polynomials(
            self.cell_type,
            points,
            self.embedded_superdegree,
            nderivs,
            &mut table,
        );

        for d in 0..table.shape()[0] {
            for p in 0..points.shape()[1] {
                for j in 0..self.value_size {
                    for b in 0..self.dim {
                        // data[d, p, b, j] = inner(self.coefficients[b, j, :], table[d, :, p])
                        *data.get_mut([d, p, b, j]).unwrap() = self
                            .coefficients
                            .view()
                            .slice(0, b)
                            .slice(0, j)
                            .inner(table.view().slice(0, d).slice(1, p));
                    }
                }
            }
        }
    }
    fn entity_dofs(&self, entity_dim: usize, entity_number: usize) -> Option<&[usize]> {
        if entity_dim < 4 && entity_number < self.entity_dofs[entity_dim].len() {
            Some(&self.entity_dofs[entity_dim][entity_number])
        } else {
            None
        }
    }
    fn entity_closure_dofs(&self, entity_dim: usize, entity_number: usize) -> Option<&[usize]> {
        if entity_dim < 4 && entity_number < self.entity_closure_dofs[entity_dim].len() {
            Some(&self.entity_closure_dofs[entity_dim][entity_number])
        } else {
            None
        }
    }
    fn tabulate_array_shape(&self, nderivs: usize, npoints: usize) -> [usize; 4] {
        let deriv_count = compute_derivative_count(nderivs, self.cell_type());
        let point_count = npoints;
        let basis_count = self.dim();
        let value_size = self.value_size();
        [deriv_count, point_count, basis_count, value_size]
    }
    fn push_forward<
        Array3Real: RandomAccessByRef<3, Item = <Self::T as RlstScalar>::Real> + Shape<3>,
        Array4: RandomAccessByRef<4, Item = Self::T> + Shape<4>,
        Array4Mut: RandomAccessMut<4, Item = Self::T> + Shape<4>,
    >(
        &self,
        reference_values: &Array4,
        nderivs: usize,
        jacobians: &Array3Real,
        jacobian_determinants: &[<Self::T as RlstScalar>::Real],
        inverse_jacobians: &Array3Real,
        physical_values: &mut Array4Mut,
    ) {
        self.map.push_forward(
            reference_values,
            nderivs,
            jacobians,
            jacobian_determinants,
            inverse_jacobians,
            physical_values,
        )
    }
    fn pull_back<
        Array3Real: RandomAccessByRef<3, Item = <Self::T as RlstScalar>::Real> + Shape<3>,
        Array4: RandomAccessByRef<4, Item = Self::T> + Shape<4>,
        Array4Mut: RandomAccessMut<4, Item = Self::T> + Shape<4>,
    >(
        &self,
        physical_values: &Array4,
        nderivs: usize,
        jacobians: &Array3Real,
        jacobian_determinants: &[<Self::T as RlstScalar>::Real],
        inverse_jacobians: &Array3Real,
        reference_values: &mut Array4Mut,
    ) {
        self.map.pull_back(
            physical_values,
            nderivs,
            jacobians,
            jacobian_determinants,
            inverse_jacobians,
            reference_values,
        )
    }
    fn physical_value_shape(&self, gdim: usize) -> Vec<usize> {
        self.map.physical_value_shape(gdim)
    }
    fn dof_transformation(
        &self,
        entity: ReferenceCellType,
        transformation: Transformation,
    ) -> Option<&DofTransformation<T>> {
        self.dof_transformations.get(&(entity, transformation))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::*;
    use paste::paste;
    use rlst::rlst_dynamic_array4;

    fn check_dofs(e: impl FiniteElement<CellType = ReferenceCellType>) {
        let mut ndofs = 0;
        for (dim, entity_count) in match e.cell_type() {
            ReferenceCellType::Point => vec![1],
            ReferenceCellType::Interval => vec![2, 1],
            ReferenceCellType::Triangle => vec![3, 3, 1],
            ReferenceCellType::Quadrilateral => vec![4, 4, 1],
            ReferenceCellType::Tetrahedron => vec![4, 6, 4, 1],
            ReferenceCellType::Hexahedron => vec![8, 12, 6, 1],
            ReferenceCellType::Prism => vec![6, 9, 5, 1],
            ReferenceCellType::Pyramid => vec![5, 8, 5, 1],
        }
        .iter()
        .enumerate()
        {
            for entity in 0..*entity_count {
                ndofs += e.entity_dofs(dim, entity).unwrap().len();
            }
        }
        assert_eq!(ndofs, e.dim());
    }

    #[test]
    fn test_lagrange_1() {
        let e = lagrange::create::<f64>(ReferenceCellType::Triangle, 1, Continuity::Standard);
        assert_eq!(e.value_size(), 1);
    }

    #[test]
    fn test_lagrange_0_interval() {
        let e = lagrange::create::<f64>(ReferenceCellType::Interval, 0, Continuity::Discontinuous);
        assert_eq!(e.value_size(), 1);
        let mut data = rlst_dynamic_array4!(f64, e.tabulate_array_shape(0, 4));
        let mut points = rlst_dynamic_array2!(f64, [1, 4]);
        *points.get_mut([0, 0]).unwrap() = 0.0;
        *points.get_mut([0, 1]).unwrap() = 0.2;
        *points.get_mut([0, 2]).unwrap() = 0.4;
        *points.get_mut([0, 3]).unwrap() = 1.0;
        e.tabulate(&points, 0, &mut data);

        for pt in 0..4 {
            assert_relative_eq!(*data.get([0, pt, 0, 0]).unwrap(), 1.0);
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_1_interval() {
        let e = lagrange::create::<f64>(ReferenceCellType::Interval, 1, Continuity::Standard);
        assert_eq!(e.value_size(), 1);
        let mut data = rlst_dynamic_array4!(f64, e.tabulate_array_shape(0, 4));
        let mut points = rlst_dynamic_array2!(f64, [1, 4]);
        *points.get_mut([0, 0]).unwrap() = 0.0;
        *points.get_mut([0, 1]).unwrap() = 0.2;
        *points.get_mut([0, 2]).unwrap() = 0.4;
        *points.get_mut([0, 3]).unwrap() = 1.0;
        e.tabulate(&points, 0, &mut data);

        for pt in 0..4 {
            let x = *points.get([0, pt]).unwrap();
            assert_relative_eq!(*data.get([0, pt, 0, 0]).unwrap(), 1.0 - x);
            assert_relative_eq!(*data.get([0, pt, 1, 0]).unwrap(), x);
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_0_triangle() {
        let e = lagrange::create::<f64>(ReferenceCellType::Triangle, 0, Continuity::Discontinuous);
        assert_eq!(e.value_size(), 1);
        let mut data = rlst_dynamic_array4!(f64, e.tabulate_array_shape(0, 6));

        let mut points = rlst_dynamic_array2!(f64, [2, 6]);
        *points.get_mut([0, 0]).unwrap() = 0.0;
        *points.get_mut([1, 0]).unwrap() = 0.0;
        *points.get_mut([0, 1]).unwrap() = 1.0;
        *points.get_mut([1, 1]).unwrap() = 0.0;
        *points.get_mut([0, 2]).unwrap() = 0.0;
        *points.get_mut([1, 2]).unwrap() = 1.0;
        *points.get_mut([0, 3]).unwrap() = 0.5;
        *points.get_mut([1, 3]).unwrap() = 0.0;
        *points.get_mut([0, 4]).unwrap() = 0.0;
        *points.get_mut([1, 4]).unwrap() = 0.5;
        *points.get_mut([0, 5]).unwrap() = 0.5;
        *points.get_mut([1, 5]).unwrap() = 0.5;

        e.tabulate(&points, 0, &mut data);

        for pt in 0..6 {
            assert_relative_eq!(*data.get([0, pt, 0, 0]).unwrap(), 1.0);
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_1_triangle() {
        let e = lagrange::create::<f64>(ReferenceCellType::Triangle, 1, Continuity::Standard);
        assert_eq!(e.value_size(), 1);
        let mut data = rlst_dynamic_array4!(f64, e.tabulate_array_shape(0, 6));
        let mut points = rlst_dynamic_array2!(f64, [2, 6]);
        *points.get_mut([0, 0]).unwrap() = 0.0;
        *points.get_mut([1, 0]).unwrap() = 0.0;
        *points.get_mut([0, 1]).unwrap() = 1.0;
        *points.get_mut([1, 1]).unwrap() = 0.0;
        *points.get_mut([0, 2]).unwrap() = 0.0;
        *points.get_mut([1, 2]).unwrap() = 1.0;
        *points.get_mut([0, 3]).unwrap() = 0.5;
        *points.get_mut([1, 3]).unwrap() = 0.0;
        *points.get_mut([0, 4]).unwrap() = 0.0;
        *points.get_mut([1, 4]).unwrap() = 0.5;
        *points.get_mut([0, 5]).unwrap() = 0.5;
        *points.get_mut([1, 5]).unwrap() = 0.5;
        e.tabulate(&points, 0, &mut data);

        for pt in 0..6 {
            let x = *points.get([0, pt]).unwrap();
            let y = *points.get([1, pt]).unwrap();
            assert_relative_eq!(*data.get([0, pt, 0, 0]).unwrap(), 1.0 - x - y);
            assert_relative_eq!(*data.get([0, pt, 1, 0]).unwrap(), x);
            assert_relative_eq!(*data.get([0, pt, 2, 0]).unwrap(), y);
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_0_quadrilateral() {
        let e = lagrange::create::<f64>(
            ReferenceCellType::Quadrilateral,
            0,
            Continuity::Discontinuous,
        );
        assert_eq!(e.value_size(), 1);
        let mut data = rlst_dynamic_array4!(f64, e.tabulate_array_shape(0, 6));
        let mut points = rlst_dynamic_array2!(f64, [2, 6]);
        *points.get_mut([0, 0]).unwrap() = 0.0;
        *points.get_mut([1, 0]).unwrap() = 0.0;
        *points.get_mut([0, 1]).unwrap() = 1.0;
        *points.get_mut([1, 1]).unwrap() = 0.0;
        *points.get_mut([0, 2]).unwrap() = 0.0;
        *points.get_mut([1, 2]).unwrap() = 1.0;
        *points.get_mut([0, 3]).unwrap() = 0.5;
        *points.get_mut([1, 3]).unwrap() = 0.0;
        *points.get_mut([0, 4]).unwrap() = 0.0;
        *points.get_mut([1, 4]).unwrap() = 0.5;
        *points.get_mut([0, 5]).unwrap() = 0.5;
        *points.get_mut([1, 5]).unwrap() = 0.5;
        e.tabulate(&points, 0, &mut data);

        for pt in 0..6 {
            assert_relative_eq!(*data.get([0, pt, 0, 0]).unwrap(), 1.0);
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_1_quadrilateral() {
        let e = lagrange::create::<f64>(ReferenceCellType::Quadrilateral, 1, Continuity::Standard);
        assert_eq!(e.value_size(), 1);
        let mut data = rlst_dynamic_array4!(f64, e.tabulate_array_shape(0, 6));
        let mut points = rlst_dynamic_array2!(f64, [2, 6]);
        *points.get_mut([0, 0]).unwrap() = 0.0;
        *points.get_mut([1, 0]).unwrap() = 0.0;
        *points.get_mut([0, 1]).unwrap() = 1.0;
        *points.get_mut([1, 1]).unwrap() = 0.0;
        *points.get_mut([0, 2]).unwrap() = 0.0;
        *points.get_mut([1, 2]).unwrap() = 1.0;
        *points.get_mut([0, 3]).unwrap() = 1.0;
        *points.get_mut([1, 3]).unwrap() = 1.0;
        *points.get_mut([0, 4]).unwrap() = 0.25;
        *points.get_mut([1, 4]).unwrap() = 0.5;
        *points.get_mut([0, 5]).unwrap() = 0.3;
        *points.get_mut([1, 5]).unwrap() = 0.2;

        e.tabulate(&points, 0, &mut data);

        for pt in 0..6 {
            let x = *points.get([0, pt]).unwrap();
            let y = *points.get([1, pt]).unwrap();
            assert_relative_eq!(*data.get([0, pt, 0, 0]).unwrap(), (1.0 - x) * (1.0 - y));
            assert_relative_eq!(*data.get([0, pt, 1, 0]).unwrap(), x * (1.0 - y));
            assert_relative_eq!(*data.get([0, pt, 2, 0]).unwrap(), (1.0 - x) * y);
            assert_relative_eq!(*data.get([0, pt, 3, 0]).unwrap(), x * y);
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_2_quadrilateral() {
        let e = lagrange::create::<f64>(ReferenceCellType::Quadrilateral, 2, Continuity::Standard);
        assert_eq!(e.value_size(), 1);
        let mut data = rlst_dynamic_array4!(f64, e.tabulate_array_shape(0, 6));
        let mut points = rlst_dynamic_array2!(f64, [2, 6]);
        *points.get_mut([0, 0]).unwrap() = 0.0;
        *points.get_mut([1, 0]).unwrap() = 0.0;
        *points.get_mut([0, 1]).unwrap() = 1.0;
        *points.get_mut([1, 1]).unwrap() = 0.0;
        *points.get_mut([0, 2]).unwrap() = 0.0;
        *points.get_mut([1, 2]).unwrap() = 1.0;
        *points.get_mut([0, 3]).unwrap() = 1.0;
        *points.get_mut([1, 3]).unwrap() = 1.0;
        *points.get_mut([0, 4]).unwrap() = 0.25;
        *points.get_mut([1, 4]).unwrap() = 0.5;
        *points.get_mut([0, 5]).unwrap() = 0.3;
        *points.get_mut([1, 5]).unwrap() = 0.2;

        e.tabulate(&points, 0, &mut data);

        for pt in 0..6 {
            let x = *points.get([0, pt]).unwrap();
            let y = *points.get([1, pt]).unwrap();
            assert_relative_eq!(
                *data.get([0, pt, 0, 0]).unwrap(),
                (1.0 - x) * (1.0 - 2.0 * x) * (1.0 - y) * (1.0 - 2.0 * y),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 1, 0]).unwrap(),
                x * (2.0 * x - 1.0) * (1.0 - y) * (1.0 - 2.0 * y),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 2, 0]).unwrap(),
                (1.0 - x) * (1.0 - 2.0 * x) * y * (2.0 * y - 1.0),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 3, 0]).unwrap(),
                x * (2.0 * x - 1.0) * y * (2.0 * y - 1.0),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 4, 0]).unwrap(),
                4.0 * x * (1.0 - x) * (1.0 - y) * (1.0 - 2.0 * y),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 5, 0]).unwrap(),
                (1.0 - x) * (1.0 - 2.0 * x) * 4.0 * y * (1.0 - y),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 6, 0]).unwrap(),
                x * (2.0 * x - 1.0) * 4.0 * y * (1.0 - y),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 7, 0]).unwrap(),
                4.0 * x * (1.0 - x) * y * (2.0 * y - 1.0),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 8, 0]).unwrap(),
                4.0 * x * (1.0 - x) * 4.0 * y * (1.0 - y),
                epsilon = 1e-14
            );
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_0_tetrahedron() {
        let e =
            lagrange::create::<f64>(ReferenceCellType::Tetrahedron, 0, Continuity::Discontinuous);
        assert_eq!(e.value_size(), 1);
        let mut data = rlst_dynamic_array4!(f64, e.tabulate_array_shape(0, 6));
        let mut points = rlst_dynamic_array2!(f64, [3, 6]);
        *points.get_mut([0, 0]).unwrap() = 0.0;
        *points.get_mut([1, 0]).unwrap() = 0.0;
        *points.get_mut([2, 0]).unwrap() = 0.0;
        *points.get_mut([0, 1]).unwrap() = 1.0;
        *points.get_mut([1, 1]).unwrap() = 0.0;
        *points.get_mut([2, 1]).unwrap() = 0.0;
        *points.get_mut([0, 2]).unwrap() = 0.0;
        *points.get_mut([1, 2]).unwrap() = 1.0;
        *points.get_mut([2, 2]).unwrap() = 0.0;
        *points.get_mut([0, 3]).unwrap() = 0.5;
        *points.get_mut([1, 3]).unwrap() = 0.0;
        *points.get_mut([2, 3]).unwrap() = 0.5;
        *points.get_mut([0, 4]).unwrap() = 0.0;
        *points.get_mut([1, 4]).unwrap() = 0.5;
        *points.get_mut([2, 4]).unwrap() = 0.5;
        *points.get_mut([0, 5]).unwrap() = 0.5;
        *points.get_mut([1, 5]).unwrap() = 0.2;
        *points.get_mut([2, 5]).unwrap() = 0.3;
        e.tabulate(&points, 0, &mut data);

        for pt in 0..6 {
            assert_relative_eq!(*data.get([0, pt, 0, 0]).unwrap(), 1.0);
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_1_tetrahedron() {
        let e = lagrange::create::<f64>(ReferenceCellType::Tetrahedron, 1, Continuity::Standard);
        assert_eq!(e.value_size(), 1);
        let mut data = rlst_dynamic_array4!(f64, e.tabulate_array_shape(0, 6));
        let mut points = rlst_dynamic_array2!(f64, [3, 6]);
        *points.get_mut([0, 0]).unwrap() = 0.0;
        *points.get_mut([1, 0]).unwrap() = 0.0;
        *points.get_mut([2, 0]).unwrap() = 0.0;
        *points.get_mut([0, 1]).unwrap() = 1.0;
        *points.get_mut([1, 1]).unwrap() = 0.0;
        *points.get_mut([2, 1]).unwrap() = 0.0;
        *points.get_mut([0, 2]).unwrap() = 0.0;
        *points.get_mut([1, 2]).unwrap() = 0.8;
        *points.get_mut([2, 2]).unwrap() = 0.2;
        *points.get_mut([0, 3]).unwrap() = 0.0;
        *points.get_mut([1, 3]).unwrap() = 0.0;
        *points.get_mut([2, 3]).unwrap() = 0.8;
        *points.get_mut([0, 4]).unwrap() = 0.25;
        *points.get_mut([1, 4]).unwrap() = 0.5;
        *points.get_mut([2, 4]).unwrap() = 0.1;
        *points.get_mut([0, 5]).unwrap() = 0.2;
        *points.get_mut([1, 5]).unwrap() = 0.1;
        *points.get_mut([2, 5]).unwrap() = 0.15;

        e.tabulate(&points, 0, &mut data);

        for pt in 0..6 {
            let x = *points.get([0, pt]).unwrap();
            let y = *points.get([1, pt]).unwrap();
            let z = *points.get([2, pt]).unwrap();
            assert_relative_eq!(
                *data.get([0, pt, 0, 0]).unwrap(),
                1.0 - x - y - z,
                epsilon = 1e-14
            );
            assert_relative_eq!(*data.get([0, pt, 1, 0]).unwrap(), x, epsilon = 1e-14);
            assert_relative_eq!(*data.get([0, pt, 2, 0]).unwrap(), y, epsilon = 1e-14);
            assert_relative_eq!(*data.get([0, pt, 3, 0]).unwrap(), z, epsilon = 1e-14);
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_0_hexahedron() {
        let e =
            lagrange::create::<f64>(ReferenceCellType::Hexahedron, 0, Continuity::Discontinuous);
        assert_eq!(e.value_size(), 1);
        let mut data = rlst_dynamic_array4!(f64, e.tabulate_array_shape(0, 6));
        let mut points = rlst_dynamic_array2!(f64, [3, 6]);
        *points.get_mut([0, 0]).unwrap() = 0.0;
        *points.get_mut([1, 0]).unwrap() = 0.0;
        *points.get_mut([2, 0]).unwrap() = 0.0;
        *points.get_mut([0, 1]).unwrap() = 1.0;
        *points.get_mut([1, 1]).unwrap() = 0.0;
        *points.get_mut([2, 1]).unwrap() = 0.0;
        *points.get_mut([0, 2]).unwrap() = 0.0;
        *points.get_mut([1, 2]).unwrap() = 1.0;
        *points.get_mut([2, 2]).unwrap() = 0.0;
        *points.get_mut([0, 3]).unwrap() = 0.5;
        *points.get_mut([1, 3]).unwrap() = 0.0;
        *points.get_mut([2, 3]).unwrap() = 0.5;
        *points.get_mut([0, 4]).unwrap() = 0.0;
        *points.get_mut([1, 4]).unwrap() = 0.5;
        *points.get_mut([2, 4]).unwrap() = 0.5;
        *points.get_mut([0, 5]).unwrap() = 0.5;
        *points.get_mut([1, 5]).unwrap() = 0.5;
        *points.get_mut([2, 5]).unwrap() = 0.5;
        e.tabulate(&points, 0, &mut data);

        for pt in 0..6 {
            assert_relative_eq!(*data.get([0, pt, 0, 0]).unwrap(), 1.0);
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_1_hexahedron() {
        let e = lagrange::create::<f64>(ReferenceCellType::Hexahedron, 1, Continuity::Standard);
        assert_eq!(e.value_size(), 1);
        let mut data = rlst_dynamic_array4!(f64, e.tabulate_array_shape(0, 6));
        let mut points = rlst_dynamic_array2!(f64, [3, 6]);
        *points.get_mut([0, 0]).unwrap() = 0.0;
        *points.get_mut([1, 0]).unwrap() = 0.0;
        *points.get_mut([2, 0]).unwrap() = 0.0;
        *points.get_mut([0, 1]).unwrap() = 1.0;
        *points.get_mut([1, 1]).unwrap() = 0.0;
        *points.get_mut([2, 1]).unwrap() = 0.0;
        *points.get_mut([0, 2]).unwrap() = 0.0;
        *points.get_mut([1, 2]).unwrap() = 1.0;
        *points.get_mut([2, 2]).unwrap() = 1.0;
        *points.get_mut([0, 3]).unwrap() = 1.0;
        *points.get_mut([1, 3]).unwrap() = 1.0;
        *points.get_mut([2, 3]).unwrap() = 1.0;
        *points.get_mut([0, 4]).unwrap() = 0.25;
        *points.get_mut([1, 4]).unwrap() = 0.5;
        *points.get_mut([2, 4]).unwrap() = 0.1;
        *points.get_mut([0, 5]).unwrap() = 0.3;
        *points.get_mut([1, 5]).unwrap() = 0.2;
        *points.get_mut([2, 5]).unwrap() = 0.4;

        e.tabulate(&points, 0, &mut data);

        for pt in 0..6 {
            let x = *points.get([0, pt]).unwrap();
            let y = *points.get([1, pt]).unwrap();
            let z = *points.get([2, pt]).unwrap();
            assert_relative_eq!(
                *data.get([0, pt, 0, 0]).unwrap(),
                (1.0 - x) * (1.0 - y) * (1.0 - z),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 1, 0]).unwrap(),
                x * (1.0 - y) * (1.0 - z),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 2, 0]).unwrap(),
                (1.0 - x) * y * (1.0 - z),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 3, 0]).unwrap(),
                x * y * (1.0 - z),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 4, 0]).unwrap(),
                (1.0 - x) * (1.0 - y) * z,
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 5, 0]).unwrap(),
                x * (1.0 - y) * z,
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 6, 0]).unwrap(),
                (1.0 - x) * y * z,
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 7, 0]).unwrap(),
                x * y * z,
                epsilon = 1e-14
            );
        }
        check_dofs(e);
    }

    #[test]
    fn test_lagrange_higher_degree_triangle() {
        lagrange::create::<f64>(ReferenceCellType::Triangle, 2, Continuity::Standard);
        lagrange::create::<f64>(ReferenceCellType::Triangle, 3, Continuity::Standard);
        lagrange::create::<f64>(ReferenceCellType::Triangle, 4, Continuity::Standard);
        lagrange::create::<f64>(ReferenceCellType::Triangle, 5, Continuity::Standard);

        lagrange::create::<f64>(ReferenceCellType::Triangle, 2, Continuity::Discontinuous);
        lagrange::create::<f64>(ReferenceCellType::Triangle, 3, Continuity::Discontinuous);
        lagrange::create::<f64>(ReferenceCellType::Triangle, 4, Continuity::Discontinuous);
        lagrange::create::<f64>(ReferenceCellType::Triangle, 5, Continuity::Discontinuous);
    }

    #[test]
    fn test_lagrange_higher_degree_interval() {
        lagrange::create::<f64>(ReferenceCellType::Interval, 2, Continuity::Standard);
        lagrange::create::<f64>(ReferenceCellType::Interval, 3, Continuity::Standard);
        lagrange::create::<f64>(ReferenceCellType::Interval, 4, Continuity::Standard);
        lagrange::create::<f64>(ReferenceCellType::Interval, 5, Continuity::Standard);

        lagrange::create::<f64>(ReferenceCellType::Interval, 2, Continuity::Discontinuous);
        lagrange::create::<f64>(ReferenceCellType::Interval, 3, Continuity::Discontinuous);
        lagrange::create::<f64>(ReferenceCellType::Interval, 4, Continuity::Discontinuous);
        lagrange::create::<f64>(ReferenceCellType::Interval, 5, Continuity::Discontinuous);
    }

    #[test]
    fn test_lagrange_higher_degree_quadrilateral() {
        lagrange::create::<f64>(ReferenceCellType::Quadrilateral, 2, Continuity::Standard);
        lagrange::create::<f64>(ReferenceCellType::Quadrilateral, 3, Continuity::Standard);
        lagrange::create::<f64>(ReferenceCellType::Quadrilateral, 4, Continuity::Standard);
        lagrange::create::<f64>(ReferenceCellType::Quadrilateral, 5, Continuity::Standard);

        lagrange::create::<f64>(
            ReferenceCellType::Quadrilateral,
            2,
            Continuity::Discontinuous,
        );
        lagrange::create::<f64>(
            ReferenceCellType::Quadrilateral,
            3,
            Continuity::Discontinuous,
        );
        lagrange::create::<f64>(
            ReferenceCellType::Quadrilateral,
            4,
            Continuity::Discontinuous,
        );
        lagrange::create::<f64>(
            ReferenceCellType::Quadrilateral,
            5,
            Continuity::Discontinuous,
        );
    }

    #[test]
    fn test_raviart_thomas_1_triangle() {
        let e = raviart_thomas::create(ReferenceCellType::Triangle, 1, Continuity::Standard);
        assert_eq!(e.value_size(), 2);
        let mut data = rlst_dynamic_array4!(f64, e.tabulate_array_shape(0, 6));
        let mut points = rlst_dynamic_array2!(f64, [2, 6]);
        *points.get_mut([0, 0]).unwrap() = 0.0;
        *points.get_mut([1, 0]).unwrap() = 0.0;
        *points.get_mut([0, 1]).unwrap() = 1.0;
        *points.get_mut([1, 1]).unwrap() = 0.0;
        *points.get_mut([0, 2]).unwrap() = 0.0;
        *points.get_mut([1, 2]).unwrap() = 1.0;
        *points.get_mut([0, 3]).unwrap() = 0.5;
        *points.get_mut([1, 3]).unwrap() = 0.0;
        *points.get_mut([0, 4]).unwrap() = 0.0;
        *points.get_mut([1, 4]).unwrap() = 0.5;
        *points.get_mut([0, 5]).unwrap() = 0.5;
        *points.get_mut([1, 5]).unwrap() = 0.5;
        e.tabulate(&points, 0, &mut data);

        for pt in 0..6 {
            let x = *points.get([0, pt]).unwrap();
            let y = *points.get([1, pt]).unwrap();
            for (i, basis_f) in [[-x, -y], [x - 1.0, y], [-x, 1.0 - y]].iter().enumerate() {
                for (d, value) in basis_f.iter().enumerate() {
                    assert_relative_eq!(*data.get([0, pt, i, d]).unwrap(), value, epsilon = 1e-14);
                }
            }
        }
        check_dofs(e);
    }

    #[test]
    fn test_raviart_thomas_2_triangle() {
        let e = raviart_thomas::create::<f64>(ReferenceCellType::Triangle, 2, Continuity::Standard);
        assert_eq!(e.value_size(), 2);
        check_dofs(e);
    }

    #[test]
    fn test_raviart_thomas_3_triangle() {
        let e = raviart_thomas::create::<f64>(ReferenceCellType::Triangle, 3, Continuity::Standard);
        assert_eq!(e.value_size(), 2);
        check_dofs(e);
    }

    #[test]
    fn test_raviart_thomas_1_quadrilateral() {
        let e = raviart_thomas::create(ReferenceCellType::Quadrilateral, 1, Continuity::Standard);
        assert_eq!(e.value_size(), 2);
        let mut data = rlst_dynamic_array4!(f64, e.tabulate_array_shape(0, 6));
        let mut points = rlst_dynamic_array2!(f64, [2, 6]);
        *points.get_mut([0, 0]).unwrap() = 0.0;
        *points.get_mut([1, 0]).unwrap() = 0.0;
        *points.get_mut([0, 1]).unwrap() = 1.0;
        *points.get_mut([1, 1]).unwrap() = 0.0;
        *points.get_mut([0, 2]).unwrap() = 0.0;
        *points.get_mut([1, 2]).unwrap() = 1.0;
        *points.get_mut([0, 3]).unwrap() = 0.5;
        *points.get_mut([1, 3]).unwrap() = 0.0;
        *points.get_mut([0, 4]).unwrap() = 1.0;
        *points.get_mut([1, 4]).unwrap() = 0.5;
        *points.get_mut([0, 5]).unwrap() = 0.5;
        *points.get_mut([1, 5]).unwrap() = 0.5;
        e.tabulate(&points, 0, &mut data);

        for pt in 0..6 {
            let x = *points.get([0, pt]).unwrap();
            let y = *points.get([1, pt]).unwrap();
            for (i, basis_f) in [[0.0, 1.0 - y], [x - 1.0, 0.0], [-x, 0.0], [0.0, y]]
                .iter()
                .enumerate()
            {
                for (d, value) in basis_f.iter().enumerate() {
                    assert_relative_eq!(*data.get([0, pt, i, d]).unwrap(), value, epsilon = 1e-14);
                }
            }
        }
        check_dofs(e);
    }

    #[test]
    fn test_raviart_thomas_2_quadrilateral() {
        let e = raviart_thomas::create::<f64>(
            ReferenceCellType::Quadrilateral,
            2,
            Continuity::Standard,
        );
        assert_eq!(e.value_size(), 2);
        check_dofs(e);
    }

    #[test]
    fn test_raviart_thomas_3_quadrilateral() {
        let e = raviart_thomas::create::<f64>(
            ReferenceCellType::Quadrilateral,
            3,
            Continuity::Standard,
        );
        assert_eq!(e.value_size(), 2);
        check_dofs(e);
    }

    #[test]
    fn test_raviart_thomas_1_tetrahedron() {
        let e =
            raviart_thomas::create::<f64>(ReferenceCellType::Tetrahedron, 1, Continuity::Standard);
        assert_eq!(e.value_size(), 3);
        check_dofs(e);
    }

    #[test]
    fn test_raviart_thomas_2_tetrahedron() {
        let e =
            raviart_thomas::create::<f64>(ReferenceCellType::Tetrahedron, 2, Continuity::Standard);
        assert_eq!(e.value_size(), 3);
        check_dofs(e);
    }

    #[test]
    fn test_raviart_thomas_3_tetrahedron() {
        let e =
            raviart_thomas::create::<f64>(ReferenceCellType::Tetrahedron, 3, Continuity::Standard);
        assert_eq!(e.value_size(), 3);
        check_dofs(e);
    }

    #[test]
    fn test_raviart_thomas_1_hexahedron() {
        let e = raviart_thomas::create(ReferenceCellType::Hexahedron, 1, Continuity::Standard);
        assert_eq!(e.value_size(), 3);
        let mut data = rlst_dynamic_array4!(f64, e.tabulate_array_shape(0, 6));
        let mut points = rlst_dynamic_array2!(f64, [3, 6]);
        *points.get_mut([0, 0]).unwrap() = 0.0;
        *points.get_mut([1, 0]).unwrap() = 0.0;
        *points.get_mut([2, 0]).unwrap() = 0.0;
        *points.get_mut([0, 1]).unwrap() = 1.0;
        *points.get_mut([1, 1]).unwrap() = 0.0;
        *points.get_mut([2, 1]).unwrap() = 0.8;
        *points.get_mut([0, 2]).unwrap() = 0.0;
        *points.get_mut([1, 2]).unwrap() = 1.0;
        *points.get_mut([2, 2]).unwrap() = 1.0;
        *points.get_mut([0, 3]).unwrap() = 0.5;
        *points.get_mut([1, 3]).unwrap() = 0.0;
        *points.get_mut([2, 3]).unwrap() = 0.1;
        *points.get_mut([0, 4]).unwrap() = 1.0;
        *points.get_mut([1, 4]).unwrap() = 0.5;
        *points.get_mut([2, 4]).unwrap() = 0.5;
        *points.get_mut([0, 5]).unwrap() = 0.5;
        *points.get_mut([1, 5]).unwrap() = 0.5;
        *points.get_mut([2, 5]).unwrap() = 1.0;
        e.tabulate(&points, 0, &mut data);

        for pt in 0..6 {
            let x = *points.get([0, pt]).unwrap();
            let y = *points.get([1, pt]).unwrap();
            let z = *points.get([2, pt]).unwrap();
            for (i, basis_f) in [
                [0.0, 0.0, 1.0 - z],
                [0.0, y - 1.0, 0.0],
                [1.0 - x, 0.0, 0.0],
                [x, 0.0, 0.0],
                [0.0, -y, 0.0],
                [0.0, 0.0, z],
            ]
            .iter()
            .enumerate()
            {
                for (d, value) in basis_f.iter().enumerate() {
                    assert_relative_eq!(*data.get([0, pt, i, d]).unwrap(), value, epsilon = 1e-14);
                }
            }
        }
        check_dofs(e);
    }

    #[test]
    fn test_raviart_thomas_2_hexahedron() {
        let e =
            raviart_thomas::create::<f64>(ReferenceCellType::Hexahedron, 2, Continuity::Standard);
        assert_eq!(e.value_size(), 3);
        check_dofs(e);
    }

    #[test]
    fn test_nedelec_1_triangle() {
        let e = nedelec::create(ReferenceCellType::Triangle, 1, Continuity::Standard);
        assert_eq!(e.value_size(), 2);
        let mut data = rlst_dynamic_array4!(f64, e.tabulate_array_shape(0, 6));
        let mut points = rlst_dynamic_array2!(f64, [2, 6]);
        *points.get_mut([0, 0]).unwrap() = 0.0;
        *points.get_mut([1, 0]).unwrap() = 0.0;
        *points.get_mut([0, 1]).unwrap() = 1.0;
        *points.get_mut([1, 1]).unwrap() = 0.0;
        *points.get_mut([0, 2]).unwrap() = 0.0;
        *points.get_mut([1, 2]).unwrap() = 1.0;
        *points.get_mut([0, 3]).unwrap() = 0.5;
        *points.get_mut([1, 3]).unwrap() = 0.0;
        *points.get_mut([0, 4]).unwrap() = 0.0;
        *points.get_mut([1, 4]).unwrap() = 0.5;
        *points.get_mut([0, 5]).unwrap() = 0.5;
        *points.get_mut([1, 5]).unwrap() = 0.5;
        e.tabulate(&points, 0, &mut data);

        for pt in 0..6 {
            let x = *points.get([0, pt]).unwrap();
            let y = *points.get([1, pt]).unwrap();
            for (i, basis_f) in [[-y, x], [y, 1.0 - x], [1.0 - y, x]].iter().enumerate() {
                for (d, value) in basis_f.iter().enumerate() {
                    assert_relative_eq!(*data.get([0, pt, i, d]).unwrap(), value, epsilon = 1e-14);
                }
            }
        }
        check_dofs(e);
    }

    #[test]
    fn test_nedelec_2_triangle() {
        let e = nedelec::create::<f64>(ReferenceCellType::Triangle, 2, Continuity::Standard);
        assert_eq!(e.value_size(), 2);
        check_dofs(e);
    }

    #[test]
    fn test_nedelec_3_triangle() {
        let e = nedelec::create::<f64>(ReferenceCellType::Triangle, 3, Continuity::Standard);
        assert_eq!(e.value_size(), 2);
        check_dofs(e);
    }

    #[test]
    fn test_nedelec_1_quadrilateral() {
        let e = nedelec::create(ReferenceCellType::Quadrilateral, 1, Continuity::Standard);
        assert_eq!(e.value_size(), 2);
        let mut data = rlst_dynamic_array4!(f64, e.tabulate_array_shape(0, 6));
        let mut points = rlst_dynamic_array2!(f64, [2, 6]);
        *points.get_mut([0, 0]).unwrap() = 0.0;
        *points.get_mut([1, 0]).unwrap() = 0.0;
        *points.get_mut([0, 1]).unwrap() = 1.0;
        *points.get_mut([1, 1]).unwrap() = 0.0;
        *points.get_mut([0, 2]).unwrap() = 0.0;
        *points.get_mut([1, 2]).unwrap() = 1.0;
        *points.get_mut([0, 3]).unwrap() = 0.5;
        *points.get_mut([1, 3]).unwrap() = 0.0;
        *points.get_mut([0, 4]).unwrap() = 1.0;
        *points.get_mut([1, 4]).unwrap() = 0.5;
        *points.get_mut([0, 5]).unwrap() = 0.5;
        *points.get_mut([1, 5]).unwrap() = 0.5;
        e.tabulate(&points, 0, &mut data);

        for pt in 0..6 {
            let x = *points.get([0, pt]).unwrap();
            let y = *points.get([1, pt]).unwrap();
            for (i, basis_f) in [[1.0 - y, 0.0], [0.0, 1.0 - x], [0.0, x], [y, 0.0]]
                .iter()
                .enumerate()
            {
                for (d, value) in basis_f.iter().enumerate() {
                    assert_relative_eq!(*data.get([0, pt, i, d]).unwrap(), value, epsilon = 1e-14);
                }
            }
        }
        check_dofs(e);
    }

    #[test]
    fn test_nedelec_2_quadrilateral() {
        let e = nedelec::create::<f64>(ReferenceCellType::Quadrilateral, 2, Continuity::Standard);
        assert_eq!(e.value_size(), 2);
        check_dofs(e);
    }

    #[test]
    fn test_nedelec_3_quadrilateral() {
        let e = nedelec::create::<f64>(ReferenceCellType::Quadrilateral, 3, Continuity::Standard);
        assert_eq!(e.value_size(), 2);
        check_dofs(e);
    }

    #[test]
    fn test_nedelec_1_tetrahedron() {
        let e = nedelec::create::<f64>(ReferenceCellType::Tetrahedron, 1, Continuity::Standard);
        assert_eq!(e.value_size(), 3);
        check_dofs(e);
    }

    #[test]
    fn test_nedelec_2_tetrahedron() {
        let e = nedelec::create::<f64>(ReferenceCellType::Tetrahedron, 2, Continuity::Standard);
        assert_eq!(e.value_size(), 3);
        check_dofs(e);
    }

    #[test]
    fn test_nedelec_3_tetrahedron() {
        let e = nedelec::create::<f64>(ReferenceCellType::Tetrahedron, 3, Continuity::Standard);
        assert_eq!(e.value_size(), 3);
        check_dofs(e);
    }

    #[test]
    fn test_nedelec_1_hexahedron() {
        let e = nedelec::create(ReferenceCellType::Hexahedron, 1, Continuity::Standard);
        assert_eq!(e.value_size(), 3);
        let mut data = rlst_dynamic_array4!(f64, e.tabulate_array_shape(0, 6));
        let mut points = rlst_dynamic_array2!(f64, [3, 6]);
        *points.get_mut([0, 0]).unwrap() = 0.0;
        *points.get_mut([1, 0]).unwrap() = 0.0;
        *points.get_mut([2, 0]).unwrap() = 0.0;
        *points.get_mut([0, 1]).unwrap() = 1.0;
        *points.get_mut([1, 1]).unwrap() = 0.0;
        *points.get_mut([2, 1]).unwrap() = 0.8;
        *points.get_mut([0, 2]).unwrap() = 0.0;
        *points.get_mut([1, 2]).unwrap() = 1.0;
        *points.get_mut([2, 2]).unwrap() = 1.0;
        *points.get_mut([0, 3]).unwrap() = 0.5;
        *points.get_mut([1, 3]).unwrap() = 0.0;
        *points.get_mut([2, 3]).unwrap() = 0.1;
        *points.get_mut([0, 4]).unwrap() = 1.0;
        *points.get_mut([1, 4]).unwrap() = 0.5;
        *points.get_mut([2, 4]).unwrap() = 0.5;
        *points.get_mut([0, 5]).unwrap() = 0.5;
        *points.get_mut([1, 5]).unwrap() = 0.5;
        *points.get_mut([2, 5]).unwrap() = 1.0;
        e.tabulate(&points, 0, &mut data);

        for pt in 0..6 {
            let x = *points.get([0, pt]).unwrap();
            let y = *points.get([1, pt]).unwrap();
            let z = *points.get([2, pt]).unwrap();
            for (i, basis_f) in [
                [(1.0 - y) * (1.0 - z), 0.0, 0.0],
                [0.0, (1.0 - x) * (1.0 - z), 0.0],
                [0.0, 0.0, (1.0 - x) * (1.0 - y)],
                [0.0, x * (1.0 - z), 0.0],
                [0.0, 0.0, x * (1.0 - y)],
                [y * (1.0 - z), 0.0, 0.0],
                [0.0, 0.0, (1.0 - x) * y],
                [0.0, 0.0, x * y],
                [(1.0 - y) * z, 0.0, 0.0],
                [0.0, (1.0 - x) * z, 0.0],
                [0.0, x * z, 0.0],
                [y * z, 0.0, 0.0],
            ]
            .iter()
            .enumerate()
            {
                for (d, value) in basis_f.iter().enumerate() {
                    assert_relative_eq!(*data.get([0, pt, i, d]).unwrap(), value, epsilon = 1e-14);
                }
            }
        }
        check_dofs(e);
    }

    #[test]
    fn test_nedelec_2_hexahedron() {
        let e = nedelec::create::<f64>(ReferenceCellType::Hexahedron, 2, Continuity::Standard);
        assert_eq!(e.value_size(), 3);
        check_dofs(e);
    }

    macro_rules! test_entity_closure_dofs_lagrange {
        ($cell:ident, $degree:expr) => {
            paste! {
                #[test]
                fn [<test_entity_closure_dofs_ $cell:lower _ $degree>]() {
                    let e = lagrange::create::<f64>(ReferenceCellType::[<$cell>], [<$degree>], Continuity::Standard);
                    let c = reference_cell::connectivity(ReferenceCellType::[<$cell>]);

                    for (dim, entities) in c.iter().enumerate() {
                        for (n, entity) in entities.iter().enumerate() {
                            let ecd = e.entity_closure_dofs(dim, n).unwrap();
                            let mut len = 0;
                            for (sub_dim, sub_entities) in entity.iter().take(dim + 1).enumerate() {
                                for sub_entity in sub_entities {
                                    let dofs = e.entity_dofs(sub_dim, *sub_entity).unwrap();
                                    len += dofs.len();
                                    for dof in dofs {
                                        assert!(ecd.contains(dof));
                                    }
                                }
                            }
                            assert_eq!(ecd.len(), len);
                        }
                    }
                }
            }
        };
    }

    test_entity_closure_dofs_lagrange!(Interval, 2);
    test_entity_closure_dofs_lagrange!(Interval, 3);
    test_entity_closure_dofs_lagrange!(Interval, 4);
    test_entity_closure_dofs_lagrange!(Interval, 5);
    test_entity_closure_dofs_lagrange!(Triangle, 2);
    test_entity_closure_dofs_lagrange!(Triangle, 3);
    test_entity_closure_dofs_lagrange!(Triangle, 4);
    test_entity_closure_dofs_lagrange!(Triangle, 5);
    test_entity_closure_dofs_lagrange!(Quadrilateral, 2);
    test_entity_closure_dofs_lagrange!(Quadrilateral, 3);
    test_entity_closure_dofs_lagrange!(Quadrilateral, 4);
    test_entity_closure_dofs_lagrange!(Quadrilateral, 5);
    test_entity_closure_dofs_lagrange!(Tetrahedron, 2);
    test_entity_closure_dofs_lagrange!(Tetrahedron, 3);
    test_entity_closure_dofs_lagrange!(Tetrahedron, 4);
    test_entity_closure_dofs_lagrange!(Tetrahedron, 5);
    test_entity_closure_dofs_lagrange!(Hexahedron, 2);
    test_entity_closure_dofs_lagrange!(Hexahedron, 3);

    macro_rules! test_dof_transformations {
        ($cell:ident, $element:ident, $degree:expr) => {
            paste! {

                #[test]
                fn [<test_dof_transformations_ $cell:lower _ $element:lower _ $degree>]() {
                    let e = [<$element>]::create::<f64>(ReferenceCellType::[<$cell>], [<$degree>], Continuity::Standard);
                    let tdim = reference_cell::dim(ReferenceCellType::[<$cell>]);
                    for edim in 1..tdim {
                        for entity in &reference_cell::entity_types(ReferenceCellType::[<$cell>])[edim] {
                            for t in match edim {
                                1 => vec![Transformation::Reflection],
                                2 => vec![Transformation::Reflection, Transformation::Rotation],
                                _ => { panic!("Shouldn't test this dimension"); },
                            } {
                                let order = match t {
                                    Transformation::Reflection => 2,
                                    Transformation::Rotation => match entity {
                                        ReferenceCellType::Triangle => 3,
                                        ReferenceCellType::Quadrilateral => 4,
                                        _ => {
                                            panic!("Unsupported entity type: {entity:?}");
                                        }
                                    },
                                };
                                match e.dof_transformation(*entity, t).unwrap() {
                                    DofTransformation::Identity => {}
                                    DofTransformation::Permutation(p) => {
                                        let mut data = (0..p.len()).collect::<Vec<_>>();
                                        for _ in 0..order {
                                            math::apply_permutation(p, &mut data);
                                        }
                                        for (i, j) in data.iter().enumerate() {
                                            assert_eq!(i, *j);
                                        }
                                    }
                                    DofTransformation::Transformation(m, p) => {
                                        let mut data = (0..p.len()).map(|i| i as f64).collect::<Vec<_>>();
                                        for _ in 0..order {
                                            math::apply_perm_and_matrix(m, p, &mut data);
                                        }
                                        for (i, j) in data.iter().enumerate() {
                                            assert_relative_eq!(i as f64, j, epsilon=1e-10);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        };
    }

    test_dof_transformations!(Triangle, lagrange, 1);
    test_dof_transformations!(Triangle, lagrange, 2);
    test_dof_transformations!(Triangle, lagrange, 3);
    test_dof_transformations!(Quadrilateral, lagrange, 1);
    test_dof_transformations!(Quadrilateral, lagrange, 2);
    test_dof_transformations!(Quadrilateral, lagrange, 3);
    test_dof_transformations!(Tetrahedron, lagrange, 1);
    test_dof_transformations!(Tetrahedron, lagrange, 2);
    test_dof_transformations!(Tetrahedron, lagrange, 3);
    test_dof_transformations!(Hexahedron, lagrange, 1);
    test_dof_transformations!(Hexahedron, lagrange, 2);
    test_dof_transformations!(Hexahedron, lagrange, 3);
    test_dof_transformations!(Triangle, nedelec, 1);
    test_dof_transformations!(Triangle, nedelec, 2);
    test_dof_transformations!(Triangle, nedelec, 3);
    test_dof_transformations!(Quadrilateral, nedelec, 1);
    test_dof_transformations!(Quadrilateral, nedelec, 2);
    test_dof_transformations!(Quadrilateral, nedelec, 3);
    test_dof_transformations!(Tetrahedron, nedelec, 1);
    test_dof_transformations!(Tetrahedron, nedelec, 2);
    test_dof_transformations!(Tetrahedron, nedelec, 3);
    test_dof_transformations!(Hexahedron, nedelec, 1);
    test_dof_transformations!(Hexahedron, nedelec, 2);
    test_dof_transformations!(Triangle, raviart_thomas, 1);
    test_dof_transformations!(Triangle, raviart_thomas, 2);
    test_dof_transformations!(Triangle, raviart_thomas, 3);
    test_dof_transformations!(Quadrilateral, raviart_thomas, 1);
    test_dof_transformations!(Quadrilateral, raviart_thomas, 2);
    test_dof_transformations!(Quadrilateral, raviart_thomas, 3);
    test_dof_transformations!(Tetrahedron, raviart_thomas, 1);
    test_dof_transformations!(Tetrahedron, raviart_thomas, 2);
    test_dof_transformations!(Tetrahedron, raviart_thomas, 3);
    test_dof_transformations!(Hexahedron, raviart_thomas, 1);
    test_dof_transformations!(Hexahedron, raviart_thomas, 2);

    fn assert_dof_transformation_equal<Array2: RandomAccessByRef<2, Item = f64> + Shape<2>>(
        p: &[usize],
        m: &Array2,
        expected: Vec<Vec<f64>>,
    ) {
        let ndofs = p.len();
        assert_eq!(m.shape()[0], ndofs);
        assert_eq!(m.shape()[1], ndofs);
        assert_eq!(expected.len(), ndofs);
        for i in 0..ndofs {
            let mut col = vec![0.0; ndofs];
            col[i] = 1.0;
            math::apply_perm_and_matrix(m, p, &mut col);
            for (j, c) in col.iter().enumerate() {
                assert_relative_eq!(expected[j][i], *c, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_nedelec1_triangle_dof_transformations() {
        let e = nedelec::create::<f64>(ReferenceCellType::Triangle, 1, Continuity::Standard);
        if let DofTransformation::Transformation(m, p) = e
            .dof_transformation(ReferenceCellType::Interval, Transformation::Reflection)
            .unwrap()
        {
            assert_eq!(p.len(), 1);
            assert_eq!(m.shape()[0], 1);
            assert_eq!(m.shape()[1], 1);
            assert_dof_transformation_equal(p, m, vec![vec![-1.0]]);
        } else {
            panic!("Should be a linear transformation");
        }
    }

    #[test]
    fn test_nedelec2_triangle_dof_transformations() {
        let e = nedelec::create::<f64>(ReferenceCellType::Triangle, 2, Continuity::Standard);
        if let DofTransformation::Transformation(m, p) = e
            .dof_transformation(ReferenceCellType::Interval, Transformation::Reflection)
            .unwrap()
        {
            assert_eq!(p.len(), 2);
            assert_eq!(m.shape()[0], 2);
            assert_eq!(m.shape()[1], 2);
            assert_dof_transformation_equal(p, m, vec![vec![-1.0, 0.0], vec![0.0, 1.0]]);
        } else {
            panic!("Should be a linear transformation");
        }
    }

    #[test]
    fn test_nedelec1_tetrahedron_dof_transformations() {
        let e = nedelec::create::<f64>(ReferenceCellType::Tetrahedron, 1, Continuity::Standard);
        if let DofTransformation::Transformation(m, p) = e
            .dof_transformation(ReferenceCellType::Interval, Transformation::Reflection)
            .unwrap()
        {
            assert_eq!(p.len(), 1);
            assert_eq!(m.shape()[0], 1);
            assert_eq!(m.shape()[1], 1);
            assert_dof_transformation_equal(p, m, vec![vec![-1.0]]);
        } else {
            panic!("Should be a linear transformation");
        }
        if let DofTransformation::Identity = e
            .dof_transformation(ReferenceCellType::Triangle, Transformation::Reflection)
            .unwrap()
        {
        } else {
            panic!("Should be an identity transformation");
        }
        if let DofTransformation::Identity = e
            .dof_transformation(ReferenceCellType::Triangle, Transformation::Rotation)
            .unwrap()
        {
        } else {
            panic!("Should be an identity transformation");
        }
    }

    #[test]
    fn test_nedelec2_tetrahedron_dof_transformations() {
        let e = nedelec::create::<f64>(ReferenceCellType::Tetrahedron, 2, Continuity::Standard);
        if let DofTransformation::Transformation(m, p) = e
            .dof_transformation(ReferenceCellType::Interval, Transformation::Reflection)
            .unwrap()
        {
            assert_eq!(p.len(), 2);
            assert_eq!(m.shape()[0], 2);
            assert_eq!(m.shape()[1], 2);
            assert_dof_transformation_equal(p, m, vec![vec![-1.0, 0.0], vec![0.0, 1.0]]);
        } else {
            panic!("Should be a linear transformation");
        }
        if let DofTransformation::Permutation(p) = e
            .dof_transformation(ReferenceCellType::Triangle, Transformation::Reflection)
            .unwrap()
        {
            assert_eq!(p.len(), 2);
            let mut indices = vec![0, 1];
            math::apply_permutation(p, &mut indices);
            assert_eq!(indices[0], 1);
            assert_eq!(indices[1], 0);
        } else {
            panic!("Should be a permutation");
        }
        if let DofTransformation::Transformation(m, p) = e
            .dof_transformation(ReferenceCellType::Triangle, Transformation::Rotation)
            .unwrap()
        {
            assert_eq!(p.len(), 2);
            assert_eq!(m.shape()[0], 2);
            assert_eq!(m.shape()[1], 2);
            assert_dof_transformation_equal(p, m, vec![vec![-1.0, -1.0], vec![1.0, 0.0]]);
        } else {
            panic!("Should be a linear transformation");
        }
    }

    #[test]
    fn test_nedelec2_quadrilateral_dof_transformations() {
        let e = nedelec::create::<f64>(ReferenceCellType::Hexahedron, 2, Continuity::Standard);
        if let DofTransformation::Transformation(m, p) = e
            .dof_transformation(ReferenceCellType::Interval, Transformation::Reflection)
            .unwrap()
        {
            assert_eq!(p.len(), 2);
            assert_eq!(m.shape()[0], 2);
            assert_eq!(m.shape()[1], 2);
            assert_dof_transformation_equal(p, m, vec![vec![-1.0, 0.0], vec![0.0, 1.0]]);
        } else {
            panic!("Should be a linear transformation");
        }
    }

    #[test]
    fn test_nedelec2_hexahedron_dof_transformations() {
        let e = nedelec::create::<f64>(ReferenceCellType::Hexahedron, 2, Continuity::Standard);
        if let DofTransformation::Transformation(m, p) = e
            .dof_transformation(ReferenceCellType::Interval, Transformation::Reflection)
            .unwrap()
        {
            assert_eq!(p.len(), 2);
            assert_eq!(m.shape()[0], 2);
            assert_eq!(m.shape()[1], 2);
            assert_dof_transformation_equal(p, m, vec![vec![-1.0, 0.0], vec![0.0, 1.0]]);
        } else {
            panic!("Should be a linear transformation");
        }
        if let DofTransformation::Permutation(p) = e
            .dof_transformation(ReferenceCellType::Quadrilateral, Transformation::Reflection)
            .unwrap()
        {
            assert_eq!(p.len(), 4);
            let mut indices = vec![0, 1, 2, 3];
            math::apply_permutation(p, &mut indices);
            assert_eq!(indices[0], 1);
            assert_eq!(indices[1], 0);
            assert_eq!(indices[2], 3);
            assert_eq!(indices[3], 2);
        } else {
            panic!("Should be a permutation");
        }
        if let DofTransformation::Transformation(m, p) = e
            .dof_transformation(ReferenceCellType::Quadrilateral, Transformation::Rotation)
            .unwrap()
        {
            assert_eq!(p.len(), 4);
            assert_eq!(m.shape()[0], 4);
            assert_eq!(m.shape()[1], 4);
            assert_dof_transformation_equal(
                p,
                m,
                vec![
                    vec![0.0, -1.0, 0.0, 0.0],
                    vec![1.0, 0.0, 0.0, 0.0],
                    vec![0.0, 0.0, 0.0, 1.0],
                    vec![0.0, 0.0, 1.0, 0.0],
                ],
            );
        } else {
            panic!("Should be a linear transformation");
        }
    }

    #[test]
    fn test_lagrange4_tetrahedron_dof_transformations() {
        let e = lagrange::create::<f64>(ReferenceCellType::Tetrahedron, 4, Continuity::Standard);
        if let DofTransformation::Permutation(p) = e
            .dof_transformation(ReferenceCellType::Interval, Transformation::Reflection)
            .unwrap()
        {
            assert_eq!(p.len(), 3);
            let mut indices = vec![0, 1, 2];
            math::apply_permutation(p, &mut indices);
            assert_eq!(indices[0], 2);
            assert_eq!(indices[1], 1);
            assert_eq!(indices[2], 0);
        } else {
            panic!("Should be a permutation");
        }
        if let DofTransformation::Permutation(p) = e
            .dof_transformation(ReferenceCellType::Triangle, Transformation::Reflection)
            .unwrap()
        {
            assert_eq!(p.len(), 3);
            let mut indices = vec![0, 1, 2];
            math::apply_permutation(p, &mut indices);
            assert_eq!(indices[0], 0);
            assert_eq!(indices[1], 2);
            assert_eq!(indices[2], 1);
        } else {
            panic!("Should be a permutation");
        }
        if let DofTransformation::Permutation(p) = e
            .dof_transformation(ReferenceCellType::Triangle, Transformation::Rotation)
            .unwrap()
        {
            assert_eq!(p.len(), 3);
            let mut indices = vec![0, 1, 2];
            math::apply_permutation(p, &mut indices);
            assert_eq!(indices[0], 1);
            assert_eq!(indices[1], 2);
            assert_eq!(indices[2], 0);
        } else {
            panic!("Should be a permutation");
        }
    }
}
