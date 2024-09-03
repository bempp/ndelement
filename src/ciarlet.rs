//! Finite element definitions

use crate::polynomials::{legendre_shape, polynomial_count, tabulate_legendre_polynomials};
use crate::reference_cell;
use crate::traits::FiniteElement;
use crate::types::{Continuity, MapType, ReferenceCellType};
use itertools::izip;
use rlst::{
    rlst_dynamic_array2, rlst_dynamic_array3, Array, BaseArray, MatrixInverse, RandomAccessByRef,
    RandomAccessMut, RlstScalar, Shape, VectorContainer,
};
use std::fmt::{Debug, Formatter};

pub mod lagrange;
pub mod raviart_thomas;
pub use lagrange::LagrangeElementFamily;
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
pub struct CiarletElement<T: RlstScalar + MatrixInverse> {
    family_name: String,
    cell_type: ReferenceCellType,
    degree: usize,
    embedded_superdegree: usize,
    map_type: MapType,
    value_shape: Vec<usize>,
    value_size: usize,
    continuity: Continuity,
    dim: usize,
    coefficients: Array<T, BaseArray<T, VectorContainer<T>, 3>, 3>,
    entity_dofs: [Vec<Vec<usize>>; 4],
    entity_closure_dofs: [Vec<Vec<usize>>; 4],
    interpolation_points: EntityPoints<T::Real>,
    interpolation_weights: EntityWeights<T>,
}

impl<T: RlstScalar + MatrixInverse> Debug for CiarletElement<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        f.debug_tuple("CiarletElement")
            .field(&self.family_name)
            .field(&self.cell_type)
            .field(&self.degree)
            .finish()
    }
}

impl<T: RlstScalar + MatrixInverse> CiarletElement<T> {
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
        map_type: MapType,
        continuity: Continuity,
        embedded_superdegree: usize,
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

        let mut inverse = rlst::rlst_dynamic_array2!(T, [dim, dim]);

        for i in 0..dim {
            for j in 0..dim {
                let entry = inverse.get_mut([i, j]).unwrap();
                *entry = T::from(0.0).unwrap();
                for k in 0..value_size {
                    for l in 0..pdim {
                        *entry += *polynomial_coeffs.get([i, k, l]).unwrap()
                            * *d_matrix.get([k, l, j]).unwrap();
                    }
                }
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

        let mut entity_dofs = [vec![], vec![], vec![], vec![]];
        let mut dof = 0;
        for i in 0..4 {
            for pts in &new_pts[i] {
                let dofs = (dof..dof + pts.shape()[1]).collect::<Vec<_>>();
                entity_dofs[i].push(dofs);
                dof += pts.shape()[1];
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
        CiarletElement::<T> {
            family_name,
            cell_type,
            degree,
            embedded_superdegree,
            map_type,
            value_shape,
            value_size,
            continuity,
            dim,
            coefficients,
            entity_dofs,
            entity_closure_dofs,
            interpolation_points: new_pts,
            interpolation_weights: new_wts,
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
impl<T: RlstScalar + MatrixInverse> FiniteElement for CiarletElement<T> {
    type CellType = ReferenceCellType;
    type MapType = MapType;
    type T = T;
    fn value_shape(&self) -> &[usize] {
        &self.value_shape
    }
    fn value_size(&self) -> usize {
        self.value_size
    }
    fn map_type(&self) -> MapType {
        self.map_type
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
            legendre_shape(self.cell_type, points, self.embedded_superdegree, nderivs,)
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
            assert_relative_eq!(
                *data.get([0, pt, 0, 0]).unwrap(),
                1.0 - *points.get([0, pt]).unwrap()
            );
            assert_relative_eq!(
                *data.get([0, pt, 1, 0]).unwrap(),
                *points.get([0, pt]).unwrap()
            );
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
            assert_relative_eq!(
                *data.get([0, pt, 0, 0]).unwrap(),
                1.0 - *points.get([0, pt]).unwrap() - *points.get([1, pt]).unwrap()
            );
            assert_relative_eq!(
                *data.get([0, pt, 1, 0]).unwrap(),
                *points.get([0, pt]).unwrap()
            );
            assert_relative_eq!(
                *data.get([0, pt, 2, 0]).unwrap(),
                *points.get([1, pt]).unwrap()
            );
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
            assert_relative_eq!(
                *data.get([0, pt, 0, 0]).unwrap(),
                (1.0 - *points.get([0, pt]).unwrap()) * (1.0 - *points.get([1, pt]).unwrap())
            );
            assert_relative_eq!(
                *data.get([0, pt, 1, 0]).unwrap(),
                *points.get([0, pt]).unwrap() * (1.0 - *points.get([1, pt]).unwrap())
            );
            assert_relative_eq!(
                *data.get([0, pt, 2, 0]).unwrap(),
                (1.0 - *points.get([0, pt]).unwrap()) * *points.get([1, pt]).unwrap()
            );
            assert_relative_eq!(
                *data.get([0, pt, 3, 0]).unwrap(),
                *points.get([0, pt]).unwrap() * *points.get([1, pt]).unwrap()
            );
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
            assert_relative_eq!(
                *data.get([0, pt, 0, 0]).unwrap(),
                1.0 - *points.get([0, pt]).unwrap()
                    - *points.get([1, pt]).unwrap()
                    - *points.get([2, pt]).unwrap(),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 1, 0]).unwrap(),
                *points.get([0, pt]).unwrap(),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 2, 0]).unwrap(),
                *points.get([1, pt]).unwrap(),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 3, 0]).unwrap(),
                *points.get([2, pt]).unwrap(),
                epsilon = 1e-14
            );
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
            assert_relative_eq!(
                *data.get([0, pt, 0, 0]).unwrap(),
                (1.0 - *points.get([0, pt]).unwrap())
                    * (1.0 - *points.get([1, pt]).unwrap())
                    * (1.0 - *points.get([2, pt]).unwrap()),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 1, 0]).unwrap(),
                *points.get([0, pt]).unwrap()
                    * (1.0 - *points.get([1, pt]).unwrap())
                    * (1.0 - *points.get([2, pt]).unwrap()),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 2, 0]).unwrap(),
                (1.0 - *points.get([0, pt]).unwrap())
                    * *points.get([1, pt]).unwrap()
                    * (1.0 - *points.get([2, pt]).unwrap()),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 3, 0]).unwrap(),
                *points.get([0, pt]).unwrap()
                    * *points.get([1, pt]).unwrap()
                    * (1.0 - *points.get([2, pt]).unwrap()),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 4, 0]).unwrap(),
                (1.0 - *points.get([0, pt]).unwrap())
                    * (1.0 - *points.get([1, pt]).unwrap())
                    * *points.get([2, pt]).unwrap(),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 5, 0]).unwrap(),
                *points.get([0, pt]).unwrap()
                    * (1.0 - *points.get([1, pt]).unwrap())
                    * *points.get([2, pt]).unwrap(),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 6, 0]).unwrap(),
                (1.0 - *points.get([0, pt]).unwrap())
                    * *points.get([1, pt]).unwrap()
                    * *points.get([2, pt]).unwrap(),
                epsilon = 1e-14
            );
            assert_relative_eq!(
                *data.get([0, pt, 7, 0]).unwrap(),
                *points.get([0, pt]).unwrap()
                    * *points.get([1, pt]).unwrap()
                    * *points.get([2, pt]).unwrap(),
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
            assert_relative_eq!(
                *data.get([0, pt, 0, 0]).unwrap(),
                -*points.get([0, pt]).unwrap()
            );
            assert_relative_eq!(
                *data.get([0, pt, 0, 1]).unwrap(),
                -*points.get([1, pt]).unwrap()
            );
            assert_relative_eq!(
                *data.get([0, pt, 1, 0]).unwrap(),
                *points.get([0, pt]).unwrap() - 1.0
            );
            assert_relative_eq!(
                *data.get([0, pt, 1, 1]).unwrap(),
                *points.get([1, pt]).unwrap()
            );
            assert_relative_eq!(
                *data.get([0, pt, 2, 0]).unwrap(),
                -*points.get([0, pt]).unwrap()
            );
            assert_relative_eq!(
                *data.get([0, pt, 2, 1]).unwrap(),
                1.0 - *points.get([1, pt]).unwrap()
            );
        }
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
}
