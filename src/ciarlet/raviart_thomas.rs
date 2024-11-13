//! Raviart-Thomas elements

use super::CiarletElement;
use crate::polynomials::polynomial_count;
use crate::reference_cell;
use crate::traits::ElementFamily;
use crate::types::{Continuity, MapType, ReferenceCellType};
use rlst::MatrixInverse;
use rlst::RlstScalar;
use rlst::{rlst_dynamic_array2, rlst_dynamic_array3, RandomAccessMut};
use std::marker::PhantomData;

/// Create a Raviart-Thomas element
pub fn create<T: RlstScalar + MatrixInverse>(
    cell_type: ReferenceCellType,
    degree: usize,
    continuity: Continuity,
) -> CiarletElement<T> {
    if cell_type != ReferenceCellType::Triangle && cell_type != ReferenceCellType::Quadrilateral {
        panic!("Raviart-Thomas only implemented for 2D cells");
    }

    if cell_type != ReferenceCellType::Triangle {
        panic!("RT elements on quadrilaterals not implemented yet");
    }

    let subpdim = polynomial_count(cell_type, degree - 1);
    let pdim = polynomial_count(cell_type, degree);
    let tdim = reference_cell::dim(cell_type);
    let edim = tdim * subpdim + degree;

    let mut wcoeffs = rlst_dynamic_array3!(T, [edim, tdim, pdim]);

    // vector polynomials of degree <= n-1
    for i in 0..tdim {
        for j in 0..subpdim {
            *wcoeffs.get_mut([i * subpdim + j, i, j]).unwrap() = T::from(1.0).unwrap();
        }
    }
    // (px, py, pz) , where p = scalar polynomial of degree = n-1

    *wcoeffs.get_mut([2, 0, 1]).unwrap() = T::from(-0.5).unwrap() / T::sqrt(T::from(2.0).unwrap());
    *wcoeffs.get_mut([2, 0, 2]).unwrap() = T::from(0.5).unwrap() * T::sqrt(T::from(1.5).unwrap());
    *wcoeffs.get_mut([2, 1, 1]).unwrap() = T::from(1.0).unwrap() / T::sqrt(T::from(2.0).unwrap());

    let mut x = [vec![], vec![], vec![], vec![]];
    let mut m = [vec![], vec![], vec![], vec![]];

    let entity_counts = reference_cell::entity_counts(cell_type);
    let vertices = reference_cell::vertices::<T::Real>(cell_type);
    let edges = reference_cell::edges(cell_type);

    for _e in 0..entity_counts[0] {
        x[0].push(rlst_dynamic_array2!(T::Real, [tdim, 0]));
        m[0].push(rlst_dynamic_array3!(T, [0, 2, 0]));
    }

    for e in &edges {
        let mut pts = rlst_dynamic_array2!(T::Real, [tdim, 1]);
        let mut mat = rlst_dynamic_array3!(T, [1, 2, 1]);
        let [vn0, vn1] = e[..] else {
            panic!();
        };
        let v0 = &vertices[vn0];
        let v1 = &vertices[vn1];
        for i in 0..tdim {
            *pts.get_mut([i, 0]).unwrap() = num::cast::<_, T::Real>(v0[i] + v1[i]).unwrap()
                / num::cast::<_, T::Real>(2.0).unwrap();
        }
        *mat.get_mut([0, 0, 0]).unwrap() = T::from(v0[1] - v1[1]).unwrap();
        *mat.get_mut([0, 1, 0]).unwrap() = T::from(v1[0] - v0[0]).unwrap();
        x[1].push(pts);
        m[1].push(mat);
    }

    for _e in 0..entity_counts[2] {
        x[2].push(rlst_dynamic_array2!(T::Real, [tdim, 0]));
        m[2].push(rlst_dynamic_array3!(T, [0, 2, 0]))
    }

    CiarletElement::create(
        "Raviart-Thomas".to_string(),
        cell_type,
        degree,
        vec![2],
        wcoeffs,
        x,
        m,
        MapType::ContravariantPiola,
        continuity,
        degree,
    )
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
    type FiniteElement = CiarletElement<T>;
    fn element(&self, cell_type: ReferenceCellType) -> CiarletElement<T> {
        create::<T>(cell_type, self.degree, self.continuity)
    }
}
