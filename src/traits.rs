//! Traits
use rlst::{RandomAccessByRef, RandomAccessMut, RlstScalar, Shape};
use std::fmt::Debug;
use std::hash::Hash;

pub trait FiniteElement {
    //! A finite element defined on a reference cell
    /// The scalar type
    type T: RlstScalar;
    /// Cell type
    type CellType: Debug + PartialEq + Eq + Clone + Copy + Hash;

    /// The reference cell type
    fn cell_type(&self) -> Self::CellType;

    /// The highest degree polynomial in the element's polynomial set
    fn embedded_superdegree(&self) -> usize;

    /// The number of basis functions
    fn dim(&self) -> usize;

    /// The value shape
    fn value_shape(&self) -> &[usize];

    /// The value size
    fn value_size(&self) -> usize;

    /// Tabulate the values of the basis functions and their derivatives at a set of points
    fn tabulate<Array2: RandomAccessByRef<2, Item = <Self::T as RlstScalar>::Real> + Shape<2>>(
        &self,
        points: &Array2,
        nderivs: usize,
        data: &mut impl RandomAccessMut<4, Item = Self::T>,
    );

    /// The DOFs that are associated with a subentity of the reference cell
    fn entity_dofs(&self, entity_dim: usize, entity_number: usize) -> Option<&[usize]>;

    /// The DOFs that are associated with a closure of a subentity of the reference cell
    fn entity_closure_dofs(&self, entity_dim: usize, entity_number: usize) -> Option<&[usize]>;

    /// Get the required shape for a tabulation array
    fn tabulate_array_shape(&self, nderivs: usize, npoints: usize) -> [usize; 4];

    /// Push function values forward to a physical cell
    fn push_forward<
        Array2: RandomAccessByRef<2, Item = <Self::T as RlstScalar>::Real> + Shape<2>,
        Array3: RandomAccessByRef<3, Item = <Self::T as RlstScalar>::Real> + Shape<3>,
        Array4: RandomAccessByRef<4, Item = <Self::T as RlstScalar>::Real> + Shape<4>,
        Array4Mut: RandomAccessMut<4, Item = <Self::T as RlstScalar>::Real> + Shape<4>,
    >(
        &self,
        reference_values: &Array4,
        nderivs: usize,
        jacobians: &Array3,
        jacobian_determinants: &Array2,
        inverse_jacobians: &Array3,
        physical_values: &mut Array4Mut,
    );

    /// Pull function values back to the reference cell
    fn pull_back<
        Array2: RandomAccessByRef<2, Item = <Self::T as RlstScalar>::Real> + Shape<2>,
        Array3: RandomAccessByRef<3, Item = <Self::T as RlstScalar>::Real> + Shape<3>,
        Array4: RandomAccessByRef<4, Item = <Self::T as RlstScalar>::Real> + Shape<4>,
        Array4Mut: RandomAccessMut<4, Item = <Self::T as RlstScalar>::Real> + Shape<4>,
    >(
        &self,
        physical_values: &Array4,
        nderivs: usize,
        jacobians: &Array3,
        jacobian_determinants: &Array2,
        inverse_jacobians: &Array3,
        reference_values: &mut Array4Mut,
    );
}

pub trait ElementFamily {
    //! A family of finite elements
    /// The scalar type
    type T: RlstScalar;
    /// Cell type
    type CellType: Debug + PartialEq + Eq + Clone + Copy + Hash;
    /// The finite element type
    type FiniteElement: FiniteElement<T = Self::T, CellType = Self::CellType> + 'static;

    /// Get an elenent for a cell type
    fn element(&self, cell_type: Self::CellType) -> Self::FiniteElement;
}

pub trait QuadratureRule {
    //! A quadrature rule
    /// The scalar type
    type T: RlstScalar;
    /// Quadrature points
    fn points(&self) -> &[Self::T];
    /// Quadrature weights
    fn weights(&self) -> &[Self::T];
    /// Number of quadrature points
    fn npoints(&self) -> usize;
    /// Topological dimension of cell (ie number of components of each point)
    fn dim(&self) -> usize;
}

pub trait Map {
    //! A map from the reference cell to physical cells

    /// Push function values forward to a physical cell
    fn push_forward<
        T: RlstScalar<Real = T>,
        Array2: RandomAccessByRef<2, Item = T> + Shape<2>,
        Array3: RandomAccessByRef<3, Item = T> + Shape<3>,
        Array4: RandomAccessByRef<4, Item = T> + Shape<4>,
        Array4Mut: RandomAccessMut<4, Item = T> + Shape<4>,
    >(
        &self,
        reference_values: &Array4,
        nderivs: usize,
        jacobians: &Array3,
        jacobian_determinants: &Array2,
        inverse_jacobians: &Array3,
        physical_values: &mut Array4Mut,
    );

    /// Pull function values back to the reference cell
    fn pull_back<
        T: RlstScalar<Real = T>,
        Array2: RandomAccessByRef<2, Item = T> + Shape<2>,
        Array3: RandomAccessByRef<3, Item = T> + Shape<3>,
        Array4: RandomAccessByRef<4, Item = T> + Shape<4>,
        Array4Mut: RandomAccessMut<4, Item = T> + Shape<4>,
    >(
        &self,
        physical_values: &Array4,
        nderivs: usize,
        jacobians: &Array3,
        jacobian_determinants: &Array2,
        inverse_jacobians: &Array3,
        reference_values: &mut Array4Mut,
    );
}
