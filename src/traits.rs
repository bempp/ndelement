//! Traits
use crate::types::DofTransformation;
use rlst::{Array, MutableArrayImpl, RlstScalar, ValueArrayImpl};
use std::fmt::Debug;
use std::hash::Hash;

/// This trait provides the definition of a finite element.
pub trait FiniteElement {
    /// The scalar type
    type T: RlstScalar;

    /// Cell type
    ///
    /// The cell type is typically defined through [ReferenceCellType](crate::types::ReferenceCellType).
    type CellType: Debug + PartialEq + Eq + Clone + Copy + Hash;

    /// Transformation type
    ///
    /// The Transformation type specifies possible transformations of the dofs on the reference element.
    /// In most cases these will be rotations and reflections as defined in [Transformation](crate::types::Transformation).
    type TransformationType: Debug + PartialEq + Eq + Clone + Copy + Hash;

    /// The reference cell type, eg one of `Point`, `Interval`, `Triangle`, etc.
    fn cell_type(&self) -> Self::CellType;

    /// The smallest degree Lagrange space that contains all possible polynomials of the finite element's polynomial space.
    ///
    /// Details on the definition of the degree of Lagrange spaces of finite elements are
    /// given [here](https://defelement.org/ciarlet.html#The+degree+of+a+finite+element).
    fn lagrange_superdegree(&self) -> usize;

    /// The number of basis functions.
    fn dim(&self) -> usize;

    /// The shape of the values returned by functions in $\mathcal{V}$.
    fn value_shape(&self) -> &[usize];

    /// The number of values returned.
    ///
    /// If eg `value_shape = [3, 4]` then `value_size = 3 x 4 = 12`.
    fn value_size(&self) -> usize;

    /// Tabulate the values of the basis functions and their derivatives at a set of points
    ///
    /// - `points` is a two-dimensional array where each column contains the coordinates of one point.
    /// - `nderivs` is the desired number of derivatives (0 for no derivatives, 1 for all first order derivatives, etc.).
    /// - `data` is the 4-dimensional output array. The first dimension is the number of derivatives,
    ///   the second dimension is the number of evaluation points, the third dimension is the number `n` of
    ///   basis functions on the element and the last dimension is the value size of the basis function output.
    ///   For example, `data[3, 2, 1, 0]` returns the 0th value of the third derivative on the point with index 2 for the
    ///   basis function with index 1.
    ///
    /// ## Remark
    ///
    /// Let $d^{i + k} = dx^{i}dy^{j}$ be a derivative with respect to $x$, $y$ in two dimensions and    
    /// $d^{i + k + j} = dx^{i}dy^{j}dz^{k}$ be a derivative with respect to $x$, $y$, and $z$ in three dimensions.
    /// Then the corresponding index $\ell$ in the first dimension of the `data` array is computed as follows.
    /// - Triangle: $\ell = (i + j + 1) * (i + j) / 2 + j$
    /// - Quadrilateral: $\ell = i * (n + 1) + j$
    /// - Tetrahedron: $\ell = (i + j + k) * (i + j + k + 1) * (i + j + k + 2) / 6 + (j + k) * (j + k + 1) / 2 + k$
    /// - Hexahedron $\ell = i * (n + 1) * (n + 1) + j * (n + 1) + k$.
    ///
    /// For the quadrilateral and hexahedron the parameter $n$ denotes the degree of the Lagrange space.
    ///
    fn tabulate<
        Array2Impl: ValueArrayImpl<<Self::T as RlstScalar>::Real, 2>,
        Array4MutImpl: MutableArrayImpl<Self::T, 4>,
    >(
        &self,
        points: &Array<Array2Impl, 2>,
        nderivs: usize,
        data: &mut Array<Array4MutImpl, 4>,
    );

    /// Return the dof indices that are associated with the subentity with index `entity_number` and dimension `entity_dim`.
    ///
    /// - For `entity_dim = 0` this returns the dof associated with the corresponding point.
    /// - For `entity_dim = 1` this returns the dofs associated with the corresponding edge.
    /// - For `entity_dim = 2` this returns the dofs associated with the corresponding face.
    ///
    /// Note that this does not return dofs on the boundary of an entity, that means eg
    /// for an edge the dofs associated with the two vertices at the boundary of the edge are not returned.
    /// To return also the boundary dofs use [FiniteElement::entity_closure_dofs].
    fn entity_dofs(&self, entity_dim: usize, entity_number: usize) -> Option<&[usize]>;

    /// The DOFs that are associated with a closure of a subentity of the reference cell.
    ///
    /// This method is similar to [FiniteElement::entity_dofs]. But it returns additionally the dofs
    /// associated with the boundary of an entity, eg for an edge it returns also the dofs associated
    /// with the boundary vertices of they exist.
    fn entity_closure_dofs(&self, entity_dim: usize, entity_number: usize) -> Option<&[usize]>;

    /// Get the required shape for a tabulation array.
    fn tabulate_array_shape(&self, nderivs: usize, npoints: usize) -> [usize; 4];

    /// Push function values forward to a physical cell.
    ///
    /// Usually this is just an identity map. But for certain element types function values
    /// on the reference cell differ from those on the physical cell, eg in the case of a
    /// Piola transform. This method implements the corresponding transformation or an identity
    /// map if no transformation is required.
    ///
    /// - `reference_values`: The values on the reference cell.
    /// - `nderivs`: The number (degree) of derivatives.
    /// - `jacobians:` A three-dimensional array of jacobians of the map from reference to physical cell.
    ///   The first dimension is the reference point. The second dimension is the geometric dimension of the physical space and
    ///   the third dimension is the topological dimension of the reference element, eg
    ///   for the map of 5 points from the reference triangle to a physical surface triangle embedded in 3d space the dimension
    ///   of `jacobians` is `[5, 3, 2]`.
    /// - `jacobian_determinants`: Let $J$ be the Jacobian from the map of the reference to the physical element at a given point.
    ///   The corresponding Jacobian determinant is given as $d = \sqrt{\det(J^TJ)}$. `jacobian_determinants[j]` stores the Jacobian
    ///   determinant at position `j`.
    /// - `inverse_jacobians`: A three dimensional array that stores the inverse Jacobian for the point with index j at position
    ///   `inverse_jacobians[j, :, :]`. The first dimension of `inverse_jacobians` is the point index. The second dimension
    ///   is the topological dimension, and the third dimension is the geometric dimension. If the Jacobian is rectangular then the
    ///   inverse Jacobian is the pseudo-inverse of the Jacobian such that $J^\dagger J = I$.
    /// - `physical_values`: The output array of the push operation. Its required shape can be queried with [FiniteElement::physical_value_shape].
    fn push_forward<
        Array3RealImpl: ValueArrayImpl<<Self::T as RlstScalar>::Real, 3>,
        Array4Impl: ValueArrayImpl<Self::T, 4>,
        Array4MutImpl: MutableArrayImpl<Self::T, 4>,
    >(
        &self,
        reference_values: &Array<Array4Impl, 4>,
        nderivs: usize,
        jacobians: &Array<Array3RealImpl, 3>,
        jacobian_determinants: &[<Self::T as RlstScalar>::Real],
        inverse_jacobians: &Array<Array3RealImpl, 3>,
        physical_values: &mut Array<Array4MutImpl, 4>,
    );

    /// Pull function values back to the reference cell.
    ///
    /// This is the inverse operation to [FiniteElement::push_forward].
    fn pull_back<
        Array3RealImpl: ValueArrayImpl<<Self::T as RlstScalar>::Real, 3>,
        Array4Impl: ValueArrayImpl<Self::T, 4>,
        Array4MutImpl: MutableArrayImpl<Self::T, 4>,
    >(
        &self,
        physical_values: &Array<Array4Impl, 4>,
        nderivs: usize,
        jacobians: &Array<Array3RealImpl, 3>,
        jacobian_determinants: &[<Self::T as RlstScalar>::Real],
        inverse_jacobians: &Array<Array3RealImpl, 3>,
        reference_values: &mut Array<Array4MutImpl, 4>,
    );

    /// The value shape on a physical cell
    fn physical_value_shape(&self, gdim: usize) -> Vec<usize>;

    /// The value size on a physical cell
    fn physical_value_size(&self, gdim: usize) -> usize {
        let mut vs = 1;
        for i in self.physical_value_shape(gdim) {
            vs *= i;
        }
        vs
    }

    /// The DOF transformation for a sub-entity.
    fn dof_transformation(
        &self,
        entity: Self::CellType,
        transformation: Self::TransformationType,
    ) -> Option<&DofTransformation<Self::T>>;

    /// Apply permutation parts of DOF transformations.
    fn apply_dof_permutations<T>(&self, data: &mut [T], cell_orientation: i32);

    /// Apply non-permutation parts of DOF transformations.
    fn apply_dof_transformations(&self, data: &mut [Self::T], cell_orientation: i32);

    /// Apply DOF transformations.
    fn apply_dof_permutations_and_transformations(
        &self,
        data: &mut [Self::T],
        cell_orientation: i32,
    ) {
        self.apply_dof_permutations(data, cell_orientation);
        self.apply_dof_transformations(data, cell_orientation);
    }
}

/// A factory that can create elements belonging to a given element family.
pub trait ElementFamily {
    /// The scalar type
    type T: RlstScalar;
    /// Cell type
    type CellType: Debug + PartialEq + Eq + Clone + Copy + Hash;
    /// The finite element type
    type FiniteElement: FiniteElement<T = Self::T, CellType = Self::CellType> + 'static;

    /// Create an element for the given cell type.
    fn element(&self, cell_type: Self::CellType) -> Self::FiniteElement;
}

/// A map from the reference cell to physical cells.
pub trait Map {
    /// Push values from the reference cell to physical cells.
    fn push_forward<
        T: RlstScalar,
        Array3RealImpl: ValueArrayImpl<T::Real, 3>,
        Array4Impl: ValueArrayImpl<T, 4>,
        Array4MutImpl: MutableArrayImpl<T, 4>,
    >(
        &self,
        reference_values: &Array<Array4Impl, 4>,
        nderivs: usize,
        jacobians: &Array<Array3RealImpl, 3>,
        jacobian_determinants: &[T::Real],
        inverse_jacobians: &Array<Array3RealImpl, 3>,
        physical_values: &mut Array<Array4MutImpl, 4>,
    );

    /// Pull values back to the reference cell.
    fn pull_back<
        T: RlstScalar,
        Array3RealImpl: ValueArrayImpl<T::Real, 3>,
        Array4Impl: ValueArrayImpl<T, 4>,
        Array4MutImpl: MutableArrayImpl<T, 4>,
    >(
        &self,
        physical_values: &Array<Array4Impl, 4>,
        nderivs: usize,
        jacobians: &Array<Array3RealImpl, 3>,
        jacobian_determinants: &[T::Real],
        inverse_jacobians: &Array<Array3RealImpl, 3>,
        reference_values: &mut Array<Array4MutImpl, 4>,
    );

    /// The value shape on a physical cell.
    fn physical_value_shape(&self, gdim: usize) -> Vec<usize>;
}
