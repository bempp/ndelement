//! Traits.
use crate::types::DofTransformation;
use rlst::{Array, MutableArrayImpl, RlstScalar, ValueArrayImpl};
use std::fmt::Debug;
use std::hash::Hash;

/// A finite element.
pub trait FiniteElement {
    /// The scalar type
    type T: RlstScalar;

    /// Cell type
    ///
    /// The cell type is typically defined through [ReferenceCellType](crate::types::ReferenceCellType).
    type CellType: Debug + PartialEq + Eq + Clone + Copy + Hash;

    /// The reference cell type, eg one of `Point`, `Interval`, `Triangle`, etc.
    fn cell_type(&self) -> Self::CellType;

    /// The number of basis functions.
    fn dim(&self) -> usize;

    /// The shape of the values returned by functions in $\mathcal{V}$.
    ///
    /// If the values are scalar an empty slice is returned.
    fn value_shape(&self) -> &[usize];

    /// The number of values returned.
    ///
    /// If (for example) `value_shape` is `[3, 4]` then `value_size` is $3\times4 = 12$.
    /// If `value_shape` returns an empty array (ie the shape functions are scalar) the
    // convention is used that the product of the elements of an empty array is 1.
    fn value_size(&self) -> usize;

    /// Tabulate the values of the basis functions and their derivatives at a set of points
    ///
    /// - `points` is a two-dimensional array where each column contains the coordinates of one point.
    /// - `nderivs` is the desired number of derivatives (0 for no derivatives, 1 for all first order derivatives, etc.).
    /// - `data` is the 4-dimensional output array. The first dimension is the total number of partial derivatives,
    ///   the second dimension is the number of evaluation points, the third dimension is the number of
    ///   basis functions of the element, and the fourth dimension is the value size of the basis function output.
    ///   For example, `data[3, 2, 1, 0]` returns the 0th value of the third partial derivative on the point with index 2 for the
    ///   basis function with index 1.
    ///
    /// ## Remark
    /// Let $d^{i + k} = dx^{i}dy^{j}$ be a derivative with respect to $x$, $y$ in two dimensions and    
    /// $d^{i + k + j} = dx^{i}dy^{j}dz^{k}$ be a derivative with respect to $x$, $y$, and $z$ in three dimensions.
    /// Then the corresponding index $\ell$ in the first dimension of the `data` array is computed as follows.
    ///
    /// - Triangle: $l = (i + j + 1) * (i + j) / 2 + j$
    /// - Quadrilateral: $l = i * (n + 1) + j$
    /// - Tetrahedron: $l = (i + j + k) * (i + j + k + 1) * (i + j + k + 2) / 6 + (j + k) * (j + k + 1) / 2 + k$
    /// - Hexahedron $l = i * (n + 1) * (n + 1) + j * (n + 1) + k$.
    ///
    /// For the quadrilaterals and hexahedra, the parameter $n$ denotes the Lagrange superdegree.
    fn tabulate<
        TGeo: RlstScalar,
        Array2Impl: ValueArrayImpl<TGeo, 2>,
        Array4MutImpl: MutableArrayImpl<Self::T, 4>,
    >(
        &self,
        points: &Array<Array2Impl, 2>,
        nderivs: usize,
        data: &mut Array<Array4MutImpl, 4>,
    );

    /// Get the required shape for a tabulation array.
    fn tabulate_array_shape(&self, nderivs: usize, npoints: usize) -> [usize; 4];

    /// Return the dof indices that are associated with the subentity with index `entity_number` and dimension `entity_dim`.
    ///
    /// - For `entity_dim = 0` this returns the degrees of freedom (dofs) associated with the corresponding point.
    /// - For `entity_dim = 1` this returns the dofs associated with the corresponding edge.
    /// - For `entity_dim = 2` this returns the dofs associated with the corresponding face.
    ///
    /// Note that this does not return dofs on the boundary of an entity, that means that (for example)
    /// for an edge the dofs associated with the two vertices at the boundary of the edge are not returned.
    /// To return also the boundary dofs use [FiniteElement::entity_closure_dofs].
    fn entity_dofs(&self, entity_dim: usize, entity_number: usize) -> Option<&[usize]>;

    /// The DOFs that are associated with a closure of a subentity of the reference cell.
    ///
    /// This method is similar to [FiniteElement::entity_dofs], but it includes the dofs
    /// associated with the boundary of an entity. For an edge (for example) it returns the dofs associated
    /// with the vertices at the boundary of the edge (as well as the dofs associated with the edge itself).
    fn entity_closure_dofs(&self, entity_dim: usize, entity_number: usize) -> Option<&[usize]>;
}

/// A finite element that is mapped from a reference cell.
pub trait MappedFiniteElement: FiniteElement {
    /// Transformation type
    ///
    /// The Transformation type specifies possible transformations of the dofs on the reference element.
    /// In most cases these will be rotations and reflections as defined in [Transformation](crate::types::Transformation).
    type TransformationType: Debug + PartialEq + Eq + Clone + Copy + Hash;

    /// The smallest degree Lagrange space that contains all possible polynomials of the finite element's polynomial space.
    ///
    /// Details on the definition of the degree of Lagrange spaces of finite elements are
    /// given [here](https://defelement.org/ciarlet.html#The+degree+of+a+finite+element).
    fn lagrange_superdegree(&self) -> usize;

    /// Push function values forward to a physical cell.
    ///
    /// For Lagrange elements, this is an identity map. For many other element types, the function values
    /// on the reference cell differ from those on the physical cell: for example Nedlec elements use a covariant
    /// Piola transform. This method implements the appropriate transformation for the element.
    ///
    /// - `reference_values`: The values on the reference cell. The shape of this input is the same as the `data` input to the function
    ///   [[FiniteElement::tabulate].
    /// - `nderivs`: The number of derivatives.
    /// - `jacobians:` A three-dimensional array of jacobians of the map from reference to physical cell.
    ///   The first dimension is the reference point, the second dimension is the geometric dimension of the physical space, and
    ///   the third dimension is the topological dimension of the reference element. For example,
    ///   for the map of 5 points from the reference triangle to a physical surface triangle embedded in 3d space the dimension
    ///   of `jacobians` is `[5, 3, 2]`.
    /// - `jacobian_determinants`: The determinant of the jacobian at each point. If the jacobian $J$ is not square, then the
    ///   determinant is computed using $d=\sqrt{\det(J^TJ)}$.
    /// - `inverse_jacobians`: A three dimensional array that stores the inverse jacobian for the point with index j at position
    ///   `inverse_jacobians[j, :, :]`. The first dimension of `inverse_jacobians` is the point index, the second dimension
    ///   is the topological dimension, and the third dimension is the geometric dimension. If the Jacobian is rectangular then the
    ///   inverse Jacobian is the pseudo-inverse of the Jacobian, ie the matrix $J^\dagger$ such that $J^\dagger J = I$.
    /// - `physical_values`: The output array of the push operation. This shape of this array is the same as the `reference_values`
    ///   input, with the [MappedFiniteElement::physical_value_size] used instead of the reference value size.
    fn push_forward<
        TGeo: RlstScalar,
        Array3GeoImpl: ValueArrayImpl<TGeo, 3>,
        Array4Impl: ValueArrayImpl<Self::T, 4>,
        Array4MutImpl: MutableArrayImpl<Self::T, 4>,
    >(
        &self,
        reference_values: &Array<Array4Impl, 4>,
        nderivs: usize,
        jacobians: &Array<Array3GeoImpl, 3>,
        jacobian_determinants: &[TGeo],
        inverse_jacobians: &Array<Array3GeoImpl, 3>,
        physical_values: &mut Array<Array4MutImpl, 4>,
    );

    /// Pull function values back to the reference cell.
    ///
    /// This is the inverse operation to [MappedFiniteElement::push_forward].
    fn pull_back<
        TGeo: RlstScalar,
        Array3GeoImpl: ValueArrayImpl<TGeo, 3>,
        Array4Impl: ValueArrayImpl<Self::T, 4>,
        Array4MutImpl: MutableArrayImpl<Self::T, 4>,
    >(
        &self,
        physical_values: &Array<Array4Impl, 4>,
        nderivs: usize,
        jacobians: &Array<Array3GeoImpl, 3>,
        jacobian_determinants: &[TGeo],
        inverse_jacobians: &Array<Array3GeoImpl, 3>,
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

/// A family of finite elements defined on various cell types, with appropriate continuity
/// between different cell types.
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
        TGeo: RlstScalar,
        Array3GeoImpl: ValueArrayImpl<TGeo, 3>,
        Array4Impl: ValueArrayImpl<T, 4>,
        Array4MutImpl: MutableArrayImpl<T, 4>,
    >(
        &self,
        reference_values: &Array<Array4Impl, 4>,
        nderivs: usize,
        jacobians: &Array<Array3GeoImpl, 3>,
        jacobian_determinants: &[TGeo],
        inverse_jacobians: &Array<Array3GeoImpl, 3>,
        physical_values: &mut Array<Array4MutImpl, 4>,
    );

    /// Pull values back to the reference cell.
    fn pull_back<
        T: RlstScalar,
        TGeo: RlstScalar,
        Array3GeoImpl: ValueArrayImpl<TGeo, 3>,
        Array4Impl: ValueArrayImpl<T, 4>,
        Array4MutImpl: MutableArrayImpl<T, 4>,
    >(
        &self,
        physical_values: &Array<Array4Impl, 4>,
        nderivs: usize,
        jacobians: &Array<Array3GeoImpl, 3>,
        jacobian_determinants: &[TGeo],
        inverse_jacobians: &Array<Array3GeoImpl, 3>,
        reference_values: &mut Array<Array4MutImpl, 4>,
    );

    /// The value shape on a physical cell.
    fn physical_value_shape(&self, gdim: usize) -> Vec<usize>;
}
