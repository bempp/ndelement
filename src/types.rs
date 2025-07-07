//! Types
#[cfg(feature = "mpi")]
use mpi::traits::Equivalence;
use rlst::{Array, BaseArray, RlstScalar, VectorContainer};
use strum_macros::EnumIter;

/// An N-dimensional array
pub type ArrayND<const N: usize, T> = Array<T, BaseArray<T, VectorContainer<T>, N>, N>;
/// A 2-dimensional array
pub type Array2D<T> = ArrayND<2, T>;

/// Continuity type
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
#[repr(u8)]
pub enum Continuity {
    /// The element has standard continuity between cells
    ///
    /// For some element, this option does not indicate that the values are fully continuous.
    /// For example, for Raviart-Thomas elements it only indicates that the normal components
    /// are continuous across edges
    Standard,
    /// The element is discontinuous betweeen cells
    Discontinuous,
}

/// The map type used by an element
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
#[repr(u8)]
pub enum MapType {
    /// Identity map
    Identity,
    /// Covariant Piola map
    ///
    /// This map is used by H(curl) elements
    CovariantPiola,
    /// Contravariant Piola map
    ///
    /// This map is used by H(div) elements
    ContravariantPiola,
    /// L2 Piola map
    L2Piola,
}

/// The type of a reference cell
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(EnumIter, Debug, PartialEq, Eq, Clone, Copy, Hash)]
#[repr(u8)]
pub enum ReferenceCellType {
    /// A point
    Point,
    /// A line interval
    Interval,
    /// A triangle
    Triangle,
    /// A quadrilateral
    Quadrilateral,
    /// A tetrahedron (whose faces are all triangles)
    Tetrahedron,
    /// A hexahedron (whose faces are all quadrilaterals)
    Hexahedron,
    /// A triangular prism
    Prism,
    /// A square-based pyramid
    Pyramid,
}

#[cfg(feature = "mpi")]
unsafe impl Equivalence for Continuity {
    type Out = <u8 as Equivalence>::Out;
    fn equivalent_datatype() -> <u8 as Equivalence>::Out {
        <u8 as Equivalence>::equivalent_datatype()
    }
}
#[cfg(feature = "mpi")]
unsafe impl Equivalence for MapType {
    type Out = <u8 as Equivalence>::Out;
    fn equivalent_datatype() -> <u8 as Equivalence>::Out {
        <u8 as Equivalence>::equivalent_datatype()
    }
}
#[cfg(feature = "mpi")]
unsafe impl Equivalence for ReferenceCellType {
    type Out = <u8 as Equivalence>::Out;
    fn equivalent_datatype() -> <u8 as Equivalence>::Out {
        <u8 as Equivalence>::equivalent_datatype()
    }
}

/// A DOF transformation
#[derive(Debug)]
#[repr(u8)]
pub enum DofTransformation<T: RlstScalar> {
    /// An identity transformation
    Identity,
    /// A permutation
    Permutation(Vec<usize>),
    /// A linear transformation
    Transformation(Array2D<T>, Vec<usize>),
}

/// A transformation of a sub-entity
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
#[repr(u8)]
pub enum Transformation {
    /// A reflection
    Reflection,
    /// A rotation
    Rotation,
}
