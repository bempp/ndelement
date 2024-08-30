//! Types
#[cfg(feature = "mpi")]
use mpi::traits::Equivalence;
use rlst::{Array, BaseArray, VectorContainer};

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
    Standard = 0,
    /// The element is discontinuous betweeen cells
    Discontinuous = 1,
}

/// The map type used by an element
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
#[repr(u8)]
pub enum MapType {
    /// Identity map
    Identity = 0,
    /// Covariant Piola map
    ///
    /// This map is used by H(curl) elements
    CovariantPiola = 1,
    /// Contravariant Piola map
    ///
    /// This map is used by H(div) elements
    ContravariantPiola = 2,
    /// L2 Piola map
    L2Piola = 3,
}

/// The type of a reference cell
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
#[repr(u8)]
pub enum ReferenceCellType {
    /// A point
    Point = 0,
    /// A line interval
    Interval = 1,
    /// A triangle
    Triangle = 2,
    /// A quadrilateral
    Quadrilateral = 3,
    /// A tetrahedron (whose faces are all triangles)
    Tetrahedron = 4,
    /// A hexahedron (whose faces are all quadrilaterals)
    Hexahedron = 5,
    /// A triangular prism
    Prism = 6,
    /// A square-based pyramid
    Pyramid = 7,
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

impl ReferenceCellType {
    /// Create a reference cell type from a u8
    pub fn from(i: u8) -> Option<ReferenceCellType> {
        match i {
            0 => Some(ReferenceCellType::Point),
            1 => Some(ReferenceCellType::Interval),
            2 => Some(ReferenceCellType::Triangle),
            3 => Some(ReferenceCellType::Quadrilateral),
            4 => Some(ReferenceCellType::Tetrahedron),
            5 => Some(ReferenceCellType::Hexahedron),
            6 => Some(ReferenceCellType::Prism),
            7 => Some(ReferenceCellType::Pyramid),
            _ => None,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_reference_cell_type() {
        assert_eq!(
            ReferenceCellType::Point,
            ReferenceCellType::from(ReferenceCellType::Point as u8).unwrap()
        );
        assert_eq!(
            ReferenceCellType::Interval,
            ReferenceCellType::from(ReferenceCellType::Interval as u8).unwrap()
        );
        assert_eq!(
            ReferenceCellType::Triangle,
            ReferenceCellType::from(ReferenceCellType::Triangle as u8).unwrap()
        );
        assert_eq!(
            ReferenceCellType::Quadrilateral,
            ReferenceCellType::from(ReferenceCellType::Quadrilateral as u8).unwrap()
        );
        assert_eq!(
            ReferenceCellType::Tetrahedron,
            ReferenceCellType::from(ReferenceCellType::Tetrahedron as u8).unwrap()
        );
        assert_eq!(
            ReferenceCellType::Hexahedron,
            ReferenceCellType::from(ReferenceCellType::Hexahedron as u8).unwrap()
        );
        assert_eq!(
            ReferenceCellType::Prism,
            ReferenceCellType::from(ReferenceCellType::Prism as u8).unwrap()
        );
        assert_eq!(
            ReferenceCellType::Pyramid,
            ReferenceCellType::from(ReferenceCellType::Pyramid as u8).unwrap()
        );
    }
}
