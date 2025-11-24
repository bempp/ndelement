//! A library for the definition and tabulation of finite elements in Rust.
//!
//! `ndelement` provides definition of many frequently used low and high order finite elements
//! and provides routines for the tabulation of their values.
//!
//! The following presents a simple example of how to use `ndelement`.
//!
//! ```
//! use ndelement::ciarlet::LagrangeElementFamily;
//! use ndelement::traits::{ElementFamily, FiniteElement};
//! use ndelement::types::{Continuity, ReferenceCellType};
//! use rlst::DynArray;
//!
//! // Create the degree 2 Lagrange element family. A family is a set of finite elements with the
//! // same family type, degree, and continuity across a set of cells
//! let family = LagrangeElementFamily::<f64>::new(2, Continuity::Standard);
//!
//! // Get the element in the family on a triangle
//! let element = family.element(ReferenceCellType::Triangle);
//! println!("Cell: {:?}", element.cell_type());
//!
//! // Get the element in the family on a quadrilateral
//! let element = family.element(ReferenceCellType::Quadrilateral);
//! println!("Cell: {:?}", element.cell_type());
//!
//! // Create an array to store the basis function values
//! let mut basis_values = DynArray::<f64, 4>::from_shape(element.tabulate_array_shape(0, 1));
//! // Create array containing the point [1/2, 1/2]
//! let mut points = DynArray::<f64, 2>::from_shape([2, 1]);
//! points[[0, 0]] = 1.0 / 2.0;
//! points[[1, 0]] = 1.0 / 2.0;
//! // Tabulate the element's basis functions at the point
//! element.tabulate(&points, 0, &mut basis_values);
//! println!(
//!     "The values of the basis functions at the point (1/2, 1/2) are: {:?}",
//!     basis_values.data()
//! );
//! ```
#![cfg_attr(feature = "strict", deny(warnings), deny(unused_crate_dependencies))]
#![warn(missing_docs)]

pub mod bindings;
pub mod ciarlet;
pub mod map;
pub mod math;
pub mod orientation;
pub mod polynomials;
pub mod quadrature;
pub mod reference_cell;
pub mod traits;
pub mod types;
