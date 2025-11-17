//! A library for the definition and tabulation of finite elements in Rust.
//!
//! `ndelement` provides definition of many frequently used low and high order finite elements
//! and provides routines for the tabulation of their values.
//!
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
