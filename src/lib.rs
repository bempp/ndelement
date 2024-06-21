//! n-dimensional grid
#![cfg_attr(feature = "strict", deny(warnings))]
#![warn(missing_docs)]

pub mod ciarlet;
pub mod polynomials;
pub mod reference_cell;
pub mod traits;
pub mod types;

#[cfg(test)]
mod test {
    extern crate blas_src;
    extern crate lapack_src;
}
