//! Quadrature
mod gauss_jacobi;
pub use gauss_jacobi::make_gauss_jacobi_quadrature;

use crate::traits::QuadratureRule as QuadratureRuleTrait;
use rlst::RlstScalar;

/// Quadrature rule
pub struct QuadratureRule<T: RlstScalar<Real = T>> {
    points: Vec<T>,
    npoints: usize,
    dim: usize,
    weights: Vec<T>,
}

impl<T: RlstScalar<Real = T>> QuadratureRule<T> {
    /// Create new
    pub fn new(points: Vec<T>, weights: Vec<T>) -> Self {
        let npoints = weights.len();
        debug_assert!(points.len() % npoints == 0);
        let dim = points.len() / npoints;
        Self {
            points,
            npoints,
            dim,
            weights,
        }
    }
}
impl<T: RlstScalar<Real = T>> QuadratureRuleTrait for QuadratureRule<T> {
    type T = T;
    fn points(&self) -> &[T] {
        &self.points
    }
    fn weights(&self) -> &[T] {
        &self.weights
    }
    fn npoints(&self) -> usize {
        self.npoints
    }
    fn dim(&self) -> usize {
        self.dim
    }
}
