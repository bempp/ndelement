//! Quadrature
mod gauss_jacobi;
pub use gauss_jacobi::{
    make_quadrature as make_gauss_jacobi_quadrature, npoints as gauss_jacobi_quadrature_npoints,
};

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

#[cfg(test)]
mod test {
    use super::*;
    use crate::types::ReferenceCellType;
    use approx::*;
    use itertools::izip;

    #[test]
    fn test_against_known_tetrahedron() {
        let rule = make_gauss_jacobi_quadrature::<f64>(ReferenceCellType::Tetrahedron, 2);

        println!("{:?}", rule.points());
        println!("{:?}", rule.weights());

        for (i, pt) in [
            [0.15668263733681834, 0.136054976802846, 0.12251482265544139],
            [0.08139566701467026, 0.0706797241593969, 0.5441518440112253],
            [0.06583868706004443, 0.5659331650728009, 0.12251482265544139],
            [0.03420279323676642, 0.29399880063162287, 0.5441518440112253],
            [0.5847475632048944, 0.136054976802846, 0.12251482265544139],
            [0.30377276481470755, 0.0706797241593969, 0.5441518440112253],
            [0.24571332521171338, 0.5659331650728009, 0.12251482265544139],
            [0.12764656212038544, 0.29399880063162287, 0.5441518440112253],
        ]
        .iter()
        .enumerate()
        {
            for (j, c) in pt.iter().enumerate() {
                assert_relative_eq!(rule.points()[3 * i + j], c);
            }
        }
        for (w0, w1) in izip!(
            rule.weights(),
            [
                0.03697985635885291,
                0.01602704059847662,
                0.02115700645452406,
                0.009169429921479746,
                0.03697985635885291,
                0.01602704059847662,
                0.02115700645452406,
                0.009169429921479746
            ]
        ) {
            assert_relative_eq!(*w0, w1);
        }
    }

    #[test]
    fn test_against_known_triangle() {
        let rule = make_gauss_jacobi_quadrature::<f64>(ReferenceCellType::Triangle, 3);

        for (i, pt) in [
            [0.17855872826361643, 0.15505102572168217],
            [0.07503111022260814, 0.6449489742783178],
            [0.6663902460147014, 0.15505102572168217],
            [0.2800199154990741, 0.6449489742783178],
        ]
        .iter()
        .enumerate()
        {
            for (j, c) in pt.iter().enumerate() {
                assert_relative_eq!(rule.points()[2 * i + j], c);
            }
        }
        for (w0, w1) in izip!(
            rule.weights(),
            [
                0.15902069087198858,
                0.09097930912801142,
                0.15902069087198858,
                0.09097930912801142
            ]
        ) {
            assert_relative_eq!(*w0, w1);
        }
    }
}
