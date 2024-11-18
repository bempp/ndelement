//! Mathematical functions
use crate::types::{Array2D, ArrayND};
use rlst::{RlstScalar, Shape};

/// Orthogonalise the rows of a matrix, starting with the row numbered `start`
pub fn orthogonalise<T: RlstScalar>(mat: &mut Array2D<T>, start: usize) {
    for row in start..mat.shape()[0] {
        let norm = (0..mat.shape()[1])
            .map(|i| mat[[row, i]].powi(2))
            .sum::<T>()
            .sqrt();
        for i in 0..mat.shape()[1] {
            mat[[row, i]] /= norm;
        }
        for r in row + 1..mat.shape()[0] {
            let dot = (0..mat.shape()[1])
                .map(|i| mat[[row, i]] * mat[[r, i]])
                .sum::<T>();
            for i in 0..mat.shape()[1] {
                let sub = dot * mat[[row, i]];
                mat[[r, i]] -= sub;
            }
        }
    }
}

/// Orthogonalise the rows of a matrix, starting with the row numbered `start`
pub fn orthogonalise_3<T: RlstScalar>(mat: &mut ArrayND<3, T>, start: usize) {
    for row in start..mat.shape()[0] {
        let norm = (0..mat.shape()[1])
            .map(|i| {
                (0..mat.shape()[2])
                    .map(|j| mat[[row, i, j]].powi(2))
                    .sum::<T>()
            })
            .sum::<T>()
            .sqrt();
        for i in 0..mat.shape()[1] {
            for j in 0..mat.shape()[2] {
                mat[[row, i, j]] /= norm;
            }
        }
        for r in row + 1..mat.shape()[0] {
            let dot = (0..mat.shape()[1])
                .map(|i| {
                    (0..mat.shape()[2])
                        .map(|j| mat[[row, i, j]] * mat[[r, i, j]])
                        .sum::<T>()
                })
                .sum::<T>();
            for i in 0..mat.shape()[1] {
                for j in 0..mat.shape()[2] {
                    let sub = dot * mat[[row, i, j]];
                    mat[[r, i, j]] -= sub;
                }
            }
        }
    }
}
