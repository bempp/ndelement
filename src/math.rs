//! Mathematical functions.
use rlst::{Array, MutableArrayImpl, RlstScalar, ValueArrayImpl};

/// Orthogonalise the rows of a matrix, starting with the row numbered `start`
pub(crate) fn orthogonalise_3<T: RlstScalar, Array3MutImpl: MutableArrayImpl<T, 3>>(
    mat: &mut Array<Array3MutImpl, 3>,
    start: usize,
) {
    for row in start..mat.shape()[0] {
        let norm = (0..mat.shape()[1])
            .map(|i| {
                (0..mat.shape()[2])
                    .map(|j| mat.get([row, i, j]).unwrap().powi(2))
                    .sum::<T>()
            })
            .sum::<T>()
            .sqrt();
        for i in 0..mat.shape()[1] {
            for j in 0..mat.shape()[2] {
                *mat.get_mut([row, i, j]).unwrap() /= norm;
            }
        }
        for r in row + 1..mat.shape()[0] {
            let dot = (0..mat.shape()[1])
                .map(|i| {
                    (0..mat.shape()[2])
                        .map(|j| *mat.get([row, i, j]).unwrap() * *mat.get([r, i, j]).unwrap())
                        .sum::<T>()
                })
                .sum::<T>();
            for i in 0..mat.shape()[1] {
                for j in 0..mat.shape()[2] {
                    let sub = dot * *mat.get([row, i, j]).unwrap();
                    *mat.get_mut([r, i, j]).unwrap() -= sub;
                }
            }
        }
    }
}

/// Swap two entries in a matrix
unsafe fn entry_swap<const N: usize, T: RlstScalar, ArrayMut: MutableArrayImpl<T, N>>(
    mat: &mut Array<ArrayMut, N>,
    mindex0: [usize; N],
    mindex1: [usize; N],
) {
    unsafe {
        let value = *mat.get_unchecked(mindex0);
        *mat.get_unchecked_mut(mindex0) = *mat.get_unchecked(mindex1);
        *mat.get_unchecked_mut(mindex1) = value;
    }
}

/// Compute the LU decomposition of the transpose of a square matrix
pub fn lu_transpose<T: RlstScalar, Array2MutImpl: MutableArrayImpl<T, 2>>(
    mat: &mut Array<Array2MutImpl, 2>,
) -> Vec<usize> {
    let dim = mat.shape()[0];
    assert_eq!(mat.shape()[1], dim);
    let mut perm = (0..dim).collect::<Vec<_>>();
    if dim > 0 {
        for i in 0..dim - 1 {
            let mut max_col = i;
            let mut max_value = unsafe { mat.get_unchecked([i, i]).abs() };
            for j in i + 1..dim {
                let value = unsafe { mat.get_unchecked([i, j]).abs() };
                if value > max_value {
                    max_col = j;
                    max_value = value;
                }
            }
            for j in 0..dim {
                unsafe {
                    entry_swap(mat, [j, i], [j, max_col]);
                }
            }
            perm.swap(i, max_col);

            let diag = unsafe { *mat.get_unchecked([i, i]) };
            for j in i + 1..dim {
                unsafe {
                    *mat.get_unchecked_mut([i, j]) /= diag;
                }
                for k in i + 1..dim {
                    unsafe {
                        let sub = *mat.get_unchecked([i, j]) * *mat.get_unchecked([k, i]);
                        *mat.get_unchecked_mut([k, j]) -= sub;
                    }
                }
            }
        }
    }
    perm
}

/// Comvert a permutation into the format expected by `apply_permutation`
pub fn prepare_permutation(perm: &mut [usize]) {
    for i in 0..perm.len() {
        while perm[i] < i {
            perm[i] = perm[perm[i]];
        }
    }
}

/// Apply a permutation to some data
pub fn apply_permutation<T>(perm: &[usize], data: &mut [T]) {
    debug_assert!(data.len().is_multiple_of(perm.len()));
    let block_size = data.len() / perm.len();
    for (i, j) in perm.iter().enumerate() {
        for k in 0..block_size {
            data.swap(i * block_size + k, *j * block_size + k);
        }
    }
}

/// Convert a linear transformation info the format expected by `apply_matrix` and return the premutation to pass into `apply_matrix`
pub fn prepare_matrix<T: RlstScalar, Array2Mut: MutableArrayImpl<T, 2>>(
    mat: &mut Array<Array2Mut, 2>,
) -> Vec<usize> {
    let mut perm = lu_transpose(mat);
    prepare_permutation(&mut perm);
    perm
}

/// Apply a permutation and a matrix to some data
pub fn apply_perm_and_matrix<T: RlstScalar, Array2Impl: ValueArrayImpl<T, 2>>(
    mat: &Array<Array2Impl, 2>,
    perm: &[usize],
    data: &mut [T],
) {
    apply_permutation(perm, data);
    apply_matrix(mat, data);
}

/// Apply a matrix to some data
pub fn apply_matrix<T: RlstScalar, Array2Impl: ValueArrayImpl<T, 2>>(
    mat: &Array<Array2Impl, 2>,
    data: &mut [T],
) {
    let dim = mat.shape()[0];
    debug_assert!(data.len().is_multiple_of(dim));
    let block_size = data.len() / dim;
    for i in 0..dim {
        for j in i + 1..dim {
            for k in 0..block_size {
                data[i * block_size + k] +=
                    mat.get_value([i, j]).unwrap() * data[j * block_size + k];
            }
        }
    }
    for i in 1..=dim {
        for k in 0..block_size {
            data[(dim - i) * block_size + k] *= mat.get_value([dim - i, dim - i]).unwrap();
        }
        for j in 0..dim - i {
            for k in 0..block_size {
                data[(dim - i) * block_size + k] +=
                    mat.get_value([dim - i, j]).unwrap() * data[j * block_size + k];
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::*;
    use itertools::izip;
    use rlst::rlst_dynamic_array;

    #[test]
    fn test_permutation() {
        let perm = vec![1, 4, 3, 0, 6, 5, 2];
        let data = vec![9, 4, 1, 5, 3, 2, 10];

        let mut perm2 = perm.clone();
        let mut data2 = data.clone();

        prepare_permutation(&mut perm2);
        apply_permutation(&perm2, &mut data2);
        for (i, p) in perm.iter().enumerate() {
            assert_eq!(data2[i], data[*p]);
        }

        let data = (0..21).map(|i| format!("{i}")).collect::<Vec<_>>();
        let mut data2 = data.clone();

        apply_permutation(&perm2, &mut data2);
        for (i, p) in perm.iter().enumerate() {
            for (a, b) in izip!(&data2[3 * i..3 * i + 3], &data[3 * p..3 * p + 3]) {
                assert_eq!(a, b);
            }
        }
    }

    #[test]
    fn test_matrix_2by2() {
        let mut matrix = rlst_dynamic_array!(f64, [2, 2]);
        matrix[[0, 0]] = 0.5;
        matrix[[0, 1]] = 1.5;
        matrix[[1, 0]] = 1.0;
        matrix[[1, 1]] = 1.0;

        let perm = prepare_matrix(&mut matrix);

        assert_eq!(perm[0], 1);
        assert_eq!(perm[1], 1);

        assert_relative_eq!(*matrix.get([0, 0]).unwrap(), 1.5);
        assert_relative_eq!(*matrix.get([0, 1]).unwrap(), 1.0 / 3.0);
        assert_relative_eq!(*matrix.get([1, 0]).unwrap(), 1.0);
        assert_relative_eq!(*matrix.get([1, 1]).unwrap(), 2.0 / 3.0);

        let mut data = vec![1.0, 2.0];
        apply_perm_and_matrix(&matrix, &perm, &mut data);

        assert_relative_eq!(data[0], 3.5);
        assert_relative_eq!(data[1], 3.0);

        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        apply_perm_and_matrix(&matrix, &perm, &mut data);

        assert_relative_eq!(data[0], 5.0);
        assert_relative_eq!(data[1], 7.0);
        assert_relative_eq!(data[2], 4.0);
        assert_relative_eq!(data[3], 6.0);
    }

    #[test]
    fn test_matrix_3by3() {
        let mut matrix = rlst_dynamic_array!(f64, [3, 3]);
        matrix[[0, 0]] = 0.5;
        matrix[[0, 1]] = 1.5;
        matrix[[0, 2]] = 1.0;
        matrix[[1, 0]] = 1.0;
        matrix[[1, 1]] = 1.0;
        matrix[[1, 2]] = 1.0;
        matrix[[2, 0]] = 0.5;
        matrix[[2, 1]] = 1.0;
        matrix[[2, 2]] = 0.5;

        let perm = prepare_matrix(&mut matrix);

        assert_eq!(perm[0], 1);
        assert_eq!(perm[1], 1);
        assert_eq!(perm[2], 2);

        assert_relative_eq!(*matrix.get([0, 0]).unwrap(), 1.5);
        assert_relative_eq!(*matrix.get([0, 1]).unwrap(), 1.0 / 3.0);
        assert_relative_eq!(*matrix.get([0, 2]).unwrap(), 2.0 / 3.0);
        assert_relative_eq!(*matrix.get([1, 0]).unwrap(), 1.0);
        assert_relative_eq!(*matrix.get([1, 1]).unwrap(), 2.0 / 3.0);
        assert_relative_eq!(*matrix.get([1, 2]).unwrap(), 0.5);
        assert_relative_eq!(*matrix.get([2, 0]).unwrap(), 1.0);
        assert_relative_eq!(*matrix.get([2, 1]).unwrap(), 1.0 / 6.0);
        assert_relative_eq!(*matrix.get([2, 2]).unwrap(), -0.25);

        let mut data = vec![1.0, 2.0, 3.0];
        apply_perm_and_matrix(&matrix, &perm, &mut data);

        assert_relative_eq!(data[0], 6.5);
        assert_relative_eq!(data[1], 6.0);
        assert_relative_eq!(data[2], 4.0);

        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        apply_perm_and_matrix(&matrix, &perm, &mut data);

        assert_relative_eq!(data[0], 10.0);
        assert_relative_eq!(data[1], 13.0);
        assert_relative_eq!(data[2], 9.0);
        assert_relative_eq!(data[3], 12.0);
        assert_relative_eq!(data[4], 6.0);
        assert_relative_eq!(data[5], 8.0);
    }
}
