//! Mathematical functions
use rlst::{
    RandomAccessByRef, RandomAccessMut, RlstScalar, Shape, UnsafeRandomAccessByRef,
    UnsafeRandomAccessMut,
};

/// Orthogonalise the rows of a matrix, starting with the row numbered `start`
pub fn orthogonalise<
    T: RlstScalar,
    Array2: RandomAccessByRef<2, Item = T> + RandomAccessMut<2, Item = T> + Shape<2>,
>(
    mat: &mut Array2,
    start: usize,
) {
    for row in start..mat.shape()[0] {
        let norm = (0..mat.shape()[1])
            .map(|i| mat.get([row, i]).unwrap().powi(2))
            .sum::<T>()
            .sqrt();
        for i in 0..mat.shape()[1] {
            *mat.get_mut([row, i]).unwrap() /= norm;
        }
        for r in row + 1..mat.shape()[0] {
            let dot = (0..mat.shape()[1])
                .map(|i| *mat.get([row, i]).unwrap() * *mat.get([r, i]).unwrap())
                .sum::<T>();
            for i in 0..mat.shape()[1] {
                let sub = dot * *mat.get([row, i]).unwrap();
                *mat.get_mut([r, i]).unwrap() -= sub;
            }
        }
    }
}

/// Orthogonalise the rows of a matrix, starting with the row numbered `start`
pub fn orthogonalise_3<
    T: RlstScalar,
    Array3: RandomAccessByRef<3, Item = T> + RandomAccessMut<3, Item = T> + Shape<3>,
>(
    mat: &mut Array3,
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
unsafe fn entry_swap<
    const N: usize,
    T: RlstScalar,
    ArrayMut: UnsafeRandomAccessMut<N, Item = T> + UnsafeRandomAccessByRef<N, Item = T> + Shape<N>,
>(
    mat: &mut ArrayMut,
    mindex0: [usize; N],
    mindex1: [usize; N],
) {
    let value = *mat.get_unchecked(mindex0);
    *mat.get_unchecked_mut(mindex0) = *mat.get_unchecked(mindex1);
    *mat.get_unchecked_mut(mindex1) = value;
}

/// Compute the LU decomposition of the transpose of a square matrix
pub fn lu_transpose<
    T: RlstScalar,
    Array2Mut: UnsafeRandomAccessMut<2, Item = T> + UnsafeRandomAccessByRef<2, Item = T> + Shape<2>,
>(
    mat: &mut Array2Mut,
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
    for (i, j) in perm.iter().enumerate() {
        data.swap(i, *j);
    }
}

/// Convert a linear transformation info the format expected by `apply_matrix` and return the premutation to pass into `apply_matrix`
pub fn prepare_matrix<
    T: RlstScalar,
    Array2Mut: UnsafeRandomAccessMut<2, Item = T> + UnsafeRandomAccessByRef<2, Item = T> + Shape<2>,
>(
    mat: &mut Array2Mut,
) -> Vec<usize> {
    let mut perm = lu_transpose(mat);
    prepare_permutation(&mut perm);
    perm
}

/// Apply a permutation and a matrix to some data
pub fn apply_perm_and_matrix<T: RlstScalar, Array2: RandomAccessByRef<2, Item = T> + Shape<2>>(
    mat: &Array2,
    perm: &[usize],
    data: &mut [T],
) {
    apply_permutation(perm, data);
    apply_matrix(mat, data);
}

/// Apply a matrix to some data
pub fn apply_matrix<T: RlstScalar, Array2: RandomAccessByRef<2, Item = T> + Shape<2>>(
    mat: &Array2,
    data: &mut [T],
) {
    let dim = mat.shape()[0];
    for i in 0..dim {
        for j in i + 1..dim {
            data[i] += *mat.get([i, j]).unwrap() * data[j];
        }
    }
    for i in 1..=dim {
        data[dim - i] *= *mat.get([dim - i, dim - i]).unwrap();
        for j in 0..dim - i {
            data[dim - i] += *mat.get([dim - i, j]).unwrap() * data[j];
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::*;
    use rlst::rlst_dynamic_array2;

    #[test]
    fn test_permutation() {
        let perm = vec![1, 4, 3, 0, 6, 5, 2];
        let data = vec![9, 4, 1, 5, 3, 2, 10];

        let mut perm2 = perm.clone();
        let mut data2 = data.clone();

        prepare_permutation(&mut perm2);
        apply_permutation(&perm2, &mut data2);
        for (i, d) in data2.iter().enumerate() {
            assert_eq!(*d, data[perm[i]]);
        }
    }

    #[test]
    fn test_matrix_2by2() {
        let mut matrix = rlst_dynamic_array2!(f64, [2, 2]);
        matrix[[0, 0]] = 0.5;
        matrix[[0, 1]] = 1.5;
        matrix[[1, 0]] = 1.0;
        matrix[[1, 1]] = 1.0;

        let perm = prepare_matrix(&mut matrix);

        let mut data = vec![1.0, 2.0];
        apply_perm_and_matrix(&matrix, &perm, &mut data);

        assert_eq!(perm[0], 1);
        assert_eq!(perm[1], 1);

        assert_relative_eq!(*matrix.get([0, 0]).unwrap(), 1.5);
        assert_relative_eq!(*matrix.get([0, 1]).unwrap(), 1.0 / 3.0);
        assert_relative_eq!(*matrix.get([1, 0]).unwrap(), 1.0);
        assert_relative_eq!(*matrix.get([1, 1]).unwrap(), 2.0 / 3.0);

        assert_relative_eq!(data[0], 3.5);
        assert_relative_eq!(data[1], 3.0);
    }

    #[test]
    fn test_matrix_3by3() {
        let mut matrix = rlst_dynamic_array2!(f64, [3, 3]);
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

        let mut data = vec![1.0, 2.0, 3.0];
        apply_perm_and_matrix(&matrix, &perm, &mut data);

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

        assert_relative_eq!(data[0], 6.5);
        assert_relative_eq!(data[1], 6.0);
        assert_relative_eq!(data[2], 4.0);
    }
}
