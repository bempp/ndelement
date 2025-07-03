//! Maps from the reference cell to/from physical cells
use crate::traits::Map;
use rlst::{RandomAccessByRef, RandomAccessMut, RlstScalar, Shape};

/// Identity map
pub struct IdentityMap {}

impl Map for IdentityMap {
    fn push_forward<
        T: RlstScalar,
        Array3Real: RandomAccessByRef<3, Item = T::Real> + Shape<3>,
        Array4: RandomAccessByRef<4, Item = T> + Shape<4>,
        Array4Mut: RandomAccessMut<4, Item = T> + Shape<4>,
    >(
        &self,
        reference_values: &Array4,
        nderivs: usize,
        _jacobians: &Array3Real,
        _jacobian_determinants: &[T::Real],
        _inverse_jacobians: &Array3Real,
        physical_values: &mut Array4Mut,
    ) {
        assert_eq!(reference_values.shape()[0], physical_values.shape()[0]);
        assert_eq!(reference_values.shape()[1], physical_values.shape()[1]);
        assert_eq!(reference_values.shape()[2], physical_values.shape()[2]);
        assert_eq!(reference_values.shape()[3], physical_values.shape()[3]);
        if nderivs > 0 {
            unimplemented!();
        }
        for i0 in 0..reference_values.shape()[0] {
            for i1 in 0..reference_values.shape()[1] {
                for i2 in 0..reference_values.shape()[2] {
                    for i3 in 0..reference_values.shape()[3] {
                        *physical_values.get_mut([i0, i1, i2, i3]).unwrap() =
                            *reference_values.get([i0, i1, i2, i3]).unwrap();
                    }
                }
            }
        }
    }
    fn pull_back<
        T: RlstScalar,
        Array3Real: RandomAccessByRef<3, Item = T::Real> + Shape<3>,
        Array4: RandomAccessByRef<4, Item = T> + Shape<4>,
        Array4Mut: RandomAccessMut<4, Item = T> + Shape<4>,
    >(
        &self,
        physical_values: &Array4,
        nderivs: usize,
        _jacobians: &Array3Real,
        _jacobian_determinants: &[T::Real],
        _inverse_jacobians: &Array3Real,
        reference_values: &mut Array4Mut,
    ) {
        assert_eq!(reference_values.shape()[0], physical_values.shape()[0]);
        assert_eq!(reference_values.shape()[1], physical_values.shape()[1]);
        assert_eq!(reference_values.shape()[2], physical_values.shape()[2]);
        assert_eq!(reference_values.shape()[3], physical_values.shape()[3]);
        if nderivs > 0 {
            unimplemented!();
        }
        for i0 in 0..physical_values.shape()[0] {
            for i1 in 0..physical_values.shape()[1] {
                for i2 in 0..physical_values.shape()[2] {
                    for i3 in 0..physical_values.shape()[3] {
                        *reference_values.get_mut([i0, i1, i2, i3]).unwrap() =
                            *physical_values.get([i0, i1, i2, i3]).unwrap();
                    }
                }
            }
        }
    }
    fn physical_value_shape(&self, _gdim: usize) -> Vec<usize> {
        vec![1]
    }
}

/// CovariantPiola map
pub struct CovariantPiolaMap {}

impl Map for CovariantPiolaMap {
    fn push_forward<
        T: RlstScalar,
        Array3Real: RandomAccessByRef<3, Item = T::Real> + Shape<3>,
        Array4: RandomAccessByRef<4, Item = T> + Shape<4>,
        Array4Mut: RandomAccessMut<4, Item = T> + Shape<4>,
    >(
        &self,
        reference_values: &Array4,
        nderivs: usize,
        _jacobians: &Array3Real,
        _jacobian_determinants: &[T::Real],
        inverse_jacobians: &Array3Real,
        physical_values: &mut Array4Mut,
    ) {
        let tdim = inverse_jacobians.shape()[1];
        let gdim = inverse_jacobians.shape()[2];
        assert_eq!(reference_values.shape()[0], physical_values.shape()[0]);
        assert_eq!(reference_values.shape()[1], physical_values.shape()[1]);
        assert_eq!(reference_values.shape()[2], physical_values.shape()[2]);
        assert_eq!(reference_values.shape()[3], tdim);
        assert_eq!(physical_values.shape()[3], gdim);
        if nderivs > 0 {
            unimplemented!();
        }
        for p in 0..reference_values.shape()[1] {
            for b in 0..reference_values.shape()[2] {
                for gd in 0..gdim {
                    unsafe {
                        *physical_values.get_unchecked_mut([0, p, b, gd]) = (0..tdim)
                            .map(|td| {
                                T::from(*inverse_jacobians.get_unchecked([p, td, gd])).unwrap()
                                    * *reference_values.get_unchecked([0, p, b, td])
                            })
                            .sum::<T>();
                    }
                }
            }
        }
    }
    fn pull_back<
        T: RlstScalar,
        Array3Real: RandomAccessByRef<3, Item = T::Real> + Shape<3>,
        Array4: RandomAccessByRef<4, Item = T> + Shape<4>,
        Array4Mut: RandomAccessMut<4, Item = T> + Shape<4>,
    >(
        &self,
        physical_values: &Array4,
        nderivs: usize,
        jacobians: &Array3Real,
        _jacobian_determinants: &[T::Real],
        _inverse_jacobians: &Array3Real,
        reference_values: &mut Array4Mut,
    ) {
        let gdim = jacobians.shape()[1];
        let tdim = jacobians.shape()[2];
        assert_eq!(reference_values.shape()[0], physical_values.shape()[0]);
        assert_eq!(reference_values.shape()[1], physical_values.shape()[1]);
        assert_eq!(reference_values.shape()[2], physical_values.shape()[2]);
        assert_eq!(reference_values.shape()[3], tdim);
        assert_eq!(physical_values.shape()[3], gdim);
        if nderivs > 0 {
            unimplemented!();
        }
        for p in 0..physical_values.shape()[1] {
            for b in 0..physical_values.shape()[2] {
                for td in 0..tdim {
                    unsafe {
                        *reference_values.get_unchecked_mut([0, p, b, td]) = (0..gdim)
                            .map(|gd| {
                                T::from(*jacobians.get_unchecked([p, gd, td])).unwrap()
                                    * *physical_values.get_unchecked([0, p, b, gd])
                            })
                            .sum::<T>();
                    }
                }
            }
        }
    }
    fn physical_value_shape(&self, gdim: usize) -> Vec<usize> {
        vec![gdim]
    }
}

/// ContravariantPiola map
pub struct ContravariantPiolaMap {}

impl Map for ContravariantPiolaMap {
    fn push_forward<
        T: RlstScalar,
        Array3Real: RandomAccessByRef<3, Item = T::Real> + Shape<3>,
        Array4: RandomAccessByRef<4, Item = T> + Shape<4>,
        Array4Mut: RandomAccessMut<4, Item = T> + Shape<4>,
    >(
        &self,
        reference_values: &Array4,
        nderivs: usize,
        jacobians: &Array3Real,
        jacobian_determinants: &[T::Real],
        _inverse_jacobians: &Array3Real,
        physical_values: &mut Array4Mut,
    ) {
        let gdim = jacobians.shape()[1];
        let tdim = jacobians.shape()[2];
        assert_eq!(reference_values.shape()[0], physical_values.shape()[0]);
        assert_eq!(reference_values.shape()[1], physical_values.shape()[1]);
        assert_eq!(reference_values.shape()[2], physical_values.shape()[2]);
        assert_eq!(reference_values.shape()[3], tdim);
        assert_eq!(physical_values.shape()[3], gdim);
        if nderivs > 0 {
            unimplemented!();
        }
        for p in 0..physical_values.shape()[1] {
            for b in 0..physical_values.shape()[2] {
                for gd in 0..gdim {
                    unsafe {
                        *physical_values.get_unchecked_mut([0, p, b, gd]) = (0..tdim)
                            .map(|td| {
                                T::from(*jacobians.get_unchecked([p, gd, td])).unwrap()
                                    * *reference_values.get_unchecked([0, p, b, td])
                            })
                            .sum::<T>()
                            / T::from(*jacobian_determinants.get_unchecked(p)).unwrap();
                    }
                }
            }
        }
    }
    fn pull_back<
        T: RlstScalar,
        Array3Real: RandomAccessByRef<3, Item = T::Real> + Shape<3>,
        Array4: RandomAccessByRef<4, Item = T> + Shape<4>,
        Array4Mut: RandomAccessMut<4, Item = T> + Shape<4>,
    >(
        &self,
        physical_values: &Array4,
        nderivs: usize,
        _jacobians: &Array3Real,
        jacobian_determinants: &[T::Real],
        inverse_jacobians: &Array3Real,
        reference_values: &mut Array4Mut,
    ) {
        let tdim = inverse_jacobians.shape()[1];
        let gdim = inverse_jacobians.shape()[2];
        assert_eq!(reference_values.shape()[0], physical_values.shape()[0]);
        assert_eq!(reference_values.shape()[1], physical_values.shape()[1]);
        assert_eq!(reference_values.shape()[2], physical_values.shape()[2]);
        assert_eq!(reference_values.shape()[3], tdim);
        assert_eq!(physical_values.shape()[3], gdim);
        if nderivs > 0 {
            unimplemented!();
        }
        for p in 0..physical_values.shape()[1] {
            for b in 0..physical_values.shape()[2] {
                for td in 0..tdim {
                    unsafe {
                        *reference_values.get_unchecked_mut([0, p, b, td]) = (0..gdim)
                            .map(|gd| {
                                T::from(*inverse_jacobians.get_unchecked([p, td, gd])).unwrap()
                                    * *physical_values.get_unchecked([0, p, b, gd])
                            })
                            .sum::<T>()
                            * T::from(*jacobian_determinants.get_unchecked(p)).unwrap();
                    }
                }
            }
        }
    }
    fn physical_value_shape(&self, gdim: usize) -> Vec<usize> {
        vec![gdim]
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::*;
    use rlst::{rlst_dynamic_array3, rlst_dynamic_array4};

    fn set_to_zero<T: RlstScalar, Array4: RandomAccessMut<4, Item = T> + Shape<4>>(
        data: &mut Array4,
    ) {
        for i0 in 0..data.shape()[0] {
            for i1 in 0..data.shape()[1] {
                for i2 in 0..data.shape()[2] {
                    for i3 in 0..data.shape()[3] {
                        *data.get_mut([i0, i1, i2, i3]).unwrap() = T::from(0.0).unwrap();
                    }
                }
            }
        }
    }
    fn fill_jacobians<T: RlstScalar<Real = T>>(
        j: &mut impl RandomAccessMut<3, Item = T>,
        jdet: &mut [T],
        jinv: &mut impl RandomAccessMut<3, Item = T>,
    ) {
        *j.get_mut([0, 0, 0]).unwrap() = T::from(1.0).unwrap();
        *j.get_mut([0, 0, 1]).unwrap() = T::from(1.0).unwrap();
        *j.get_mut([0, 1, 0]).unwrap() = T::from(0.0).unwrap();
        *j.get_mut([0, 1, 1]).unwrap() = T::from(1.0).unwrap();
        *j.get_mut([1, 0, 0]).unwrap() = T::from(2.0).unwrap();
        *j.get_mut([1, 0, 1]).unwrap() = T::from(0.0).unwrap();
        *j.get_mut([1, 1, 0]).unwrap() = T::from(0.0).unwrap();
        *j.get_mut([1, 1, 1]).unwrap() = T::from(3.0).unwrap();
        jdet[0] = T::from(1.0).unwrap();
        jdet[1] = T::from(6.0).unwrap();
        *jinv.get_mut([0, 0, 0]).unwrap() = T::from(1.0).unwrap();
        *jinv.get_mut([0, 0, 1]).unwrap() = T::from(-1.0).unwrap();
        *jinv.get_mut([0, 1, 0]).unwrap() = T::from(0.0).unwrap();
        *jinv.get_mut([0, 1, 1]).unwrap() = T::from(1.0).unwrap();
        *jinv.get_mut([1, 0, 0]).unwrap() = T::from(0.5).unwrap();
        *jinv.get_mut([1, 0, 1]).unwrap() = T::from(0.0).unwrap();
        *jinv.get_mut([1, 1, 0]).unwrap() = T::from(0.0).unwrap();
        *jinv.get_mut([1, 1, 1]).unwrap() = T::from(1.0 / 3.0).unwrap();
    }

    #[test]
    fn test_identity() {
        let map = IdentityMap {};
        let mut values = rlst_dynamic_array4!(f64, [1, 2, 2, 1]);
        *values.get_mut([0, 0, 0, 0]).unwrap() = 1.0;
        *values.get_mut([0, 1, 0, 0]).unwrap() = 0.0;
        *values.get_mut([0, 0, 1, 0]).unwrap() = 0.5;
        *values.get_mut([0, 1, 1, 0]).unwrap() = 2.0;

        let mut j = rlst_dynamic_array3!(f64, [2, 2, 2]);
        let mut jdet = vec![0.0; 2];
        let mut jinv = rlst_dynamic_array3!(f64, [2, 2, 2]);
        fill_jacobians(&mut j, &mut jdet, &mut jinv);

        let mut physical_values = rlst_dynamic_array4!(f64, [1, 2, 2, 1]);

        map.push_forward(&values, 0, &j, &jdet, &jinv, &mut physical_values);

        assert_relative_eq!(
            *physical_values.get([0, 0, 0, 0]).unwrap(),
            1.0,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            *physical_values.get([0, 1, 0, 0]).unwrap(),
            0.0,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            *physical_values.get([0, 0, 1, 0]).unwrap(),
            0.5,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            *physical_values.get([0, 1, 1, 0]).unwrap(),
            2.0,
            epsilon = 1e-14
        );

        set_to_zero(&mut values);
        map.pull_back(&physical_values, 0, &j, &jdet, &jinv, &mut values);

        assert_relative_eq!(*values.get([0, 0, 0, 0]).unwrap(), 1.0, epsilon = 1e-14);
        assert_relative_eq!(*values.get([0, 1, 0, 0]).unwrap(), 0.0, epsilon = 1e-14);
        assert_relative_eq!(*values.get([0, 0, 1, 0]).unwrap(), 0.5, epsilon = 1e-14);
        assert_relative_eq!(*values.get([0, 1, 1, 0]).unwrap(), 2.0, epsilon = 1e-14);
    }

    #[test]
    fn test_covariant_piola() {
        let map = CovariantPiolaMap {};
        let mut values = rlst_dynamic_array4!(f64, [1, 2, 2, 2]);
        *values.get_mut([0, 0, 0, 0]).unwrap() = 1.0;
        *values.get_mut([0, 0, 0, 1]).unwrap() = 0.0;
        *values.get_mut([0, 1, 0, 0]).unwrap() = 0.0;
        *values.get_mut([0, 1, 0, 1]).unwrap() = 1.0;
        *values.get_mut([0, 0, 1, 0]).unwrap() = 0.5;
        *values.get_mut([0, 0, 1, 1]).unwrap() = 1.5;
        *values.get_mut([0, 1, 1, 0]).unwrap() = 2.0;
        *values.get_mut([0, 1, 1, 1]).unwrap() = 2.0;

        let mut j = rlst_dynamic_array3!(f64, [2, 2, 2]);
        let mut jdet = vec![0.0; 2];
        let mut jinv = rlst_dynamic_array3!(f64, [2, 2, 2]);
        fill_jacobians(&mut j, &mut jdet, &mut jinv);

        let mut physical_values = rlst_dynamic_array4!(f64, [1, 2, 2, 2]);

        map.push_forward(&values, 0, &j, &jdet, &jinv, &mut physical_values);

        assert_relative_eq!(
            *physical_values.get([0, 0, 0, 0]).unwrap(),
            1.0,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            *physical_values.get([0, 0, 0, 1]).unwrap(),
            -1.0,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            *physical_values.get([0, 1, 0, 0]).unwrap(),
            0.0,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            *physical_values.get([0, 1, 0, 1]).unwrap(),
            1.0 / 3.0,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            *physical_values.get([0, 0, 1, 0]).unwrap(),
            0.5,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            *physical_values.get([0, 0, 1, 1]).unwrap(),
            1.0,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            *physical_values.get([0, 1, 1, 0]).unwrap(),
            1.0,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            *physical_values.get([0, 1, 1, 1]).unwrap(),
            2.0 / 3.0,
            epsilon = 1e-14
        );

        set_to_zero(&mut values);
        map.pull_back(&physical_values, 0, &j, &jdet, &jinv, &mut values);

        assert_relative_eq!(*values.get([0, 0, 0, 0]).unwrap(), 1.0, epsilon = 1e-14);
        assert_relative_eq!(*values.get([0, 0, 0, 1]).unwrap(), 0.0, epsilon = 1e-14);
        assert_relative_eq!(*values.get([0, 1, 0, 0]).unwrap(), 0.0, epsilon = 1e-14);
        assert_relative_eq!(*values.get([0, 1, 0, 1]).unwrap(), 1.0, epsilon = 1e-14);
        assert_relative_eq!(*values.get([0, 0, 1, 0]).unwrap(), 0.5, epsilon = 1e-14);
        assert_relative_eq!(*values.get([0, 0, 1, 1]).unwrap(), 1.5, epsilon = 1e-14);
        assert_relative_eq!(*values.get([0, 1, 1, 0]).unwrap(), 2.0, epsilon = 1e-14);
        assert_relative_eq!(*values.get([0, 1, 1, 1]).unwrap(), 2.0, epsilon = 1e-14);
    }

    #[test]
    fn test_contravariant_piola() {
        let map = ContravariantPiolaMap {};
        let mut values = rlst_dynamic_array4!(f64, [1, 2, 2, 2]);
        *values.get_mut([0, 0, 0, 0]).unwrap() = 1.0;
        *values.get_mut([0, 0, 0, 1]).unwrap() = 0.0;
        *values.get_mut([0, 1, 0, 0]).unwrap() = 0.0;
        *values.get_mut([0, 1, 0, 1]).unwrap() = 1.0;
        *values.get_mut([0, 0, 1, 0]).unwrap() = 0.5;
        *values.get_mut([0, 0, 1, 1]).unwrap() = 1.5;
        *values.get_mut([0, 1, 1, 0]).unwrap() = 2.0;
        *values.get_mut([0, 1, 1, 1]).unwrap() = 2.0;

        let mut j = rlst_dynamic_array3!(f64, [2, 2, 2]);
        let mut jdet = vec![0.0; 2];
        let mut jinv = rlst_dynamic_array3!(f64, [2, 2, 2]);
        fill_jacobians(&mut j, &mut jdet, &mut jinv);

        let mut physical_values = rlst_dynamic_array4!(f64, [1, 2, 2, 2]);

        map.push_forward(&values, 0, &j, &jdet, &jinv, &mut physical_values);

        assert_relative_eq!(
            *physical_values.get([0, 0, 0, 0]).unwrap(),
            1.0,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            *physical_values.get([0, 0, 0, 1]).unwrap(),
            0.0,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            *physical_values.get([0, 1, 0, 0]).unwrap(),
            0.0,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            *physical_values.get([0, 1, 0, 1]).unwrap(),
            0.5,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            *physical_values.get([0, 0, 1, 0]).unwrap(),
            2.0,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            *physical_values.get([0, 0, 1, 1]).unwrap(),
            1.5,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            *physical_values.get([0, 1, 1, 0]).unwrap(),
            2.0 / 3.0,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            *physical_values.get([0, 1, 1, 1]).unwrap(),
            1.0,
            epsilon = 1e-14
        );

        set_to_zero(&mut values);
        map.pull_back(&physical_values, 0, &j, &jdet, &jinv, &mut values);

        assert_relative_eq!(*values.get([0, 0, 0, 0]).unwrap(), 1.0, epsilon = 1e-14);
        assert_relative_eq!(*values.get([0, 0, 0, 1]).unwrap(), 0.0, epsilon = 1e-14);
        assert_relative_eq!(*values.get([0, 1, 0, 0]).unwrap(), 0.0, epsilon = 1e-14);
        assert_relative_eq!(*values.get([0, 1, 0, 1]).unwrap(), 1.0, epsilon = 1e-14);
        assert_relative_eq!(*values.get([0, 0, 1, 0]).unwrap(), 0.5, epsilon = 1e-14);
        assert_relative_eq!(*values.get([0, 0, 1, 1]).unwrap(), 1.5, epsilon = 1e-14);
        assert_relative_eq!(*values.get([0, 1, 1, 0]).unwrap(), 2.0, epsilon = 1e-14);
        assert_relative_eq!(*values.get([0, 1, 1, 1]).unwrap(), 2.0, epsilon = 1e-14);
    }
}
