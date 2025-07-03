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
        unimplemented!();
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
        _reference_values: &Array4,
        _nderivs: usize,
        _jacobians: &Array3Real,
        _jacobian_determinants: &[T::Real],
        _inverse_jacobians: &Array3Real,
        _physical_values: &mut Array4Mut,
    ) {
        unimplemented!();
    }
    fn pull_back<
        T: RlstScalar,
        Array3Real: RandomAccessByRef<3, Item = T::Real> + Shape<3>,
        Array4: RandomAccessByRef<4, Item = T> + Shape<4>,
        Array4Mut: RandomAccessMut<4, Item = T> + Shape<4>,
    >(
        &self,
        _physical_values: &Array4,
        _nderivs: usize,
        _jacobians: &Array3Real,
        _jacobian_determinants: &[T::Real],
        _inverse_jacobians: &Array3Real,
        _reference_values: &mut Array4Mut,
    ) {
        unimplemented!();
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
        _reference_values: &Array4,
        _nderivs: usize,
        _jacobians: &Array3Real,
        _jacobian_determinants: &[T::Real],
        _inverse_jacobians: &Array3Real,
        _physical_values: &mut Array4Mut,
    ) {
        unimplemented!();
    }
    fn pull_back<
        T: RlstScalar,
        Array3Real: RandomAccessByRef<3, Item = T::Real> + Shape<3>,
        Array4: RandomAccessByRef<4, Item = T> + Shape<4>,
        Array4Mut: RandomAccessMut<4, Item = T> + Shape<4>,
    >(
        &self,
        _physical_values: &Array4,
        _nderivs: usize,
        _jacobians: &Array3Real,
        _jacobian_determinants: &[T::Real],
        _inverse_jacobians: &Array3Real,
        _reference_values: &mut Array4Mut,
    ) {
        unimplemented!();
    }
    fn physical_value_shape(&self, gdim: usize) -> Vec<usize> {
        vec![gdim]
    }
}
