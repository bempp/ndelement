//! Maps from the reference cell to/from physical cells.
use crate::traits::Map;
use rlst::{Array, MutableArrayImpl, RlstScalar, ValueArrayImpl};

/// Identity map
///
/// An identity map pushes values from the reference to the physical
/// cell without modifying them.
pub struct IdentityMap {}

impl Map for IdentityMap {
    fn push_forward<
        T: RlstScalar,
        TGeo: RlstScalar,
        Array3GeoImpl: ValueArrayImpl<TGeo, 3>,
        Array4Impl: ValueArrayImpl<T, 4>,
        Array4MutImpl: MutableArrayImpl<T, 4>,
    >(
        &self,
        reference_values: &Array<Array4Impl, 4>,
        nderivs: usize,
        _jacobians: &Array<Array3GeoImpl, 3>,
        _jacobian_determinants: &[TGeo],
        _inverse_jacobians: &Array<Array3GeoImpl, 3>,
        physical_values: &mut Array<Array4MutImpl, 4>,
    ) {
        if nderivs > 0 {
            unimplemented!();
        }

        physical_values.fill_from(reference_values);
    }
    fn pull_back<
        T: RlstScalar,
        TGeo: RlstScalar,
        Array3GeoImpl: ValueArrayImpl<TGeo, 3>,
        Array4Impl: ValueArrayImpl<T, 4>,
        Array4MutImpl: MutableArrayImpl<T, 4>,
    >(
        &self,
        physical_values: &Array<Array4Impl, 4>,
        nderivs: usize,
        _jacobians: &Array<Array3GeoImpl, 3>,
        _jacobian_determinants: &[TGeo],
        _inverse_jacobians: &Array<Array3GeoImpl, 3>,
        reference_values: &mut Array<Array4MutImpl, 4>,
    ) {
        if nderivs > 0 {
            unimplemented!();
        }

        reference_values.fill_from(physical_values);
    }
    fn physical_value_shape(&self, _gdim: usize) -> Vec<usize> {
        vec![1]
    }
}

/// Covariant Piola map.
///
/// Let $F$ be the map from the reference cell to the physical cell
/// and let $J$ be its Jacobian. Let $\Phi$ be the function values
/// on the reference cell.  The covariant Piola map is defined by
/// $$
/// J^{-T}\Phi\circ F^{-1}
/// $$
/// The covariant Piola map preserves tangential continuity. If $J$
/// is a rectangular matrix then the pseudo-inverse is used instead of
/// the inverse.
pub struct CovariantPiolaMap {}

impl Map for CovariantPiolaMap {
    fn push_forward<
        T: RlstScalar,
        TGeo: RlstScalar,
        Array3GeoImpl: ValueArrayImpl<TGeo, 3>,
        Array4Impl: ValueArrayImpl<T, 4>,
        Array4MutImpl: MutableArrayImpl<T, 4>,
    >(
        &self,
        reference_values: &Array<Array4Impl, 4>,
        nderivs: usize,
        _jacobians: &Array<Array3GeoImpl, 3>,
        _jacobian_determinants: &[TGeo],
        inverse_jacobians: &Array<Array3GeoImpl, 3>,
        physical_values: &mut Array<Array4MutImpl, 4>,
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
                                T::from(inverse_jacobians.get_value_unchecked([p, td, gd])).unwrap()
                                    * reference_values.get_value_unchecked([0, p, b, td])
                            })
                            .sum::<T>();
                    }
                }
            }
        }
    }
    fn pull_back<
        T: RlstScalar,
        TGeo: RlstScalar,
        Array3GeoImpl: ValueArrayImpl<TGeo, 3>,
        Array4Impl: ValueArrayImpl<T, 4>,
        Array4MutImpl: MutableArrayImpl<T, 4>,
    >(
        &self,
        physical_values: &Array<Array4Impl, 4>,
        nderivs: usize,
        jacobians: &Array<Array3GeoImpl, 3>,
        _jacobian_determinants: &[TGeo],
        _inverse_jacobians: &Array<Array3GeoImpl, 3>,
        reference_values: &mut Array<Array4MutImpl, 4>,
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
                                T::from(jacobians.get_value_unchecked([p, gd, td])).unwrap()
                                    * physical_values.get_value_unchecked([0, p, b, gd])
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

/// Contravariant Piola map.
///
/// Let $F$ be the map from the reference cell to the physical cell
/// and let $J$ be its Jacobian. Let $\Phi$ be the function values
/// on the reference cell.  The contravariant Piola map is defined by
/// $$
/// \frac{1}{\det{J}}J\Phi\circ F^{-1}
/// $$
/// The contravariant Piola map preserves normal continuity. If $J$
/// is a rectangular matrix then $\sqrt{\det{J^TJ}}$ is used instead
/// of $\det{J}$.
pub struct ContravariantPiolaMap {}

impl Map for ContravariantPiolaMap {
    fn push_forward<
        T: RlstScalar,
        TGeo: RlstScalar,
        Array3GeoImpl: ValueArrayImpl<TGeo, 3>,
        Array4Impl: ValueArrayImpl<T, 4>,
        Array4MutImpl: MutableArrayImpl<T, 4>,
    >(
        &self,
        reference_values: &Array<Array4Impl, 4>,
        nderivs: usize,
        jacobians: &Array<Array3GeoImpl, 3>,
        jacobian_determinants: &[TGeo],
        _inverse_jacobians: &Array<Array3GeoImpl, 3>,
        physical_values: &mut Array<Array4MutImpl, 4>,
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
                                T::from(jacobians.get_value_unchecked([p, gd, td])).unwrap()
                                    * reference_values.get_value_unchecked([0, p, b, td])
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
        TGeo: RlstScalar,
        Array3GeoImpl: ValueArrayImpl<TGeo, 3>,
        Array4Impl: ValueArrayImpl<T, 4>,
        Array4MutImpl: MutableArrayImpl<T, 4>,
    >(
        &self,
        physical_values: &Array<Array4Impl, 4>,
        nderivs: usize,
        _jacobians: &Array<Array3GeoImpl, 3>,
        jacobian_determinants: &[TGeo],
        inverse_jacobians: &Array<Array3GeoImpl, 3>,
        reference_values: &mut Array<Array4MutImpl, 4>,
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
                                T::from(inverse_jacobians.get_value_unchecked([p, td, gd])).unwrap()
                                    * physical_values.get_value_unchecked([0, p, b, gd])
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
    use rlst::{Array, rlst_dynamic_array};

    fn fill_jacobians<T: RlstScalar, Array3MutImpl: MutableArrayImpl<T, 3>>(
        j: &mut Array<Array3MutImpl, 3>,
        jdet: &mut [T],
        jinv: &mut Array<Array3MutImpl, 3>,
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
        let mut values = rlst_dynamic_array!(f64, [1, 2, 2, 1]);
        *values.get_mut([0, 0, 0, 0]).unwrap() = 1.0;
        *values.get_mut([0, 1, 0, 0]).unwrap() = 0.0;
        *values.get_mut([0, 0, 1, 0]).unwrap() = 0.5;
        *values.get_mut([0, 1, 1, 0]).unwrap() = 2.0;

        let mut j = rlst_dynamic_array!(f64, [2, 2, 2]);
        let mut jdet = vec![0.0; 2];
        let mut jinv = rlst_dynamic_array!(f64, [2, 2, 2]);
        fill_jacobians(&mut j, &mut jdet, &mut jinv);

        let mut physical_values = rlst_dynamic_array!(f64, [1, 2, 2, 1]);

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

        values.set_zero();

        map.pull_back(&physical_values, 0, &j, &jdet, &jinv, &mut values);

        assert_relative_eq!(*values.get([0, 0, 0, 0]).unwrap(), 1.0, epsilon = 1e-14);
        assert_relative_eq!(*values.get([0, 1, 0, 0]).unwrap(), 0.0, epsilon = 1e-14);
        assert_relative_eq!(*values.get([0, 0, 1, 0]).unwrap(), 0.5, epsilon = 1e-14);
        assert_relative_eq!(*values.get([0, 1, 1, 0]).unwrap(), 2.0, epsilon = 1e-14);
    }

    #[test]
    fn test_covariant_piola() {
        let map = CovariantPiolaMap {};
        let mut values = rlst_dynamic_array!(f64, [1, 2, 2, 2]);
        *values.get_mut([0, 0, 0, 0]).unwrap() = 1.0;
        *values.get_mut([0, 0, 0, 1]).unwrap() = 0.0;
        *values.get_mut([0, 1, 0, 0]).unwrap() = 0.0;
        *values.get_mut([0, 1, 0, 1]).unwrap() = 1.0;
        *values.get_mut([0, 0, 1, 0]).unwrap() = 0.5;
        *values.get_mut([0, 0, 1, 1]).unwrap() = 1.5;
        *values.get_mut([0, 1, 1, 0]).unwrap() = 2.0;
        *values.get_mut([0, 1, 1, 1]).unwrap() = 2.0;

        let mut j = rlst_dynamic_array!(f64, [2, 2, 2]);
        let mut jdet = vec![0.0; 2];
        let mut jinv = rlst_dynamic_array!(f64, [2, 2, 2]);
        fill_jacobians(&mut j, &mut jdet, &mut jinv);

        let mut physical_values = rlst_dynamic_array!(f64, [1, 2, 2, 2]);

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

        values.set_zero();
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
        let mut values = rlst_dynamic_array!(f64, [1, 2, 2, 2]);
        *values.get_mut([0, 0, 0, 0]).unwrap() = 1.0;
        *values.get_mut([0, 0, 0, 1]).unwrap() = 0.0;
        *values.get_mut([0, 1, 0, 0]).unwrap() = 0.0;
        *values.get_mut([0, 1, 0, 1]).unwrap() = 1.0;
        *values.get_mut([0, 0, 1, 0]).unwrap() = 0.5;
        *values.get_mut([0, 0, 1, 1]).unwrap() = 1.5;
        *values.get_mut([0, 1, 1, 0]).unwrap() = 2.0;
        *values.get_mut([0, 1, 1, 1]).unwrap() = 2.0;

        let mut j = rlst_dynamic_array!(f64, [2, 2, 2]);
        let mut jdet = vec![0.0; 2];
        let mut jinv = rlst_dynamic_array!(f64, [2, 2, 2]);
        fill_jacobians(&mut j, &mut jdet, &mut jinv);

        let mut physical_values = rlst_dynamic_array!(f64, [1, 2, 2, 2]);

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

        values.set_zero();
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
