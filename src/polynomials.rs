//! Orthonormal polynomials
mod legendre;
pub use legendre::{shape as legendre_shape, tabulate as tabulate_legendre_polynomials};

use crate::reference_cell;
use crate::types::ReferenceCellType;

/// The number of polynomials
pub fn polynomial_count(cell_type: ReferenceCellType, degree: usize) -> usize {
    match cell_type {
        ReferenceCellType::Interval => degree + 1,
        ReferenceCellType::Triangle => (degree + 1) * (degree + 2) / 2,
        ReferenceCellType::Quadrilateral => (degree + 1).pow(2),
        ReferenceCellType::Tetrahedron => (degree + 1) * (degree + 2) * (degree + 3) / 6,
        ReferenceCellType::Hexahedron => (degree + 1).pow(3),
        _ => {
            panic!("Unsupported cell type: {cell_type:?}");
        }
    }
}

/// The total number of partial derivatives up to a give degree
pub fn derivative_count(cell_type: ReferenceCellType, derivatives: usize) -> usize {
    let mut num = 1;
    let mut denom = 1;
    for i in 0..reference_cell::dim(cell_type) {
        num *= derivatives + i + 1;
        denom *= i + 1;
    }
    num / denom
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::types::{Array2D, ReferenceCellType};
    use crate::{quadrature::make_gauss_jacobi_quadrature, traits::QuadratureRule};
    use approx::*;
    use paste::paste;
    use rlst::{
        rlst_array_from_slice2, rlst_dynamic_array2, rlst_dynamic_array3, RandomAccessByRef,
        RandomAccessMut, Shape,
    };

    macro_rules! test_orthogonal {
        ($cell:ident, $degree:expr) => {
            paste! {
                #[test]
                fn [<test_orthogonal_ $cell:lower _ $degree>]() {
                    let rule = make_gauss_jacobi_quadrature(
                        ReferenceCellType::[<$cell>],
                        2 * [<$degree>],
                    );


                    let points = rlst_array_from_slice2!(rule.points(), [rule.dim(), rule.npoints()]);

                    let mut data = rlst_dynamic_array3!(
                        f64,
                        legendre_shape(ReferenceCellType::[<$cell>], &points, [<$degree>], 0,)
                    );
                    tabulate_legendre_polynomials(ReferenceCellType::[<$cell>], &points, [<$degree>], 0, &mut data);

                    for i in 0..[<$degree>] + 1 {
                        for j in 0..[<$degree>] + 1 {
                            let mut product = 0.0;
                            for k in 0..rule.npoints() {
                                product += data.get([0, i, k]).unwrap()
                                    * data.get([0, j, k]).unwrap()
                                    * rule.weights()[k];
                            }
                            if i == j {
                                assert_relative_eq!(product, 1.0, epsilon = 1e-12);
                            } else {
                                assert_relative_eq!(product, 0.0, epsilon = 1e-12);
                            }
                        }
                    }
                }
            }
        };
    }

    test_orthogonal!(Interval, 2);
    test_orthogonal!(Interval, 3);
    test_orthogonal!(Interval, 4);
    test_orthogonal!(Interval, 5);
    test_orthogonal!(Interval, 6);
    test_orthogonal!(Triangle, 2);
    test_orthogonal!(Triangle, 3);
    test_orthogonal!(Triangle, 4);
    test_orthogonal!(Triangle, 5);
    test_orthogonal!(Triangle, 6);
    test_orthogonal!(Quadrilateral, 2);
    test_orthogonal!(Quadrilateral, 3);
    test_orthogonal!(Quadrilateral, 4);
    test_orthogonal!(Quadrilateral, 5);
    test_orthogonal!(Quadrilateral, 6);
    test_orthogonal!(Tetrahedron, 2);
    test_orthogonal!(Tetrahedron, 3);
    test_orthogonal!(Tetrahedron, 4);
    test_orthogonal!(Tetrahedron, 5);
    test_orthogonal!(Tetrahedron, 6);
    test_orthogonal!(Hexahedron, 2);
    test_orthogonal!(Hexahedron, 3);
    test_orthogonal!(Hexahedron, 4);
    test_orthogonal!(Hexahedron, 5);
    test_orthogonal!(Hexahedron, 6);

    fn generate_points(cell: ReferenceCellType, epsilon: f64) -> Array2D<f64> {
        let mut points = match cell {
            ReferenceCellType::Interval => {
                let mut points = rlst_dynamic_array2!(f64, [1, 20]);
                for i in 0..10 {
                    *points.get_mut([0, 2 * i]).unwrap() = i as f64 / 10.0;
                }
                points
            }
            ReferenceCellType::Triangle => {
                let mut points = rlst_dynamic_array2!(f64, [2, 165]);
                let mut index = 0;
                for i in 0..10 {
                    for j in 0..10 - i {
                        *points.get_mut([0, 3 * index]).unwrap() = i as f64 / 10.0;
                        *points.get_mut([1, 3 * index]).unwrap() = j as f64 / 10.0;
                        index += 1;
                    }
                }
                points
            }
            ReferenceCellType::Quadrilateral => {
                let mut points = rlst_dynamic_array2!(f64, [2, 300]);
                for i in 0..10 {
                    for j in 0..10 {
                        let index = 10 * i + j;
                        *points.get_mut([0, 3 * index]).unwrap() = i as f64 / 10.0;
                        *points.get_mut([1, 3 * index]).unwrap() = j as f64 / 10.0;
                    }
                }
                points
            }
            ReferenceCellType::Tetrahedron => {
                let mut points = rlst_dynamic_array2!(f64, [3, 140]);
                let mut index = 0;
                for i in 0..5 {
                    for j in 0..5 - i {
                        for k in 0..5 - i - j {
                            *points.get_mut([0, 4 * index]).unwrap() = i as f64 / 5.0;
                            *points.get_mut([1, 4 * index]).unwrap() = j as f64 / 5.0;
                            *points.get_mut([2, 4 * index]).unwrap() = k as f64 / 5.0;
                            index += 1;
                        }
                    }
                }
                points
            }
            ReferenceCellType::Hexahedron => {
                let mut points = rlst_dynamic_array2!(f64, [3, 500]);
                for i in 0..5 {
                    for j in 0..5 {
                        for k in 0..5 {
                            let index = 25 * i + 5 * j + k;
                            *points.get_mut([0, 4 * index]).unwrap() = i as f64 / 5.0;
                            *points.get_mut([1, 4 * index]).unwrap() = j as f64 / 5.0;
                            *points.get_mut([2, 4 * index]).unwrap() = k as f64 / 5.0;
                        }
                    }
                }
                points
            }
            _ => {
                panic!("Unsupported cell type: {cell:?}");
            }
        };
        let dim = reference_cell::dim(cell);
        for index in 0..points.shape()[1] / (dim + 1) {
            for d in 0..dim {
                for i in 0..dim {
                    *points.get_mut([i, (dim + 1) * index + d + 1]).unwrap() =
                        *points.get([i, (dim + 1) * index]).unwrap()
                            + if i == d { epsilon } else { 0.0 };
                }
            }
        }
        points
    }
    macro_rules! test_derivatives {
        ($cell:ident, $degree:expr) => {
            paste! {
                #[test]
                fn [<test_legendre_derivatives_ $cell:lower _ $degree>]() {
                    let dim = reference_cell::dim(ReferenceCellType::[<$cell>]);
                    let epsilon = 1e-10;
                    let points = generate_points(ReferenceCellType::[<$cell>], epsilon);

                    let mut data = rlst_dynamic_array3!(
                        f64,
                        legendre_shape(ReferenceCellType::[<$cell>], &points, [<$degree>], 1,)
                    );
                    tabulate_legendre_polynomials(ReferenceCellType::[<$cell>], &points, [<$degree>], 1, &mut data);

                    for i in 0..data.shape()[1] {
                        for k in 0..points.shape()[1] / (dim + 1) {
                            for d in 1..dim + 1 {
                                assert_relative_eq!(
                                    *data.get([d, i, (dim + 1) * k]).unwrap(),
                                    (data.get([0, i, (dim + 1) * k + d]).unwrap() - data.get([0, i, (dim + 1) * k]).unwrap())
                                        / epsilon,
                                    epsilon = 1e-3
                                );
                            }
                        }
                    }
                }
            }
        };
    }

    test_derivatives!(Interval, 1);
    test_derivatives!(Interval, 2);
    test_derivatives!(Interval, 3);
    test_derivatives!(Interval, 4);
    test_derivatives!(Interval, 5);
    test_derivatives!(Interval, 6);
    test_derivatives!(Triangle, 1);
    test_derivatives!(Triangle, 2);
    test_derivatives!(Triangle, 3);
    test_derivatives!(Triangle, 4);
    test_derivatives!(Triangle, 5);
    test_derivatives!(Triangle, 6);
    test_derivatives!(Quadrilateral, 1);
    test_derivatives!(Quadrilateral, 2);
    test_derivatives!(Quadrilateral, 3);
    test_derivatives!(Quadrilateral, 4);
    test_derivatives!(Quadrilateral, 5);
    test_derivatives!(Quadrilateral, 6);
    test_derivatives!(Tetrahedron, 1);
    test_derivatives!(Tetrahedron, 2);
    test_derivatives!(Tetrahedron, 3);
    test_derivatives!(Tetrahedron, 4);
    test_derivatives!(Tetrahedron, 5);
    test_derivatives!(Tetrahedron, 6);
    test_derivatives!(Hexahedron, 1);
    test_derivatives!(Hexahedron, 2);
    test_derivatives!(Hexahedron, 3);
    test_derivatives!(Hexahedron, 4);
    test_derivatives!(Hexahedron, 5);
    test_derivatives!(Hexahedron, 6);

    #[test]
    fn test_legendre_interval_against_known_polynomials() {
        let degree = 3;

        let mut points = rlst_dynamic_array2!(f64, [1, 11]);
        for i in 0..11 {
            *points.get_mut([0, i]).unwrap() = i as f64 / 10.0;
        }

        let mut data = rlst_dynamic_array3!(
            f64,
            legendre_shape(ReferenceCellType::Interval, &points, degree, 3,)
        );
        tabulate_legendre_polynomials(ReferenceCellType::Interval, &points, degree, 3, &mut data);

        for k in 0..points.shape()[0] {
            let x = *points.get([0, k]).unwrap();

            // 0 => 1
            assert_relative_eq!(*data.get([0, 0, k]).unwrap(), 1.0, epsilon = 1e-12);
            assert_relative_eq!(*data.get([1, 0, k]).unwrap(), 0.0, epsilon = 1e-12);
            assert_relative_eq!(*data.get([2, 0, k]).unwrap(), 0.0, epsilon = 1e-12);
            assert_relative_eq!(*data.get([3, 0, k]).unwrap(), 0.0, epsilon = 1e-12);

            // 1 => sqrt(3)*(2x - 1)
            assert_relative_eq!(
                *data.get([0, 1, k]).unwrap(),
                f64::sqrt(3.0) * (2.0 * x - 1.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([1, 1, k]).unwrap(),
                2.0 * f64::sqrt(3.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(*data.get([2, 1, k]).unwrap(), 0.0, epsilon = 1e-12);
            assert_relative_eq!(*data.get([3, 1, k]).unwrap(), 0.0, epsilon = 1e-12);

            // 2 => sqrt(5)*(6x^2 - 6x + 1)
            assert_relative_eq!(
                *data.get([0, 2, k]).unwrap(),
                f64::sqrt(5.0) * (6.0 * x * x - 6.0 * x + 1.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([1, 2, k]).unwrap(),
                f64::sqrt(5.0) * (12.0 * x - 6.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([2, 2, k]).unwrap(),
                f64::sqrt(5.0) * 12.0,
                epsilon = 1e-12
            );
            assert_relative_eq!(*data.get([3, 2, k]).unwrap(), 0.0, epsilon = 1e-12);

            // 3 => sqrt(7)*(20x^3 - 30x^2 + 12x - 1)
            assert_relative_eq!(
                *data.get([0, 3, k]).unwrap(),
                f64::sqrt(7.0) * (20.0 * x * x * x - 30.0 * x * x + 12.0 * x - 1.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([1, 3, k]).unwrap(),
                f64::sqrt(7.0) * (60.0 * x * x - 60.0 * x + 12.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([2, 3, k]).unwrap(),
                f64::sqrt(7.0) * (120.0 * x - 60.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([3, 3, k]).unwrap(),
                f64::sqrt(7.0) * 120.0,
                epsilon = 1e-12
            );
        }
    }

    #[test]
    fn test_legendre_quadrilateral_against_known_polynomials() {
        let degree = 2;

        let mut points = rlst_dynamic_array2!(f64, [2, 121]);
        for i in 0..11 {
            for j in 0..11 {
                *points.get_mut([0, 11 * i + j]).unwrap() = i as f64 / 10.0;
                *points.get_mut([1, 11 * i + j]).unwrap() = j as f64 / 10.0;
            }
        }

        let mut data = rlst_dynamic_array3!(
            f64,
            legendre_shape(ReferenceCellType::Quadrilateral, &points, degree, 1,)
        );
        tabulate_legendre_polynomials(
            ReferenceCellType::Quadrilateral,
            &points,
            degree,
            1,
            &mut data,
        );

        for k in 0..points.shape()[1] {
            let x = *points.get([0, k]).unwrap();
            let y = *points.get([1, k]).unwrap();

            // 0 => 1
            assert_relative_eq!(*data.get([0, 0, k]).unwrap(), 1.0, epsilon = 1e-12);
            assert_relative_eq!(*data.get([1, 0, k]).unwrap(), 0.0, epsilon = 1e-12);
            assert_relative_eq!(*data.get([2, 0, k]).unwrap(), 0.0, epsilon = 1e-12);

            // 1 => sqrt(3)*(2x - 1)
            assert_relative_eq!(
                *data.get([0, 1, k]).unwrap(),
                f64::sqrt(3.0) * (2.0 * x - 1.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([1, 1, k]).unwrap(),
                2.0 * f64::sqrt(3.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(*data.get([2, 1, k]).unwrap(), 0.0, epsilon = 1e-12);

            // 2 => sqrt(5)*(6x^2 - 6x + 1)
            assert_relative_eq!(
                *data.get([0, 2, k]).unwrap(),
                f64::sqrt(5.0) * (6.0 * x * x - 6.0 * x + 1.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([1, 2, k]).unwrap(),
                f64::sqrt(5.0) * (12.0 * x - 6.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(*data.get([2, 2, k]).unwrap(), 0.0, epsilon = 1e-12);

            // 3 => sqrt(3)*(2y - 1)
            assert_relative_eq!(
                *data.get([0, 3, k]).unwrap(),
                f64::sqrt(3.0) * (2.0 * y - 1.0),
                epsilon = 1e-12
            );

            assert_relative_eq!(*data.get([1, 3, k]).unwrap(), 0.0, epsilon = 1e-12);
            assert_relative_eq!(
                *data.get([2, 3, k]).unwrap(),
                2.0 * f64::sqrt(3.0),
                epsilon = 1e-12
            );

            // 4 => 3*(2x - 1)*(2y - 1)
            assert_relative_eq!(
                *data.get([0, 4, k]).unwrap(),
                3.0 * (2.0 * x - 1.0) * (2.0 * y - 1.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([1, 4, k]).unwrap(),
                6.0 * (2.0 * y - 1.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([2, 4, k]).unwrap(),
                6.0 * (2.0 * x - 1.0),
                epsilon = 1e-12
            );

            // 5 => sqrt(15)*(6x^2 - 6x + 1)*(2y - 1)
            assert_relative_eq!(
                *data.get([0, 5, k]).unwrap(),
                f64::sqrt(15.0) * (6.0 * x * x - 6.0 * x + 1.0) * (2.0 * y - 1.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([1, 5, k]).unwrap(),
                f64::sqrt(15.0) * (12.0 * x - 6.0) * (2.0 * y - 1.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([2, 5, k]).unwrap(),
                2.0 * f64::sqrt(15.0) * (6.0 * x * x - 6.0 * x + 1.0),
                epsilon = 1e-12
            );

            // 6 => sqrt(5)*(6y^2 - 6y + 1)
            assert_relative_eq!(
                *data.get([0, 6, k]).unwrap(),
                f64::sqrt(5.0) * (6.0 * y * y - 6.0 * y + 1.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(*data.get([1, 6, k]).unwrap(), 0.0, epsilon = 1e-12);
            assert_relative_eq!(
                *data.get([2, 6, k]).unwrap(),
                f64::sqrt(5.0) * (12.0 * y - 6.0),
                epsilon = 1e-12
            );

            // 7 => sqrt(15)*(2x - 1)*(6y^2 - 6y + 1)
            assert_relative_eq!(
                *data.get([0, 7, k]).unwrap(),
                f64::sqrt(15.0) * (2.0 * x - 1.0) * (6.0 * y * y - 6.0 * y + 1.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([1, 7, k]).unwrap(),
                2.0 * f64::sqrt(15.0) * (6.0 * y * y - 6.0 * y + 1.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([2, 7, k]).unwrap(),
                f64::sqrt(15.0) * (2.0 * x - 1.0) * (12.0 * y - 6.0),
                epsilon = 1e-12
            );

            // 8 => 5*(6x^2 - 6x + 1)*(6y^2 - 6y + 1)
            assert_relative_eq!(
                *data.get([0, 8, k]).unwrap(),
                5.0 * (6.0 * x * x - 6.0 * x + 1.0) * (6.0 * y * y - 6.0 * y + 1.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([1, 8, k]).unwrap(),
                5.0 * (12.0 * x - 6.0) * (6.0 * y * y - 6.0 * y + 1.0),
                epsilon = 1e-12
            );
            assert_relative_eq!(
                *data.get([2, 8, k]).unwrap(),
                5.0 * (12.0 * y - 6.0) * (6.0 * x * x - 6.0 * x + 1.0),
                epsilon = 1e-12
            );
        }
    }
}
