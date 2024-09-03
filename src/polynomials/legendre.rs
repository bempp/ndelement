//! Orthonormal polynomials
//!
//! Adapted from the C++ code by Chris Richardson and Matthew Scroggs
//! <https://github.com/FEniCS/basix/blob/main/cpp/basix/polyset.cpp>
use super::{derivative_count, polynomial_count};
use crate::types::ReferenceCellType;
use rlst::RlstScalar;
use rlst::{RandomAccessByRef, RandomAccessMut, Shape};

fn tri_index(i: usize, j: usize) -> usize {
    (i + j + 1) * (i + j) / 2 + j
}

fn quad_index(i: usize, j: usize, n: usize) -> usize {
    j * (n + 1) + i
}

fn tet_index(i: usize, j: usize, k: usize) -> usize {
    (i + j + k) * (i + j + k + 1) * (i + j + k + 2) / 6 + (j + k) * (j + k + 1) / 2 + k
}

fn hex_index(i: usize, j: usize, k: usize, n: usize) -> usize {
    k * (n + 1) * (n + 1) + j * (n + 1) + i
}

/// The coefficients in the Jacobi Polynomial recurrence relation
fn jrc<T: RlstScalar>(a: usize, n: usize) -> (T, T, T) {
    (
        T::from((a + 2 * n + 1) * (a + 2 * n + 2)).unwrap()
            / T::from(2 * (n + 1) * (a + n + 1)).unwrap(),
        T::from(a * a * (a + 2 * n + 1)).unwrap()
            / T::from(2 * (n + 1) * (a + n + 1) * (a + 2 * n)).unwrap(),
        T::from(n * (a + n) * (a + 2 * n + 2)).unwrap()
            / T::from((n + 1) * (a + n + 1) * (a + 2 * n)).unwrap(),
    )
}

/// Tabulate orthonormal polynomials on an interval
fn tabulate_interval<
    T: RlstScalar,
    Array2: RandomAccessByRef<2, Item = T::Real> + Shape<2>,
    Array3Mut: RandomAccessMut<3, Item = T> + RandomAccessByRef<3, Item = T> + Shape<3>,
>(
    points: &Array2,
    degree: usize,
    derivatives: usize,
    data: &mut Array3Mut,
) {
    debug_assert!(data.shape()[0] == derivatives + 1);
    debug_assert!(data.shape()[1] == degree + 1);
    debug_assert!(data.shape()[2] == points.shape()[1]);
    debug_assert!(points.shape()[0] == 1);

    for i in 0..data.shape()[2] {
        *data.get_mut([0, 0, i]).unwrap() = T::from(1.0).unwrap();
    }
    for k in 1..data.shape()[0] {
        for i in 0..data.shape()[2] {
            *data.get_mut([k, 0, i]).unwrap() = T::from(0.0).unwrap();
        }
    }

    for k in 0..=derivatives {
        for p in 1..=degree {
            let a = T::from(p - 1).unwrap() / T::from(p).unwrap();
            let b = (a + T::from(1.0).unwrap())
                * ((T::from(2.0).unwrap() * T::from(p).unwrap() + T::from(1.0).unwrap())
                    / (T::from(2.0).unwrap() * T::from(p).unwrap() - T::from(1.0).unwrap()))
                .sqrt();
            for i in 0..data.shape()[2] {
                let d = *data.get([k, p - 1, i]).unwrap();
                *data.get_mut([k, p, i]).unwrap() =
                    (T::from(*points.get([0, i]).unwrap()).unwrap() * T::from(2.0).unwrap()
                        - T::from(1.0).unwrap())
                        * d
                        * b;
            }
            if p > 1 {
                let c = a
                    * ((T::from(2.0).unwrap() * T::from(p).unwrap() + T::from(1.0).unwrap())
                        / (T::from(2.0).unwrap() * T::from(p).unwrap() - T::from(3.0).unwrap()))
                    .sqrt();
                for i in 0..data.shape()[2] {
                    let d = *data.get([k, p - 2, i]).unwrap();
                    *data.get_mut([k, p, i]).unwrap() -= d * c;
                }
            }
            if k > 0 {
                for i in 0..data.shape()[2] {
                    let d = *data.get([k - 1, p - 1, i]).unwrap();
                    *data.get_mut([k, p, i]).unwrap() +=
                        T::from(2.0).unwrap() * T::from(k).unwrap() * d * b;
                }
            }
        }
    }
}

/// Tabulate orthonormal polynomials on a quadrilateral
fn tabulate_quadrilateral<
    T: RlstScalar,
    Array2: RandomAccessByRef<2, Item = T::Real> + Shape<2>,
    Array3Mut: RandomAccessMut<3, Item = T> + RandomAccessByRef<3, Item = T> + Shape<3>,
>(
    points: &Array2,
    degree: usize,
    derivatives: usize,
    data: &mut Array3Mut,
) {
    debug_assert!(data.shape()[0] == (derivatives + 1) * (derivatives + 2) / 2);
    debug_assert!(data.shape()[1] == (degree + 1) * (degree + 1));
    debug_assert!(data.shape()[2] == points.shape()[1]);
    debug_assert!(points.shape()[0] == 2);

    for i in 0..data.shape()[2] {
        *data
            .get_mut([tri_index(0, 0), quad_index(0, 0, degree), i])
            .unwrap() = T::from(1.0).unwrap();
    }

    // Tabulate polynomials in x
    for k in 1..=derivatives {
        for i in 0..data.shape()[2] {
            *data
                .get_mut([tri_index(k, 0), quad_index(0, 0, degree), i])
                .unwrap() = T::from(0.0).unwrap();
        }
    }

    for k in 0..=derivatives {
        for p in 1..=degree {
            let a = T::from(1.0).unwrap() - T::from(1.0).unwrap() / T::from(p).unwrap();
            let b = (a + T::from(1.0).unwrap())
                * ((T::from(2.0).unwrap() * T::from(p).unwrap() + T::from(1.0).unwrap())
                    / (T::from(2.0).unwrap() * T::from(p).unwrap() - T::from(1.0).unwrap()))
                .sqrt();
            for i in 0..data.shape()[2] {
                let d = *data
                    .get([tri_index(k, 0), quad_index(p - 1, 0, degree), i])
                    .unwrap();
                *data
                    .get_mut([tri_index(k, 0), quad_index(p, 0, degree), i])
                    .unwrap() = (T::from(*points.get([0, i]).unwrap()).unwrap()
                    * T::from(2.0).unwrap()
                    - T::from(1.0).unwrap())
                    * d
                    * b;
            }
            if p > 1 {
                let c = a
                    * ((T::from(2.0).unwrap() * T::from(p).unwrap() + T::from(1.0).unwrap())
                        / (T::from(2.0).unwrap() * T::from(p).unwrap() - T::from(3.0).unwrap()))
                    .sqrt();
                for i in 0..data.shape()[2] {
                    let d = *data
                        .get([tri_index(k, 0), quad_index(p - 2, 0, degree), i])
                        .unwrap();
                    *data
                        .get_mut([tri_index(k, 0), quad_index(p, 0, degree), i])
                        .unwrap() -= d * c;
                }
            }
            if k > 0 {
                for i in 0..data.shape()[2] {
                    let d = *data
                        .get([tri_index(k - 1, 0), quad_index(p - 1, 0, degree), i])
                        .unwrap();
                    *data
                        .get_mut([tri_index(k, 0), quad_index(p, 0, degree), i])
                        .unwrap() += T::from(2.0).unwrap() * T::from(k).unwrap() * d * b;
                }
            }
        }
    }

    // Tabulate polynomials in y
    for k in 1..=derivatives {
        for i in 0..data.shape()[2] {
            *data
                .get_mut([tri_index(0, k), quad_index(0, 0, degree), i])
                .unwrap() = T::from(0.0).unwrap();
        }
    }

    for k in 0..=derivatives {
        for p in 1..=degree {
            let a = T::from(1.0).unwrap() - T::from(1.0).unwrap() / T::from(p).unwrap();
            let b = (a + T::from(1.0).unwrap())
                * ((T::from(2.0).unwrap() * T::from(p).unwrap() + T::from(1.0).unwrap())
                    / (T::from(2.0).unwrap() * T::from(p).unwrap() - T::from(1.0).unwrap()))
                .sqrt();
            for i in 0..data.shape()[2] {
                let d = *data
                    .get([tri_index(0, k), quad_index(0, p - 1, degree), i])
                    .unwrap();
                *data
                    .get_mut([tri_index(0, k), quad_index(0, p, degree), i])
                    .unwrap() = (T::from(*points.get([1, i]).unwrap()).unwrap()
                    * T::from(2.0).unwrap()
                    - T::from(1.0).unwrap())
                    * d
                    * b;
            }
            if p > 1 {
                let c = a
                    * ((T::from(2.0).unwrap() * T::from(p).unwrap() + T::from(1.0).unwrap())
                        / (T::from(2.0).unwrap() * T::from(p).unwrap() - T::from(3.0).unwrap()))
                    .sqrt();
                for i in 0..data.shape()[2] {
                    let d = *data
                        .get([tri_index(0, k), quad_index(0, p - 2, degree), i])
                        .unwrap();
                    *data
                        .get_mut([tri_index(0, k), quad_index(0, p, degree), i])
                        .unwrap() -= d * c;
                }
            }
            if k > 0 {
                for i in 0..data.shape()[2] {
                    let d = *data
                        .get([tri_index(0, k - 1), quad_index(0, p - 1, degree), i])
                        .unwrap();
                    *data
                        .get_mut([tri_index(0, k), quad_index(0, p, degree), i])
                        .unwrap() += T::from(2.0).unwrap() * T::from(k).unwrap() * d * b;
                }
            }
        }
    }

    // Fill in the rest of the values as products
    for kx in 0..=derivatives {
        for ky in 0..=derivatives - kx {
            for px in 1..=degree {
                for py in 1..=degree {
                    for i in 0..data.shape()[2] {
                        let d = *data
                            .get([tri_index(0, ky), quad_index(0, py, degree), i])
                            .unwrap();
                        *data
                            .get_mut([tri_index(kx, ky), quad_index(px, py, degree), i])
                            .unwrap() = *data
                            .get([tri_index(kx, 0), quad_index(px, 0, degree), i])
                            .unwrap()
                            * d;
                    }
                }
            }
        }
    }
}
/// Tabulate orthonormal polynomials on a triangle
fn tabulate_triangle<
    T: RlstScalar,
    Array2: RandomAccessByRef<2, Item = T::Real> + Shape<2>,
    Array3Mut: RandomAccessMut<3, Item = T> + RandomAccessByRef<3, Item = T> + Shape<3>,
>(
    points: &Array2,
    degree: usize,
    derivatives: usize,
    data: &mut Array3Mut,
) {
    debug_assert!(data.shape()[0] == (derivatives + 1) * (derivatives + 2) / 2);
    debug_assert!(data.shape()[1] == (degree + 1) * (degree + 2) / 2);
    debug_assert!(data.shape()[2] == points.shape()[1]);
    debug_assert!(points.shape()[0] == 2);

    for i in 0..data.shape()[2] {
        *data.get_mut([tri_index(0, 0), tri_index(0, 0), i]).unwrap() =
            T::sqrt(T::from(2.0).unwrap());
    }

    for k in 1..data.shape()[0] {
        for i in 0..data.shape()[2] {
            *data.get_mut([k, tri_index(0, 0), i]).unwrap() = T::from(0.0).unwrap();
        }
    }

    for kx in 0..=derivatives {
        for ky in 0..=derivatives - kx {
            for p in 1..=degree {
                let a = T::from(2.0).unwrap() - T::from(1.0).unwrap() / T::from(p).unwrap();
                let scale1 = T::sqrt(
                    (T::from(p).unwrap() + T::from(0.5).unwrap())
                        * (T::from(p).unwrap() + T::from(1.0).unwrap())
                        / ((T::from(p).unwrap() - T::from(0.5).unwrap()) * T::from(p).unwrap()),
                );
                for i in 0..data.shape()[2] {
                    let d = *data
                        .get([tri_index(kx, ky), tri_index(0, p - 1), i])
                        .unwrap();
                    *data
                        .get_mut([tri_index(kx, ky), tri_index(0, p), i])
                        .unwrap() = (T::from(*points.get([0, i]).unwrap()).unwrap()
                        * T::from(2.0).unwrap()
                        + T::from(*points.get([1, i]).unwrap()).unwrap()
                        - T::from(1.0).unwrap())
                        * d
                        * a
                        * scale1;
                }
                if kx > 0 {
                    for i in 0..data.shape()[2] {
                        let d = *data
                            .get([tri_index(kx - 1, ky), tri_index(0, p - 1), i])
                            .unwrap();
                        *data
                            .get_mut([tri_index(kx, ky), tri_index(0, p), i])
                            .unwrap() +=
                            T::from(2.0).unwrap() * T::from(kx).unwrap() * a * d * scale1;
                    }
                }
                if ky > 0 {
                    for i in 0..data.shape()[2] {
                        let d = *data
                            .get([tri_index(kx, ky - 1), tri_index(0, p - 1), i])
                            .unwrap();
                        *data
                            .get_mut([tri_index(kx, ky), tri_index(0, p), i])
                            .unwrap() += T::from(ky).unwrap() * a * d * scale1;
                    }
                }
                if p > 1 {
                    let scale2 = T::sqrt(
                        (T::from(p).unwrap() + T::from(0.5).unwrap())
                            * (T::from(p).unwrap() + T::from(1.0).unwrap()),
                    ) / T::sqrt(
                        (T::from(p).unwrap() - T::from(1.5).unwrap())
                            * (T::from(p).unwrap() - T::from(1.0).unwrap()),
                    );

                    for i in 0..data.shape()[2] {
                        let b =
                            T::from(1.0).unwrap() - T::from(*points.get([1, i]).unwrap()).unwrap();
                        let d = *data
                            .get([tri_index(kx, ky), tri_index(0, p - 2), i])
                            .unwrap();
                        *data
                            .get_mut([tri_index(kx, ky), tri_index(0, p), i])
                            .unwrap() -= b * b * d * (a - T::from(1.0).unwrap()) * scale2;
                    }
                    if ky > 0 {
                        for i in 0..data.shape()[2] {
                            let d = *data
                                .get([tri_index(kx, ky - 1), tri_index(0, p - 2), i])
                                .unwrap();
                            *data
                                .get_mut([tri_index(kx, ky), tri_index(0, p), i])
                                .unwrap() -= T::from(2.0).unwrap()
                                * T::from(ky).unwrap()
                                * (T::from(*points.get([1, i]).unwrap()).unwrap()
                                    - T::from(1.0).unwrap())
                                * d
                                * scale2
                                * (a - T::from(1.0).unwrap());
                        }
                    }
                    if ky > 1 {
                        for i in 0..data.shape()[2] {
                            let d = *data
                                .get([tri_index(kx, ky - 2), tri_index(0, p - 2), i])
                                .unwrap();
                            *data
                                .get_mut([tri_index(kx, ky), tri_index(0, p), i])
                                .unwrap() -= T::from(ky).unwrap()
                                * (T::from(ky).unwrap() - T::from(1.0).unwrap())
                                * d
                                * scale2
                                * (a - T::from(1.0).unwrap());
                        }
                    }
                }
            }
            for p in 0..degree {
                let scale3 = T::sqrt(
                    (T::from(p).unwrap() + T::from(2.0).unwrap())
                        / (T::from(p).unwrap() + T::from(1.0).unwrap()),
                );
                for i in 0..data.shape()[2] {
                    *data
                        .get_mut([tri_index(kx, ky), tri_index(1, p), i])
                        .unwrap() = *data.get([tri_index(kx, ky), tri_index(0, p), i]).unwrap()
                        * scale3
                        * ((T::from(*points.get([1, i]).unwrap()).unwrap()
                            * T::from(2.0).unwrap()
                            - T::from(1.0).unwrap())
                            * (T::from(1.5).unwrap() + T::from(p).unwrap())
                            + T::from(0.5).unwrap()
                            + T::from(p).unwrap());
                }
                if ky > 0 {
                    for i in 0..data.shape()[2] {
                        let d = *data
                            .get([tri_index(kx, ky - 1), tri_index(0, p), i])
                            .unwrap();
                        *data
                            .get_mut([tri_index(kx, ky), tri_index(1, p), i])
                            .unwrap() += T::from(2.0).unwrap()
                            * T::from(ky).unwrap()
                            * (T::from(1.5).unwrap() + T::from(p).unwrap())
                            * d
                            * scale3;
                    }
                }
                for q in 1..degree - p {
                    let scale4 = T::sqrt(
                        (T::from(p).unwrap() + T::from(q).unwrap() + T::from(2.0).unwrap())
                            / (T::from(p).unwrap() + T::from(q).unwrap() + T::from(1.0).unwrap()),
                    );
                    let scale5 = T::sqrt(
                        (T::from(p).unwrap() + T::from(q).unwrap() + T::from(2.0).unwrap())
                            / (T::from(p).unwrap() + T::from(q).unwrap()),
                    );
                    let (a1, a2, a3) = jrc(2 * p + 1, q);

                    for i in 0..data.shape()[2] {
                        let d = *data.get([tri_index(kx, ky), tri_index(q, p), i]).unwrap();
                        *data
                            .get_mut([tri_index(kx, ky), tri_index(q + 1, p), i])
                            .unwrap() = d
                            * scale4
                            * ((T::from(*points.get([1, i]).unwrap()).unwrap()
                                * T::from(T::from(2.0).unwrap()).unwrap()
                                - T::from(T::from(1.0).unwrap()).unwrap())
                                * a1
                                + a2)
                            - *data
                                .get([tri_index(kx, ky), tri_index(q - 1, p), i])
                                .unwrap()
                                * scale5
                                * a3;
                    }
                    if ky > 0 {
                        for i in 0..data.shape()[2] {
                            let d = *data
                                .get([tri_index(kx, ky - 1), tri_index(q, p), i])
                                .unwrap();
                            *data
                                .get_mut([tri_index(kx, ky), tri_index(q + 1, p), i])
                                .unwrap() += T::from(T::from(2.0).unwrap() * T::from(ky).unwrap())
                                .unwrap()
                                * a1
                                * d
                                * scale4;
                        }
                    }
                }
            }
        }
    }
}

/// Tabulate orthonormal polynomials on a tetrahedron
fn tabulate_tetrahedron<
    T: RlstScalar,
    Array2: RandomAccessByRef<2, Item = T::Real> + Shape<2>,
    Array3Mut: RandomAccessMut<3, Item = T> + RandomAccessByRef<3, Item = T> + Shape<3>,
>(
    points: &Array2,
    degree: usize,
    derivatives: usize,
    data: &mut Array3Mut,
) {
    debug_assert!(data.shape()[0] == (derivatives + 1) * (derivatives + 2) * (derivatives + 3) / 6);
    debug_assert!(data.shape()[1] == (degree + 1) * (degree + 2) * (degree + 3) / 6);
    debug_assert!(data.shape()[2] == points.shape()[1]);
    debug_assert!(points.shape()[0] == 3);

    for i in 0..data.shape()[2] {
        *data
            .get_mut([tet_index(0, 0, 0), tet_index(0, 0, 0), i])
            .unwrap() = T::sqrt(T::from(6.0).unwrap());
    }

    for k in 1..data.shape()[0] {
        for i in 0..data.shape()[2] {
            *data.get_mut([k, tet_index(0, 0, 0), i]).unwrap() = T::from(0.0).unwrap();
        }
    }

    for kx in 0..=derivatives {
        for ky in 0..=derivatives - kx {
            for kz in 0..=derivatives - kx - ky {
                for p in 1..=degree {
                    let a = T::from(2 * p - 1).unwrap() / T::from(p).unwrap();
                    for i in 0..points.shape()[1] {
                        let d = *data
                            .get([tet_index(kx, ky, kz), tet_index(0, 0, p - 1), i])
                            .unwrap();
                        *data
                            .get_mut([tet_index(kx, ky, kz), tet_index(0, 0, p), i])
                            .unwrap() = (T::from(*points.get([0, i]).unwrap()).unwrap()
                            * T::from(2.0).unwrap()
                            + T::from(*points.get([1, i]).unwrap()).unwrap()
                            + T::from(*points.get([2, i]).unwrap()).unwrap()
                            - T::from(1.0).unwrap())
                            * a
                            * d;
                    }
                    if kx > 0 {
                        for i in 0..points.shape()[1] {
                            let d = *data
                                .get([tet_index(kx - 1, ky, kz), tet_index(0, 0, p - 1), i])
                                .unwrap();
                            *data
                                .get_mut([tet_index(kx, ky, kz), tet_index(0, 0, p), i])
                                .unwrap() += T::from(2 * kx).unwrap() * a * d;
                        }
                    }
                    if ky > 0 {
                        for i in 0..points.shape()[1] {
                            let d = *data
                                .get([tet_index(kx, ky - 1, kz), tet_index(0, 0, p - 1), i])
                                .unwrap();
                            *data
                                .get_mut([tet_index(kx, ky, kz), tet_index(0, 0, p), i])
                                .unwrap() += T::from(ky).unwrap() * a * d;
                        }
                    }
                    if kz > 0 {
                        for i in 0..points.shape()[1] {
                            let d = *data
                                .get([tet_index(kx, ky, kz - 1), tet_index(0, 0, p - 1), i])
                                .unwrap();
                            *data
                                .get_mut([tet_index(kx, ky, kz), tet_index(0, 0, p), i])
                                .unwrap() += T::from(kz).unwrap() * a * d;
                        }
                    }
                    if p > 1 {
                        for i in 0..points.shape()[1] {
                            let d = *data
                                .get([tet_index(kx, ky, kz), tet_index(0, 0, p - 2), i])
                                .unwrap();
                            *data
                                .get_mut([tet_index(kx, ky, kz), tet_index(0, 0, p), i])
                                .unwrap() -= (T::from(
                                *points.get([1, i]).unwrap() + *points.get([2, i]).unwrap(),
                            )
                            .unwrap()
                                - T::from(1.0).unwrap())
                            .powi(2)
                                * d
                                * (a - T::from(1.0).unwrap());
                        }
                        if ky > 0 {
                            for i in 0..points.shape()[1] {
                                let d = *data
                                    .get([tet_index(kx, ky - 1, kz), tet_index(0, 0, p - 2), i])
                                    .unwrap();
                                *data
                                    .get_mut([tet_index(kx, ky, kz), tet_index(0, 0, p), i])
                                    .unwrap() -= T::from(ky * 2).unwrap()
                                    * (T::from(
                                        *points.get([1, i]).unwrap() + *points.get([2, i]).unwrap(),
                                    )
                                    .unwrap()
                                        - T::from(1.0).unwrap())
                                    * d
                                    * (a - T::from(1.0).unwrap());
                            }
                        }
                        if ky > 1 {
                            for i in 0..points.shape()[1] {
                                let d = *data
                                    .get([tet_index(kx, ky - 2, kz), tet_index(0, 0, p - 2), i])
                                    .unwrap();
                                *data
                                    .get_mut([tet_index(kx, ky, kz), tet_index(0, 0, p), i])
                                    .unwrap() -= T::from(ky * (ky - 1)).unwrap()
                                    * d
                                    * (a - T::from(1.0).unwrap());
                            }
                        }
                        if kz > 0 {
                            for i in 0..points.shape()[1] {
                                let d = *data
                                    .get([tet_index(kx, ky, kz - 1), tet_index(0, 0, p - 2), i])
                                    .unwrap();
                                *data
                                    .get_mut([tet_index(kx, ky, kz), tet_index(0, 0, p), i])
                                    .unwrap() -= T::from(kz * 2).unwrap()
                                    * (T::from(
                                        *points.get([1, i]).unwrap() + *points.get([2, i]).unwrap(),
                                    )
                                    .unwrap()
                                        - T::from(1.0).unwrap())
                                    * d
                                    * (a - T::from(1.0).unwrap());
                            }
                        }
                        if kz > 1 {
                            for i in 0..points.shape()[1] {
                                let d = *data
                                    .get([tet_index(kx, ky, kz - 2), tet_index(0, 0, p - 2), i])
                                    .unwrap();
                                *data
                                    .get_mut([tet_index(kx, ky, kz), tet_index(0, 0, p), i])
                                    .unwrap() -= T::from(kz * (kz - 1)).unwrap()
                                    * d
                                    * (a - T::from(1.0).unwrap());
                            }
                        }
                        if ky > 0 && kz > 0 {
                            for i in 0..points.shape()[1] {
                                let d = *data
                                    .get([tet_index(kx, ky - 1, kz - 1), tet_index(0, 0, p - 2), i])
                                    .unwrap();
                                *data
                                    .get_mut([tet_index(kx, ky, kz), tet_index(0, 0, p), i])
                                    .unwrap() -=
                                    T::from(2 * ky * kz).unwrap() * d * (a - T::from(1.0).unwrap());
                            }
                        }
                    }
                }
                for p in 0..degree {
                    for i in 0..points.shape()[1] {
                        let d = *data
                            .get([tet_index(kx, ky, kz), tet_index(0, 0, p), i])
                            .unwrap();
                        *data
                            .get_mut([tet_index(kx, ky, kz), tet_index(0, 1, p), i])
                            .unwrap() = d
                            * (T::from(*points.get([1, i]).unwrap()).unwrap()
                                * T::from(2 * p + 3).unwrap()
                                + T::from(*points.get([2, i]).unwrap()).unwrap()
                                - T::from(1).unwrap());
                    }
                    if ky > 0 {
                        for i in 0..points.shape()[1] {
                            let d = *data
                                .get([tet_index(kx, ky - 1, kz), tet_index(0, 0, p), i])
                                .unwrap();
                            *data
                                .get_mut([tet_index(kx, ky, kz), tet_index(0, 1, p), i])
                                .unwrap() += T::from(2 * ky).unwrap()
                                * d
                                * (T::from(p).unwrap() + T::from(1.5).unwrap());
                        }
                    }
                    if kz > 0 {
                        for i in 0..points.shape()[1] {
                            let d = *data
                                .get([tet_index(kx, ky, kz - 1), tet_index(0, 0, p), i])
                                .unwrap();
                            *data
                                .get_mut([tet_index(kx, ky, kz), tet_index(0, 1, p), i])
                                .unwrap() += T::from(kz).unwrap() * d;
                        }
                    }

                    for q in 1..degree - p {
                        let (aq, bq, cq) = jrc::<T>(2 * p + 1, q);

                        for i in 0..points.shape()[1] {
                            let d = *data
                                .get([tet_index(kx, ky, kz), tet_index(0, q, p), i])
                                .unwrap();
                            let d2 = *data
                                .get([tet_index(kx, ky, kz), tet_index(0, q - 1, p), i])
                                .unwrap();
                            *data
                                .get_mut([tet_index(kx, ky, kz), tet_index(0, q + 1, p), i])
                                .unwrap() = d
                                * (aq
                                    * (T::from(*points.get([1, i]).unwrap()).unwrap()
                                        * T::from(2.0).unwrap()
                                        - T::from(1.0).unwrap()
                                        + T::from(*points.get([2, i]).unwrap()).unwrap())
                                    + bq * (T::from(1.0).unwrap()
                                        - T::from(*points.get([2, i]).unwrap()).unwrap()))
                                - d2 * cq
                                    * (T::from(1.0).unwrap()
                                        - T::from(*points.get([2, i]).unwrap()).unwrap())
                                    .powi(2);
                        }

                        if ky > 0 {
                            for i in 0..points.shape()[1] {
                                let d = *data
                                    .get([tet_index(kx, ky - 1, kz), tet_index(0, q, p), i])
                                    .unwrap();
                                *data
                                    .get_mut([tet_index(kx, ky, kz), tet_index(0, q + 1, p), i])
                                    .unwrap() += T::from(2 * ky).unwrap() * d * aq;
                            }
                        }
                        if kz > 0 {
                            for i in 0..points.shape()[1] {
                                let d = *data
                                    .get([tet_index(kx, ky, kz - 1), tet_index(0, q, p), i])
                                    .unwrap();
                                let d2 = *data
                                    .get([tet_index(kx, ky, kz - 1), tet_index(0, q - 1, p), i])
                                    .unwrap();
                                *data
                                    .get_mut([tet_index(kx, ky, kz), tet_index(0, q + 1, p), i])
                                    .unwrap() += T::from(kz).unwrap() * d * (aq - bq)
                                    + T::from(2 * kz).unwrap()
                                        * (T::from(1.0).unwrap()
                                            - T::from(*points.get([2, i]).unwrap()).unwrap())
                                        * d2
                                        * cq;
                            }
                        }
                        if kz > 1 {
                            for i in 0..points.shape()[1] {
                                let d = *data
                                    .get([tet_index(kx, ky, kz - 2), tet_index(0, q - 1, p), i])
                                    .unwrap();
                                *data
                                    .get_mut([tet_index(kx, ky, kz), tet_index(0, q + 1, p), i])
                                    .unwrap() -= T::from(kz * (kz - 1)).unwrap() * d * cq;
                            }
                        }
                    }
                }

                for p in 0..degree {
                    for q in 0..degree - p {
                        for i in 0..points.shape()[1] {
                            let d = *data
                                .get([tet_index(kx, ky, kz), tet_index(0, q, p), i])
                                .unwrap();
                            *data
                                .get_mut([tet_index(kx, ky, kz), tet_index(1, q, p), i])
                                .unwrap() = d
                                * (T::from(*points.get([2, i]).unwrap()).unwrap()
                                    * T::from(2 + p + q).unwrap()
                                    * T::from(2.0).unwrap()
                                    - T::from(1.0).unwrap());
                        }
                        if kz > 0 {
                            for i in 0..points.shape()[1] {
                                let d = *data
                                    .get([tet_index(kx, ky, kz - 1), tet_index(0, q, p), i])
                                    .unwrap();
                                *data
                                    .get_mut([tet_index(kx, ky, kz), tet_index(1, q, p), i])
                                    .unwrap() += T::from(2 * kz * (2 + p + q)).unwrap() * d;
                            }
                        }
                    }
                }

                if degree > 0 {
                    for p in 0..degree - 1 {
                        for q in 0..degree - 1 - p {
                            for r in 1..degree - p - q {
                                let (ar, br, cr) = jrc::<T>(2 * p + 2 * q + 2, r);

                                for i in 0..points.shape()[1] {
                                    let d = *data
                                        .get([tet_index(kx, ky, kz), tet_index(r, q, p), i])
                                        .unwrap();
                                    let d2 = *data
                                        .get([tet_index(kx, ky, kz), tet_index(r - 1, q, p), i])
                                        .unwrap();
                                    *data
                                        .get_mut([tet_index(kx, ky, kz), tet_index(r + 1, q, p), i])
                                        .unwrap() = d
                                        * (ar
                                            * (T::from(2.0).unwrap()
                                                * T::from(*points.get([2, i]).unwrap()).unwrap()
                                                - T::from(1.0).unwrap())
                                            + br)
                                        - d2 * cr;
                                }
                                if kz > 0 {
                                    for i in 0..points.shape()[1] {
                                        let d = *data
                                            .get([tet_index(kx, ky, kz - 1), tet_index(r, q, p), i])
                                            .unwrap();
                                        *data
                                            .get_mut([
                                                tet_index(kx, ky, kz),
                                                tet_index(r + 1, q, p),
                                                i,
                                            ])
                                            .unwrap() += T::from(2 * kz).unwrap() * ar * d;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Normalise
    for p in 0..=degree {
        for q in 0..=degree - p {
            for r in 0..=degree - p - q {
                let norm = T::sqrt(
                    (T::from(p).unwrap() + T::from(0.5).unwrap())
                        * T::from(p + q + 1).unwrap()
                        * (T::from(p + q + r).unwrap() + T::from(1.5).unwrap()),
                ) * T::from(2).unwrap()
                    / T::sqrt(T::from(3).unwrap());
                for j in 0..data.shape()[2] {
                    for i in 0..data.shape()[0] {
                        *data.get_mut([i, tet_index(r, q, p), j]).unwrap() *= norm;
                    }
                }
            }
        }
    }
}

/// Tabulate orthonormal polynomials on a hexahedron
fn tabulate_hexahedron<
    T: RlstScalar,
    Array2: RandomAccessByRef<2, Item = T::Real> + Shape<2>,
    Array3Mut: RandomAccessMut<3, Item = T> + RandomAccessByRef<3, Item = T> + Shape<3>,
>(
    points: &Array2,
    degree: usize,
    derivatives: usize,
    data: &mut Array3Mut,
) {
    debug_assert!(data.shape()[0] == (derivatives + 1) * (derivatives + 2) * (derivatives + 3) / 6);
    debug_assert!(data.shape()[1] == (degree + 1) * (degree + 1) * (degree + 1));
    debug_assert!(data.shape()[2] == points.shape()[1]);
    debug_assert!(points.shape()[0] == 3);

    for i in 0..data.shape()[2] {
        *data
            .get_mut([tet_index(0, 0, 0), hex_index(0, 0, 0, degree), i])
            .unwrap() = T::from(1.0).unwrap();
    }

    // Tabulate polynomials in x
    for k in 1..=derivatives {
        for i in 0..data.shape()[2] {
            *data
                .get_mut([tet_index(k, 0, 0), hex_index(0, 0, 0, degree), i])
                .unwrap() = T::from(0.0).unwrap();
        }
    }

    for k in 0..=derivatives {
        for p in 1..=degree {
            let a = T::from(1.0).unwrap() - T::from(1.0).unwrap() / T::from(p).unwrap();
            let b = (a + T::from(1.0).unwrap())
                * ((T::from(2.0).unwrap() * T::from(p).unwrap() + T::from(1.0).unwrap())
                    / (T::from(2.0).unwrap() * T::from(p).unwrap() - T::from(1.0).unwrap()))
                .sqrt();
            for i in 0..data.shape()[2] {
                let d = *data
                    .get([tet_index(k, 0, 0), hex_index(p - 1, 0, 0, degree), i])
                    .unwrap();
                *data
                    .get_mut([tet_index(k, 0, 0), hex_index(p, 0, 0, degree), i])
                    .unwrap() = (T::from(*points.get([0, i]).unwrap()).unwrap()
                    * T::from(2.0).unwrap()
                    - T::from(1.0).unwrap())
                    * d
                    * b;
            }
            if p > 1 {
                let c = a
                    * ((T::from(2.0).unwrap() * T::from(p).unwrap() + T::from(1.0).unwrap())
                        / (T::from(2.0).unwrap() * T::from(p).unwrap() - T::from(3.0).unwrap()))
                    .sqrt();
                for i in 0..data.shape()[2] {
                    let d = *data
                        .get([tet_index(k, 0, 0), hex_index(p - 2, 0, 0, degree), i])
                        .unwrap();
                    *data
                        .get_mut([tet_index(k, 0, 0), hex_index(p, 0, 0, degree), i])
                        .unwrap() -= d * c;
                }
            }
            if k > 0 {
                for i in 0..data.shape()[2] {
                    let d = *data
                        .get([tet_index(k - 1, 0, 0), hex_index(p - 1, 0, 0, degree), i])
                        .unwrap();
                    *data
                        .get_mut([tet_index(k, 0, 0), hex_index(p, 0, 0, degree), i])
                        .unwrap() += T::from(2.0).unwrap() * T::from(k).unwrap() * d * b;
                }
            }
        }
    }

    // Tabulate polynomials in y
    for k in 1..=derivatives {
        for i in 0..data.shape()[2] {
            *data
                .get_mut([tet_index(0, k, 0), hex_index(0, 0, 0, degree), i])
                .unwrap() = T::from(0.0).unwrap();
        }
    }

    for k in 0..=derivatives {
        for p in 1..=degree {
            let a = T::from(1.0).unwrap() - T::from(1.0).unwrap() / T::from(p).unwrap();
            let b = (a + T::from(1.0).unwrap())
                * ((T::from(2.0).unwrap() * T::from(p).unwrap() + T::from(1.0).unwrap())
                    / (T::from(2.0).unwrap() * T::from(p).unwrap() - T::from(1.0).unwrap()))
                .sqrt();
            for i in 0..data.shape()[2] {
                let d = *data
                    .get([tet_index(0, k, 0), hex_index(0, p - 1, 0, degree), i])
                    .unwrap();
                *data
                    .get_mut([tet_index(0, k, 0), hex_index(0, p, 0, degree), i])
                    .unwrap() = (T::from(*points.get([1, i]).unwrap()).unwrap()
                    * T::from(2.0).unwrap()
                    - T::from(1.0).unwrap())
                    * d
                    * b;
            }
            if p > 1 {
                let c = a
                    * ((T::from(2.0).unwrap() * T::from(p).unwrap() + T::from(1.0).unwrap())
                        / (T::from(2.0).unwrap() * T::from(p).unwrap() - T::from(3.0).unwrap()))
                    .sqrt();
                for i in 0..data.shape()[2] {
                    let d = *data
                        .get([tet_index(0, k, 0), hex_index(0, p - 2, 0, degree), i])
                        .unwrap();
                    *data
                        .get_mut([tet_index(0, k, 0), hex_index(0, p, 0, degree), i])
                        .unwrap() -= d * c;
                }
            }
            if k > 0 {
                for i in 0..data.shape()[2] {
                    let d = *data
                        .get([tet_index(0, k - 1, 0), hex_index(0, p - 1, 0, degree), i])
                        .unwrap();
                    *data
                        .get_mut([tet_index(0, k, 0), hex_index(0, p, 0, degree), i])
                        .unwrap() += T::from(2.0).unwrap() * T::from(k).unwrap() * d * b;
                }
            }
        }
    }

    // Tabulate polynomials in z
    for k in 1..=derivatives {
        for i in 0..data.shape()[2] {
            *data
                .get_mut([tet_index(0, 0, k), hex_index(0, 0, 0, degree), i])
                .unwrap() = T::from(0.0).unwrap();
        }
    }

    for k in 0..=derivatives {
        for p in 1..=degree {
            let a = T::from(1.0).unwrap() - T::from(1.0).unwrap() / T::from(p).unwrap();
            let b = (a + T::from(1.0).unwrap())
                * ((T::from(2.0).unwrap() * T::from(p).unwrap() + T::from(1.0).unwrap())
                    / (T::from(2.0).unwrap() * T::from(p).unwrap() - T::from(1.0).unwrap()))
                .sqrt();
            for i in 0..data.shape()[2] {
                let d = *data
                    .get([tet_index(0, 0, k), hex_index(0, 0, p - 1, degree), i])
                    .unwrap();
                *data
                    .get_mut([tet_index(0, 0, k), hex_index(0, 0, p, degree), i])
                    .unwrap() = (T::from(*points.get([2, i]).unwrap()).unwrap()
                    * T::from(2.0).unwrap()
                    - T::from(1.0).unwrap())
                    * d
                    * b;
            }
            if p > 1 {
                let c = a
                    * ((T::from(2.0).unwrap() * T::from(p).unwrap() + T::from(1.0).unwrap())
                        / (T::from(2.0).unwrap() * T::from(p).unwrap() - T::from(3.0).unwrap()))
                    .sqrt();
                for i in 0..data.shape()[2] {
                    let d = *data
                        .get([tet_index(0, 0, k), hex_index(0, 0, p - 2, degree), i])
                        .unwrap();
                    *data
                        .get_mut([tet_index(0, 0, k), hex_index(0, 0, p, degree), i])
                        .unwrap() -= d * c;
                }
            }
            if k > 0 {
                for i in 0..data.shape()[2] {
                    let d = *data
                        .get([tet_index(0, 0, k - 1), hex_index(0, 0, p - 1, degree), i])
                        .unwrap();
                    *data
                        .get_mut([tet_index(0, 0, k), hex_index(0, 0, p, degree), i])
                        .unwrap() += T::from(2.0).unwrap() * T::from(k).unwrap() * d * b;
                }
            }
        }
    }

    // Fill in the rest of the values as products
    for kx in 0..=derivatives {
        for ky in 0..=derivatives - kx {
            for kz in 0..=derivatives - kx - ky {
                for px in 0..=degree {
                    for py in if px == 0 { 1 } else { 0 }..=degree {
                        for pz in if px * py == 0 { 1 } else { 0 }..=degree {
                            for i in 0..data.shape()[2] {
                                let dx = *data
                                    .get([tet_index(kx, 0, 0), hex_index(px, 0, 0, degree), i])
                                    .unwrap();
                                let dy = *data
                                    .get([tet_index(0, ky, 0), hex_index(0, py, 0, degree), i])
                                    .unwrap();
                                let dz = *data
                                    .get([tet_index(0, 0, kz), hex_index(0, 0, pz, degree), i])
                                    .unwrap();
                                *data
                                    .get_mut([
                                        tet_index(kx, ky, kz),
                                        hex_index(px, py, pz, degree),
                                        i,
                                    ])
                                    .unwrap() = dx * dy * dz;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// The shape of a table containing the values of Legendre polynomials
pub fn shape<T, Array2: RandomAccessByRef<2, Item = T> + Shape<2>>(
    cell_type: ReferenceCellType,
    points: &Array2,
    degree: usize,
    derivatives: usize,
) -> [usize; 3] {
    [
        derivative_count(cell_type, derivatives),
        polynomial_count(cell_type, degree),
        points.shape()[1],
    ]
}

/// Tabulate orthonormal polynomials
pub fn tabulate<
    T: RlstScalar,
    Array2: RandomAccessByRef<2, Item = T::Real> + Shape<2>,
    Array3Mut: RandomAccessMut<3, Item = T> + RandomAccessByRef<3, Item = T> + Shape<3>,
>(
    cell_type: ReferenceCellType,
    points: &Array2,
    degree: usize,
    derivatives: usize,
    data: &mut Array3Mut,
) {
    match cell_type {
        ReferenceCellType::Interval => tabulate_interval(points, degree, derivatives, data),
        ReferenceCellType::Triangle => tabulate_triangle(points, degree, derivatives, data),
        ReferenceCellType::Quadrilateral => {
            tabulate_quadrilateral(points, degree, derivatives, data)
        }
        ReferenceCellType::Tetrahedron => tabulate_tetrahedron(points, degree, derivatives, data),
        ReferenceCellType::Hexahedron => tabulate_hexahedron(points, degree, derivatives, data),
        _ => {
            panic!("Unsupported cell type: {cell_type:?}");
        }
    };
}
