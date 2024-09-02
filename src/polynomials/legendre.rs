//! Orthonormal polynomials
use super::derivative_count;
use crate::types::ReferenceCellType;
use rlst::RlstScalar;
use rlst::{RandomAccessByRef, RandomAccessMut, Shape};

/// Tabulate orthonormal polynomials on a interval
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
    assert_eq!(data.shape()[0], derivatives + 1);
    assert_eq!(data.shape()[1], degree + 1);
    assert_eq!(data.shape()[2], points.shape()[1]);
    assert_eq!(points.shape()[0], 1);

    for i in 0..data.shape()[2] {
        *data.get_mut([0, 0, i]).unwrap() = T::from(1.0).unwrap();
    }
    for k in 1..data.shape()[0] {
        for i in 0..data.shape()[2] {
            *data.get_mut([k, 0, i]).unwrap() = T::from(0.0).unwrap();
        }
    }

    for k in 0..derivatives + 1 {
        for p in 1..degree + 1 {
            let a = T::from(1.0).unwrap() - T::from(1.0).unwrap() / T::from(p).unwrap();
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

fn tri_index(i: usize, j: usize) -> usize {
    (i + j + 1) * (i + j) / 2 + j
}

fn quad_index(i: usize, j: usize, n: usize) -> usize {
    j * (n + 1) + i
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
    assert_eq!(data.shape()[0], (derivatives + 1) * (derivatives + 2) / 2);
    assert_eq!(data.shape()[1], (degree + 1) * (degree + 1));
    assert_eq!(data.shape()[2], points.shape()[1]);
    assert_eq!(points.shape()[0], 2);

    for i in 0..data.shape()[2] {
        *data
            .get_mut([tri_index(0, 0), quad_index(0, 0, degree), i])
            .unwrap() = T::from(1.0).unwrap();
    }

    // Tabulate polynomials in x
    for k in 1..derivatives + 1 {
        for i in 0..data.shape()[2] {
            *data
                .get_mut([tri_index(k, 0), quad_index(0, 0, degree), i])
                .unwrap() = T::from(0.0).unwrap();
        }
    }

    for k in 0..derivatives + 1 {
        for p in 1..degree + 1 {
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
    for k in 1..derivatives + 1 {
        for i in 0..data.shape()[2] {
            *data
                .get_mut([tri_index(0, k), quad_index(0, 0, degree), i])
                .unwrap() = T::from(0.0).unwrap();
        }
    }

    for k in 0..derivatives + 1 {
        for p in 1..degree + 1 {
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
    for kx in 0..derivatives + 1 {
        for ky in 0..derivatives + 1 - kx {
            for px in 1..degree + 1 {
                for py in 1..degree + 1 {
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
    assert_eq!(data.shape()[0], (derivatives + 1) * (derivatives + 2) / 2);
    assert_eq!(data.shape()[1], (degree + 1) * (degree + 2) / 2);
    assert_eq!(data.shape()[2], points.shape()[1]);
    assert_eq!(points.shape()[0], 2);

    for i in 0..data.shape()[2] {
        *data.get_mut([tri_index(0, 0), tri_index(0, 0), i]).unwrap() =
            T::sqrt(T::from(2.0).unwrap());
    }

    for k in 1..data.shape()[0] {
        for i in 0..data.shape()[2] {
            *data.get_mut([k, tri_index(0, 0), i]).unwrap() = T::from(0.0).unwrap();
        }
    }

    for kx in 0..derivatives + 1 {
        for ky in 0..derivatives + 1 - kx {
            for p in 1..degree + 1 {
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
                    let a1 = T::from((p + q + 1) * (2 * p + 2 * q + 3)).unwrap()
                        / T::from((q + 1) * (2 * p + q + 2)).unwrap();
                    let a2 = T::from((2 * p + 1) * (2 * p + 1) * (p + q + 1)).unwrap()
                        / T::from((q + 1) * (2 * p + q + 2) * (2 * p + 2 * q + 1)).unwrap();
                    let a3 = T::from(q * (2 * p + q + 1) * (2 * p + 2 * q + 3)).unwrap()
                        / T::from((q + 1) * (2 * p + q + 2) * (2 * p + 2 * q + 1)).unwrap();

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

/// The number of polynomials
pub fn polynomial_count(cell_type: ReferenceCellType, degree: usize) -> usize {
    match cell_type {
        ReferenceCellType::Interval => degree + 1,
        ReferenceCellType::Triangle => (degree + 1) * (degree + 2) / 2,
        ReferenceCellType::Quadrilateral => (degree + 1) * (degree + 1),
        _ => {
            panic!("Unsupported cell type: {cell_type:?}");
        }
    }
}

/*
template <typename T>
void tabulate_polyset_tetrahedron_derivs(
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 3>>
        P,
    std::size_t n, std::size_t nderiv,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        x)
{
  assert(x.extent(1) == 3);
  assert(P.extent(0) == (nderiv + 1) * (nderiv + 2) * (nderiv + 3) / 6);
  assert(P.extent(1) == (n + 1) * (n + 2) * (n + 3) / 6);
  assert(P.extent(2) == x.extent(0));

  auto x0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      x, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);
  auto x1 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      x, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 1);
  auto x2 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      x, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 2);

  // Traverse derivatives in increasing order
  std::fill(P.data_handle(), P.data_handle() + P.size(), 0.0);
  for (std::size_t i = 0; i < P.extent(2); ++i)
    P(idx(0, 0, 0), 0, i) = 1.0;

  if (n == 0)
  {
    for (std::size_t i = 0; i < P.extent(2); ++i)
      P(idx(0, 0, 0), 0, i) = std::sqrt(6);
    return;
  }

  for (std::size_t kx = 0; kx <= nderiv; ++kx)
  {
    for (std::size_t ky = 0; ky <= nderiv - kx; ++ky)
    {
      for (std::size_t kz = 0; kz <= nderiv - kx - ky; ++kz)
      {
        for (std::size_t p = 1; p <= n; ++p)
        {
          auto p00 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
              P, idx(kx, ky, kz), idx(0, 0, p),
              MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
          auto p0m1 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
              P, idx(kx, ky, kz), idx(0, 0, p - 1),
              MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
          T a = static_cast<T>(2 * p - 1) / static_cast<T>(p);
          for (std::size_t i = 0; i < p00.size(); ++i)
          {
            p00[i] = ((x0[i] * 2.0 - 1.0)
                      + 0.5 * ((x1[i] * 2.0 - 1.0) + (x2[i] * 2.0 - 1.0)) + 1.0)
                     * a * p0m1[i];
          }

          if (kx > 0)
          {
            auto p0m1x = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                P, idx(kx - 1, ky, kz), idx(0, 0, p - 1),
                MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
            for (std::size_t i = 0; i < p00.size(); ++i)
              p00[i] += 2 * kx * a * p0m1x[i];
          }

          if (ky > 0)
          {
            auto p0m1y = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                P, idx(kx, ky - 1, kz), idx(0, 0, p - 1),
                MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
            for (std::size_t i = 0; i < p00.size(); ++i)
              p00[i] += ky * a * p0m1y[i];
          }

          if (kz > 0)
          {
            auto p0m1z = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                P, idx(kx, ky, kz - 1), idx(0, 0, p - 1),
                MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
            for (std::size_t i = 0; i < p00.size(); ++i)
              p00[i] += kz * a * p0m1z[i];
          }

          if (p > 1)
          {
            auto p0m2 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                P, idx(kx, ky, kz), idx(0, 0, p - 2),
                MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
            for (std::size_t i = 0; i < p00.size(); ++i)
            {
              T f2 = x1[i] + x2[i] - 1.0;
              p00[i] -= f2 * f2 * p0m2[i] * (a - 1.0);
            }
            if (ky > 0)
            {
              auto p0m2y = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                  P, idx(kx, ky - 1, kz), idx(0, 0, p - 2),
                  MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
              for (std::size_t i = 0; i < p00.size(); ++i)
              {
                p00[i] -= ky * ((x1[i] * 2.0 - 1.0) + (x2[i] * 2.0 - 1.0))
                          * p0m2y[i] * (a - 1.0);
              }
            }

            if (ky > 1)
            {
              auto p0m2y2 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                  P, idx(kx, ky - 2, kz), idx(0, 0, p - 2),
                  MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
              for (std::size_t i = 0; i < p00.size(); ++i)
                p00[i] -= ky * (ky - 1) * p0m2y2[i] * (a - 1.0);
            }

            if (kz > 0)
            {
              auto p0m2z = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                  P, idx(kx, ky, kz - 1), idx(0, 0, p - 2),
                  MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
              for (std::size_t i = 0; i < p00.size(); ++i)
                p00[i] -= kz * ((x1[i] * 2.0 - 1.0) + (x2[i] * 2.0 - 1.0))
                          * p0m2z[i] * (a - 1.0);
            }

            if (kz > 1)
            {
              auto p0m2z2 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                  P, idx(kx, ky, kz - 2), idx(0, 0, p - 2),
                  MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
              for (std::size_t i = 0; i < p00.size(); ++i)
                p00[i] -= kz * (kz - 1) * p0m2z2[i] * (a - 1.0);
            }

            if (ky > 0 and kz > 0)
            {
              auto p0m2yz = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                  P, idx(kx, ky - 1, kz - 1), idx(0, 0, p - 2),
                  MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
              for (std::size_t i = 0; i < p00.size(); ++i)
                p00[i] -= 2.0 * ky * kz * p0m2yz[i] * (a - 1.0);
            }
          }
        }

        for (std::size_t p = 0; p < n; ++p)
        {
          auto p10 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
              P, idx(kx, ky, kz), idx(0, 1, p),
              MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
          auto p00 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
              P, idx(kx, ky, kz), idx(0, 0, p),
              MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
          for (std::size_t i = 0; i < p10.size(); ++i)
            p10[i]
                = p00[i]
                  * ((1.0 + (x1[i] * 2.0 - 1.0)) * p
                     + (2.0 + (x1[i] * 2.0 - 1.0) * 3.0 + (x2[i] * 2.0 - 1.0))
                           * 0.5);
          if (ky > 0)
          {
            auto p0y = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                P, idx(kx, ky - 1, kz), idx(0, 0, p),
                MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
            for (std::size_t i = 0; i < p10.size(); ++i)
              p10[i] += 2 * ky * p0y[i] * (1.5 + p);
          }

          if (kz > 0)
          {
            auto p0z = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                P, idx(kx, ky, kz - 1), idx(0, 0, p),
                MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
            for (std::size_t i = 0; i < p10.size(); ++i)
              p10[i] += kz * p0z[i];
          }

          for (std::size_t q = 1; q < n - p; ++q)
          {
            auto [aq, bq, cq] = jrc<T>(2 * p + 1, q);
            auto pq1 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                P, idx(kx, ky, kz), idx(0, q + 1, p),
                MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
            auto pq = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                P, idx(kx, ky, kz), idx(0, q, p),
                MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
            auto pqm1 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                P, idx(kx, ky, kz), idx(0, q - 1, p),
                MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
            for (std::size_t i = 0; i < pq1.size(); ++i)
            {
              T f4 = 1.0 - x2[i];
              T f3 = (x1[i] * 2.0 - 1.0 + x2[i]);
              pq1[i] = pq[i] * (f3 * aq + f4 * bq) - pqm1[i] * f4 * f4 * cq;
            }
            if (ky > 0)
            {
              auto pqy = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                  P, idx(kx, ky - 1, kz), idx(0, q, p),
                  MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
              for (std::size_t i = 0; i < pq1.size(); ++i)
                pq1[i] += 2 * ky * pqy[i] * aq;
            }

            if (kz > 0)
            {
              auto pqz = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                  P, idx(kx, ky, kz - 1), idx(0, q, p),
                  MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
              auto pq1z = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                  P, idx(kx, ky, kz - 1), idx(0, q - 1, p),
                  MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
              for (std::size_t i = 0; i < pq1.size(); ++i)
              {
                pq1[i] += kz * pqz[i] * (aq - bq)
                          + kz * (1.0 - (x2[i] * 2.0 - 1.0)) * pq1z[i] * cq;
              }
            }

            if (kz > 1)
            {
              auto pq1z2 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                  P, idx(kx, ky, kz - 2), idx(0, q - 1, p),
                  MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
              // Quadratic term in z
              for (std::size_t i = 0; i < pq1.size(); ++i)
                pq1[i] -= kz * (kz - 1) * pq1z2[i] * cq;
            }
          }
        }

        for (std::size_t p = 0; p < n; ++p)
        {
          for (std::size_t q = 0; q < n - p; ++q)
          {
            auto pq = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                P, idx(kx, ky, kz), idx(1, q, p),
                MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
            auto pq0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                P, idx(kx, ky, kz), idx(0, q, p),
                MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
            for (std::size_t i = 0; i < pq.size(); ++i)
            {
              pq[i] = pq0[i]
                      * ((1.0 + p + q) + (x2[i] * 2.0 - 1.0) * (2.0 + p + q));
            }

            if (kz > 0)
            {
              auto pqz = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                  P, idx(kx, ky, kz - 1), idx(0, q, p),
                  MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
              for (std::size_t i = 0; i < pq.size(); ++i)
                pq[i] += 2 * kz * (2.0 + p + q) * pqz[i];
            }
          }
        }

        for (std::size_t p = 0; p + 1 < n; ++p)
        {
          for (std::size_t q = 0; q + 1 < n - p; ++q)
          {
            for (std::size_t r = 1; r < n - p - q; ++r)
            {
              auto [ar, br, cr] = jrc<T>(2 * p + 2 * q + 2, r);
              auto pqr1 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                  P, idx(kx, ky, kz), idx(r + 1, q, p),
                  MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
              auto pqr = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                  P, idx(kx, ky, kz), idx(r, q, p),
                  MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
              auto pqrm1 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                  P, idx(kx, ky, kz), idx(r - 1, q, p),
                  MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);

              for (std::size_t i = 0; i < pqr1.size(); ++i)
              {
                pqr1[i]
                    = pqr[i] * ((x2[i] * 2.0 - 1.0) * ar + br) - pqrm1[i] * cr;
              }

              if (kz > 0)
              {
                auto pqrz = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                    P, idx(kx, ky, kz - 1), idx(r, q, p),
                    MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
                for (std::size_t i = 0; i < pqr1.size(); ++i)
                  pqr1[i] += 2 * kz * ar * pqrz[i];
              }
            }
          }
        }
      }
    }
  }

  // Normalise
  for (std::size_t p = 0; p <= n; ++p)
  {
    for (std::size_t q = 0; q <= n - p; ++q)
    {
      for (std::size_t r = 0; r <= n - p - q; ++r)
      {
        auto pqr = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
            P, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, idx(r, q, p),
            MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
        for (std::size_t i = 0; i < pqr.extent(0); ++i)
          for (std::size_t j = 0; j < pqr.extent(1); ++j)
            pqr(i, j)
                *= std::sqrt(2 * (p + 0.5) * (p + q + 1.0) * (p + q + r + 1.5))
                   * 2;
      }
    }
  }
}
*/

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
        _ => {
            panic!("Unsupported cell type: {cell_type:?}");
        }
    };
}
