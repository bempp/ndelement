//! Gauss-Lobatto quadrature
//!
//! Adapted from the C++ code by Chris Richardson
//! https://github.com/FEniCS/basix/blob/main/cpp/basix/quadrature.cpp
use num::traits::FloatConst;
use std::cmp::PartialOrd;
use crate::types::Array2D;
use libm::tgamma;
use rlst::{RlstScalar, rlst_dynamic_array2, rlst_dynamic_array3, DefaultIteratorMut};
use itertools::izip;

/// Evaluate the nth Jacobi polynomial and derivatives with weight
/// parameters (a, 0) at points x
fn compute_jacobi_deriv<T: RlstScalar<Real=T>>(a: T, n: usize, nderiv: usize, x: &[T]) -> Array2D<T> {
    let mut j_all = rlst_dynamic_array3!(T, [nderiv + 1, n + 1, x.len()]);

    let one = T::from(1.0).unwrap();
    let two = T::from(2.0).unwrap();

    for i in 0..=nderiv {
        if i == 0 {
            for j in 0..x.len() {
                j_all[[i, 0, j]] = one;
            }
        }

        if n > 0 {
            if i == 0 {
                for j in 0..x.len() {
                    j_all[[i, 1, j]] = (x[j] * (a + two) + a) / two;
                }
            } else if i == 1 {
                for j in 0..x.len() {
                    j_all[[i, 1, j]] = a / two + one;
                }
            }
        }
/*
        for (std::size_t j = 2; j < n + 1; ++j)
        {
          const T a1 = 2 * j * (j + a) * (2 * j + a - 2);
          const T a2 = (2 * j + a - 1) * (a * a) / a1;
          const T a3 = (2 * j + a - 1) * (2 * j + a) / (2 * j * (j + a));
          const T a4 = 2 * (j + a - 1) * (j - 1) * (2 * j + a) / a1;
          for (std::size_t k = 0; k < Jd.extent(1); ++k)
            j_all[[i, j, k]] = j_all[[i, j - 1, k]] * (x[k] * a3 + a2) - j_all[[i, j - 2, k]] * a4;
          if (i > 0)
          {
            for (std::size_t k = 0; k < Jd.extent(1); ++k)
              j_all[[i, j, k]] += i * a3 * j_all[[i - 1, j - 1, k]];
          }
        }
*/
    }

    let mut j = rlst_dynamic_array2!(T, [nderiv + 1, x.len()]);
    for (i1, mut col) in j.col_iter_mut().enumerate() {
        for (i0, entry) in col.iter_mut().enumerate() {
            *entry = j_all[[i0, n, i1]];
        }
    }

    j
}

/// Computes the m roots of \f$P_{m}^{a,0}\f$ on [-1,1] by Newton's
/// method. The initial guesses are the Chebyshev points.  Algorithm
/// implemented from the pseudocode given by Karniadakis and Sherwin.
fn compute_gauss_jacobi_points<T: RlstScalar<Real = T> + FloatConst + PartialOrd>(a: T, m: usize) -> Vec<T> {
    let eps = T::from(1.0e-8).unwrap();
    let max_iter = 100;
    let two = T::from(2.0).unwrap();
    let one = T::from(1.0).unwrap();

    let mut x = vec![T::zero(); m];


    for k in 0..m {
        // Initial guess
        x[k] = -T::cos((two * T::from(k).unwrap() + one) * T::PI() / (two * T::from(m).unwrap()));
        if k > 0 {
            x[k] = (x[k] + x[k - 1]) / two;
        }

        let mut j = 0;
        let mut delta = T::from(10.0).unwrap() * eps;
        while j < max_iter && delta.abs() > eps {
            let s = x.iter().take(k).map(|i| one / (x[k] - *i)).sum::<T>();
            println!("{s}");
//            std::span<const T> _x(&x[k], 1);
            let f = compute_jacobi_deriv(a, m, 1, &x);
            use rlst::RawAccess;
            println!("{:?}", f.data());
            let delta = f[[0, 0]] / (f[[1, 0]] - f[[0, 0]] * s);
            j += 1;
        }
    }
    x
}

/// Note: computes on [-1, 1]
fn compute_gauss_jacobi_rule<T: RlstScalar<Real=T> + FloatConst + PartialOrd>(a: T, m: usize) -> (Vec<T>, Vec<T>) {
    let pts = compute_gauss_jacobi_points(a, m);

//  mdarray_t<T, 2> Jd = compute_jacobi_deriv<T>(a, m, 1, pts);
//  T a1 = std::pow(2.0, a + 1.0);
    let mut wts = vec![T::zero(); m];
//  for (int i = 0; i < m; ++i)
//  {
//    T x = pts[i];
//    T f = Jd(1, i);
//    wts[i] = a1 / (1.0 - x * x) / (f * f);
//  }
    (pts, wts)
}

fn make_quadrature_line<T: RlstScalar<Real=T> + FloatConst + PartialOrd>(m: usize) -> (Vec<T>, Vec<T>) {
    let (mut pts, mut wts) = compute_gauss_jacobi_rule(T::zero(), m);

    let half = T::from(0.5).unwrap();
    let one = T::from(1.0).unwrap();
    for p in pts.iter_mut() {
        *p += one;
        *p *= half;
    }
    for w in wts.iter_mut() {
        *w *= half;
    }
    (pts, wts)
}

/*

//----------------------------------------------------------------------------

//-----------------------------------------------------------------------------

//----------------------------------------------------------------------------

//-----------------------------------------------------------------------------

/// @note Computes on [-1, 1]
template <std::floating_point T>
std::array<std::vector<T>, 2> compute_gauss_jacobi_rule(T a, int m)
{
  std::vector<T> pts = compute_gauss_jacobi_points<T>(a, m);
  mdarray_t<T, 2> Jd = compute_jacobi_deriv<T>(a, m, 1, pts);
  T a1 = std::pow(2.0, a + 1.0);
  std::vector<T> wts(m);
  for (int i = 0; i < m; ++i)
  {
    T x = pts[i];
    T f = Jd(1, i);
    wts[i] = a1 / (1.0 - x * x) / (f * f);
  }

  return {std::move(pts), std::move(wts)};
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::array<std::vector<T>, 2> make_quadrature_triangle_collapsed(std::size_t m)
{
  auto [ptx, wx] = compute_gauss_jacobi_rule<T>(0.0, m);
  auto [pty, wy] = compute_gauss_jacobi_rule<T>(1.0, m);

  std::vector<T> pts(m * m * 2);
  mdspan_t<T, 2> x(pts.data(), m * m, 2);
  std::vector<T> wts(m * m);
  int c = 0;
  for (std::size_t i = 0; i < m; ++i)
  {
    for (std::size_t j = 0; j < m; ++j)
    {
      x(c, 0) = 0.25 * (1.0 + ptx[i]) * (1.0 - pty[j]);
      x(c, 1) = 0.5 * (1.0 + pty[j]);
      wts[c] = wx[i] * wy[j] * 0.125;
      ++c;
    }
  }

  return {std::move(pts), std::move(wts)};
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::array<std::vector<T>, 2>
make_quadrature_tetrahedron_collapsed(std::size_t m)
{
  auto [ptx, wx] = compute_gauss_jacobi_rule<T>(0.0, m);
  auto [pty, wy] = compute_gauss_jacobi_rule<T>(1.0, m);
  auto [ptz, wz] = compute_gauss_jacobi_rule<T>(2.0, m);

  std::vector<T> pts(m * m * m * 3);
  mdspan_t<T, 2> x(pts.data(), m * m * m, 3);
  std::vector<T> wts(m * m * m);
  int c = 0;
  for (std::size_t i = 0; i < m; ++i)
  {
    for (std::size_t j = 0; j < m; ++j)
    {
      for (std::size_t k = 0; k < m; ++k)
      {
        x(c, 0) = 0.125 * (1.0 + ptx[i]) * (1.0 - pty[j]) * (1.0 - ptz[k]);
        x(c, 1) = 0.25 * (1. + pty[j]) * (1. - ptz[k]);
        x(c, 2) = 0.5 * (1.0 + ptz[k]);
        wts[c] = wx[i] * wy[j] * wz[k] * 0.125 * 0.125;
        ++c;
      }
    }
  }

  return {std::move(pts), std::move(wts)};
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::array<std::vector<T>, 2> make_gauss_jacobi_quadrature(cell::type celltype,
                                                           std::size_t m)
{
  const std::size_t np = (m + 2) / 2;
  switch (celltype)
  {
  case cell::type::interval:
    return make_quadrature_line<T>(np);
  case cell::type::quadrilateral:
  {
    auto [QptsL, QwtsL] = make_quadrature_line<T>(np);
    std::vector<T> pts(np * np * 2);
    mdspan_t<T, 2> x(pts.data(), np * np, 2);
    std::vector<T> wts(np * np);
    int c = 0;
    for (std::size_t i = 0; i < np; ++i)
    {
      for (std::size_t j = 0; j < np; ++j)
      {
        x(c, 0) = QptsL[i];
        x(c, 1) = QptsL[j];
        wts[c] = QwtsL[i] * QwtsL[j];
        ++c;
      }
    }
    return {std::move(pts), std::move(wts)};
  }
  case cell::type::hexahedron:
  {
    auto [QptsL, QwtsL] = make_quadrature_line<T>(np);
    std::vector<T> pts(np * np * np * 3);
    mdspan_t<T, 2> x(pts.data(), np * np * np, 3);
    std::vector<T> wts(np * np * np);
    int c = 0;
    for (std::size_t i = 0; i < np; ++i)
    {
      for (std::size_t j = 0; j < np; ++j)
      {
        for (std::size_t k = 0; k < np; ++k)
        {
          x(c, 0) = QptsL[i];
          x(c, 1) = QptsL[j];
          x(c, 2) = QptsL[k];
          wts[c] = QwtsL[i] * QwtsL[j] * QwtsL[k];
          ++c;
        }
      }
    }
    return {std::move(pts), std::move(wts)};
  }
  case cell::type::prism:
  {
    const auto [QptsL, QwtsL] = make_quadrature_line<T>(np);
    const auto [_QptsT, QwtsT] = make_quadrature_triangle_collapsed<T>(np);
    mdspan_t<const T, 2> QptsT(_QptsT.data(), QwtsT.size(),
                               _QptsT.size() / QwtsT.size());
    std::vector<T> pts(np * QptsT.extent(0) * 3);
    mdspan_t<T, 2> x(pts.data(), np * QptsT.extent(0), 3);
    std::vector<T> wts(np * QptsT.extent(0));
    int c = 0;
    for (std::size_t i = 0; i < QptsT.extent(0); ++i)
    {
      for (std::size_t k = 0; k < np; ++k)
      {
        x(c, 0) = QptsT(i, 0);
        x(c, 1) = QptsT(i, 1);
        x(c, 2) = QptsL[k];
        wts[c] = QwtsT[i] * QwtsL[k];
        ++c;
      }
    }
    return {std::move(pts), std::move(wts)};
  }
  case cell::type::pyramid:
  {
    auto [pts, wts] = make_gauss_jacobi_quadrature<T>(cell::type::hexahedron, m + 2);
    mdspan_t<T, 2> x(pts.data(), pts.size() / 3, 3);
    for (std::size_t i = 0; i < x.extent(0); ++i)
    {
      const auto z = x(i, 2);
      x(i, 0) *= (1 - z);
      x(i, 1) *= (1 - z);
      wts[i] *= (1 - z) * (1 - z);
    }
    return {std::move(pts), std::move(wts)};
  }
  case cell::type::triangle:
    return make_quadrature_triangle_collapsed<T>(np);
  case cell::type::tetrahedron:
    return make_quadrature_tetrahedron_collapsed<T>(np);
  default:
    throw std::runtime_error("Unsupported celltype for make_quadrature");
  }
}

//-----------------------------------------------------------------------------

template <std::floating_point T>
std::array<std::vector<T>, 2> make_gll_quadrature(cell::type celltype,
                                                  std::size_t m)
{
  const std::size_t np = (m + 4) / 2;
  switch (celltype)
  {
  case cell::type::interval:
    return make_gll_line<T>(np);
  case cell::type::quadrilateral:
  {
    auto [QptsL, QwtsL] = make_gll_line<T>(np);
    std::vector<T> pts(np * np * 2);
    mdspan_t<T, 2> x(pts.data(), np * np, 2);
    std::vector<T> wts(np * np);
    int c = 0;
    for (std::size_t i = 0; i < np; ++i)
    {
      for (std::size_t j = 0; j < np; ++j)
      {
        x(c, 0) = QptsL[i];
        x(c, 1) = QptsL[j];
        wts[c] = QwtsL[i] * QwtsL[j];
        ++c;
      }
    }
    return {std::move(pts), std::move(wts)};
  }
  case cell::type::hexahedron:
  {
    auto [QptsL, QwtsL] = make_gll_line<T>(np);
    std::vector<T> pts(np * np * np * 3);
    mdspan_t<T, 2> x(pts.data(), np * np * np, 3);
    std::vector<T> wts(np * np * np);
    int c = 0;
    for (std::size_t i = 0; i < np; ++i)
    {
      for (std::size_t j = 0; j < np; ++j)
      {
        for (std::size_t k = 0; k < np; ++k)
        {
          x(c, 0) = QptsL[i];
          x(c, 1) = QptsL[j];
          x(c, 2) = QptsL[k];
          wts[c] = QwtsL[i] * QwtsL[j] * QwtsL[k];
          ++c;
        }
      }
    }
    return {std::move(pts), std::move(wts)};
  }
  case cell::type::prism:
    throw std::runtime_error("Prism not yet supported");
  case cell::type::pyramid:
    throw std::runtime_error("Pyramid not yet supported");
  case cell::type::triangle:
    throw std::runtime_error("Triangle not yet supported");
  case cell::type::tetrahedron:
    throw std::runtime_error("Tetrahedron not yet supported");
  default:
    throw std::runtime_error("Unsupported celltype for make_quadrature");
  }
}
//-----------------------------------------------------------------------------
std::vector<T> quadrature::get_gll_points(int m)
{
  return make_gll_line<T>(m)[0];
}
//-----------------------------------------------------------------------------
*/


#[cfg(test)]
mod test {
    use super::*;
    use approx::*;

    #[test]
    fn test_gauss() {
        let (pts, wts) = make_quadrature_line::<f64>(3);

        println!("pts = {pts:?}");
        println!("wts = {wts:?}");

        for (p, q) in izip!(&pts, [0.1127016653792583, 0.5, 0.8872983346207417]) {
            assert_relative_eq!(*p, q);
        }
        for (w, v) in izip!(&wts, [0.2777777777777777, 0.4444444444444444, 0.2777777777777777]) {
            assert_relative_eq!(*w, v);
        }
        assert_eq!(1, 0);
    }
}
