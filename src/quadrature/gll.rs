//! Gauss-Lobatto-Legendre quadrature
use libm::tgamma;
use rlst::{RlstScalar, rlst_dynamic_array2};
use itertools::izip;

fn gamma<T: RlstScalar>(x: T) -> T {
    num::cast::<f64, T>(tgamma(num::cast::<T, f64>(x).unwrap())).unwrap()
}

/// Generate the recursion coefficients alpha_k, beta_k
///
/// P_{k+1}(x) = (x-alpha_k)*P_{k}(x) - beta_k P_{k-1}(x)
///
/// for the Jacobi polynomials which are orthogonal on [-1,1]
/// with respect to the weight w(x)=[(1-x)^a]*[(1+x)^b]
///
/// Adapted from the C++ code by Chris Richardson
/// https://github.com/FEniCS/basix/blob/main/cpp/basix/quadrature.cpp
/// which was adapted from the MATLAB code by Dirk Laurie and Walter Gautschi
/// http://www.cs.purdue.edu/archives/2002/wxg/codes/r_jacobi.m
fn rec_jacobi<T: RlstScalar>(degree: usize, a: T, b: T) -> (Vec<T>, Vec<T>) {
    let one = T::from(1.0).unwrap();
    let two = T::from(2.0).unwrap();

    let nu = (b - a) / (a + b + two);
    let mu = two.pow(a + b + one) * gamma(a + one)
               * gamma(b + one) / gamma(a + b + two);

    let n = (1..degree).map(|i| num::cast::<usize, T>(i).unwrap()).collect::<Vec<_>>();
    let nab = n.iter().map(|x| two * *x + a + b).collect::<Vec<_>>();

    let mut alpha = vec![T::zero(); degree];
    alpha[0] = nu;
    for (i, j) in izip!(alpha.iter_mut().skip(1), &nab) {
        *i = (b.powi(2) - a.powi(2)) / (*j * (*j + two));
    }

    let mut beta = vec![T::zero(); degree];
    beta[0] = mu;

    for (i, j, k) in izip!(beta.iter_mut().skip(1), n, nab) {
        *i = T::from(4.0).unwrap() * (j + a) * (j + b) * j * (j + a + b) / (k.powi(2) * (k + one) * (k - one));
    }

    (alpha, beta)
}

/// Compute the eigenvalues and first components of the eigenvectors of a symmetric
/// tridiagonal matrix with diag on its diagonal and off_diag on the off-diagonals
fn eig<T: RlstScalar>(diag: &[T], off_diag: &[T]) -> (Vec<T>, Vec<T>) {
    debug_assert!(diag.len() == off_diag.len() + 1);
    println!("diag = {diag:?}");
    println!("off_diag = {off_diag:?}");
    let mut eigs = diag.to_vec();
    for (i, o) in off_diag.iter().enumerate() {
        let s = *o / diag[i];
        
        println!("DO {o} {s}");
    }
    let mut components = vec![T::zero(); diag.len()];
    println!("{eigs:?}");
    println!("{components:?}");
    (eigs, components)
}

/// Compute Gauss points and weights on the domain [-1, 1] using
/// the Golub-Welsch algorithm
///
/// https://en.wikipedia.org/wiki/Gaussian_quadrature#The_Golub-Welsch_algorithm
///
/// Adapted from the C++ code by Chris Richardson
/// https://github.com/FEniCS/basix/blob/main/cpp/basix/quadrature.cpp
fn gauss<T: RlstScalar>(alpha: &[T], beta: &[T]) -> (Vec<T>, Vec<T>) {
    let adim = alpha.len();
    let beta_sqrt = beta.iter().skip(1).map(|i| i.sqrt()).collect::<Vec<_>>();
    eig(&alpha, &beta_sqrt)
}

/// Compute the Lobatto nodes and weights with the preassigned
/// nodes xl1, xl2
///
/// Based on the section 7 of the paper "Some modified matrix eigenvalue
/// problems", https://doi.org/10.1137/1015032.
///
/// Adapted from the C++ code by Chris Richardson
/// https://github.com/FEniCS/basix/blob/main/cpp/basix/quadrature.cpp
fn lobatto<T: RlstScalar>(alpha: &[T], beta: &[T], xl1: T, xl2: T) -> (Vec<T>, Vec<T>) {
    println!("a = {alpha:?}");
    println!("b = {beta:?}");
    debug_assert!(alpha.len() == beta.len());

    // Solve tridiagonal system using Thomas algorithm
    let mut g1 = T::zero();
    let mut g2 = T::zero();
    let n = alpha.len();

    for (a, b, b1) in izip!(
        alpha.iter().take(n-1).skip(1), beta.iter().take(n-1).skip(1), beta.iter().take(n-2)
    ) {
        g1 = b.sqrt() / (*a - xl1 - b1.sqrt() * g1);
        g2 = b.sqrt() / (*a - xl2 - b1.sqrt() * g2);
    }
    let one = T::from(1.0).unwrap();
    g1 = one / (alpha[n - 1] - xl1 - (beta[n - 2]) * g1);
    g2 = one / (alpha[n - 1] - xl2 - (beta[n - 2]) * g2);

    let mut alpha_l = alpha.iter().map(|i| *i).collect::<Vec<_>>();
    let mut beta_l = beta.iter().map(|i| *i).collect::<Vec<_>>();

    alpha_l[n - 1] = (g1 * xl2 - g2 * xl1) / (g1 - g2);
    beta_l[n - 1] = (xl2 - xl1) / (g1 - g2);

    println!("a={alpha:?}");
    println!("b={beta:?}");

    gauss(&alpha_l, &beta_l)
}

/// The Gauss-Lobatto-Legendre quadrature rules on the interval using
/// Greg von Winckel's implementation. This facilitates implementing
/// spectral elements. The quadrature rule uses m points for a degree of
/// precision of 2m-3.
///
/// Adapted from the C++ code by Chris Richardson
/// https://github.com/FEniCS/basix/blob/main/cpp/basix/quadrature.cpp
fn compute_gll_rule<T: RlstScalar>(m: usize) -> (Vec<T>, Vec<T>) {
    if m < 2 {
        panic!("Gauss-Lobatto-Legendre quadrature invalid for fewer than 2 points");
    }

    // Calculate the recursion coefficients
    let (alpha, beta) = rec_jacobi(m, T::zero(), T::zero());

    println!("A = {alpha:?}");

    // Compute Lobatto nodes and weights
    let (xs_ref, ws_ref) = lobatto(&alpha, &beta, -T::from(1.0).unwrap(), T::from(1.0).unwrap());

  // Reorder to match 1d dof ordering
  //std::rotate(xs_ref.rbegin(), xs_ref.rbegin() + 1, xs_ref.rend() - 1);
  //std::rotate(ws_ref.rbegin(), ws_ref.rbegin() + 1, ws_ref.rend() - 1);
    (xs_ref, ws_ref)
}

/// Adapted from the C++ code by Chris Richardson
/// https://github.com/FEniCS/basix/blob/main/cpp/basix/quadrature.cpp
fn make_gll_line<T: RlstScalar>(m: usize) -> (Vec<T>, Vec<T>) {
    let (mut pts, mut wts) = compute_gll_rule(m);
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

/// Evaluate the nth Jacobi polynomial and derivatives with weight
/// parameters (a, 0) at points x
/// @param[in] a Jacobi weight a
/// @param[in] n Order of polynomial
/// @param[in] nderiv Number of derivatives (if zero, just compute
/// polynomial itself)
/// @param[in] x Points at which to evaluate
/// @return Array of polynomial derivative values (rows) at points
/// (columns)
template <std::floating_point T>
mdarray_t<T, 2> compute_jacobi_deriv(T a, std::size_t n, std::size_t nderiv,
                                     std::span<const T> x)
{
  std::vector<std::size_t> shape = {x.size()};
  mdarray_t<T, 3> J(nderiv + 1, n + 1, x.size());
  mdarray_t<T, 2> Jd(n + 1, x.size());
  for (std::size_t i = 0; i < nderiv + 1; ++i)
  {
    if (i == 0)
    {
      for (std::size_t j = 0; j < Jd.extent(1); ++j)
        Jd(0, j) = 1.0;
    }
    else
    {
      for (std::size_t j = 0; j < Jd.extent(1); ++j)
        Jd(0, j) = 0.0;
    }

    if (n > 0)
    {
      if (i == 0)
      {
        for (std::size_t j = 0; j < Jd.extent(1); ++j)
          Jd(1, j) = (x[j] * (a + 2.0) + a) * 0.5;
      }
      else if (i == 1)
      {
        for (std::size_t j = 0; j < Jd.extent(1); ++j)
          Jd(1, j) = a * 0.5 + 1;
      }
      else
      {
        for (std::size_t j = 0; j < Jd.extent(1); ++j)
          Jd(1, j) = 0.0;
      }
    }

    for (std::size_t j = 2; j < n + 1; ++j)
    {
      const T a1 = 2 * j * (j + a) * (2 * j + a - 2);
      const T a2 = (2 * j + a - 1) * (a * a) / a1;
      const T a3 = (2 * j + a - 1) * (2 * j + a) / (2 * j * (j + a));
      const T a4 = 2 * (j + a - 1) * (j - 1) * (2 * j + a) / a1;
      for (std::size_t k = 0; k < Jd.extent(1); ++k)
        Jd(j, k) = Jd(j - 1, k) * (x[k] * a3 + a2) - Jd(j - 2, k) * a4;
      if (i > 0)
      {
        for (std::size_t k = 0; k < Jd.extent(1); ++k)
          Jd(j, k) += i * a3 * J(i - 1, j - 1, k);
      }
    }

    for (std::size_t j = 0; j < Jd.extent(0); ++j)
      for (std::size_t k = 0; k < Jd.extent(1); ++k)
        J(i, j, k) = Jd(j, k);
  }

  mdarray_t<T, 2> result(nderiv + 1, x.size());
  for (std::size_t i = 0; i < result.extent(0); ++i)
    for (std::size_t j = 0; j < result.extent(1); ++j)
      result(i, j) = J(i, n, j);

  return result;
}
//----------------------------------------------------------------------------

/// Computes the m roots of \f$P_{m}^{a,0}\f$ on [-1,1] by Newton's
/// method. The initial guesses are the Chebyshev points.  Algorithm
/// implemented from the pseudocode given by Karniadakis and Sherwin.
template <std::floating_point T>
std::vector<T> compute_gauss_jacobi_points(T a, int m)
{
  constexpr T eps = 1.0e-8;
  constexpr int max_iter = 100;
  std::vector<T> x(m);
  for (int k = 0; k < m; ++k)
  {
    // Initial guess
    x[k] = -std::cos((2.0 * k + 1.0) * M_PI / (2.0 * m));
    if (k > 0)
      x[k] = 0.5 * (x[k] + x[k - 1]);

    int j = 0;
    while (j < max_iter)
    {
      T s = 0;
      for (int i = 0; i < k; ++i)
        s += 1.0 / (x[k] - x[i]);
      std::span<const T> _x(&x[k], 1);
      mdarray_t<T, 2> f = compute_jacobi_deriv<T>(a, m, 1, _x);
      T delta = f(0, 0) / (f(1, 0) - f(0, 0) * s);
      x[k] -= delta;
      if (std::abs(delta) < eps)
        break;
      ++j;
    }
  }

  return x;
}
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
template <std::floating_point T>
std::array<std::vector<T>, 2> make_quadrature_line(int m)
{
  auto [ptx, wx] = compute_gauss_jacobi_rule<T>(0.0, m);
  std::ranges::transform(wx, wx.begin(), [](auto w) { return 0.5 * w; });
  std::ranges::transform(ptx, ptx.begin(), [](auto x) { return 0.5 * (x + 1.0); });
  return {std::move(ptx), std::move(wx)};
}
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
        let (pts, wts) = compute_gll_rule::<f64>(3);

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
