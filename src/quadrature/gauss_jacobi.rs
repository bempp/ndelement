//! Gauss-Jacobi quadrature
//!
//! Adapted from the C++ code by Chris Richardson
//! <https://github.com/FEniCS/basix/blob/main/cpp/basix/quadrature.cpp>
use super::QuadratureRule;
use crate::traits::QuadratureRule as QuadratureRuleTrait;
use crate::types::{Array2D, ReferenceCellType};
use itertools::izip;
use num::traits::FloatConst;
use rlst::{rlst_dynamic_array2, rlst_dynamic_array3, DefaultIteratorMut, RlstScalar};
use std::cmp::PartialOrd;

/// Evaluate the nth Jacobi polynomial and derivatives with weight
/// parameters (a, 0) at points x
fn compute_deriv<T: RlstScalar<Real = T>>(a: T, n: usize, nderiv: usize, x: &[T]) -> Array2D<T> {
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
                for (j, x_j) in x.iter().enumerate() {
                    j_all[[i, 1, j]] = (*x_j * (a + two) + a) / two;
                }
            } else if i == 1 {
                for j in 0..x.len() {
                    j_all[[i, 1, j]] = a / two + one;
                }
            }
        }
        for j in 2..=n {
            let j_t = T::from(j).unwrap();
            let a1 = two * j_t * (j_t + a) * (two * j_t + a - two);
            let a2 = (two * j_t + a - one) * (a * a) / a1;
            let a3 = (two * j_t + a - one) * (two * j_t + a) / (two * j_t * (j_t + a));
            let a4 = two * (j_t + a - one) * (j_t - one) * (two * j_t + a) / a1;
            for (k, x_k) in x.iter().enumerate() {
                j_all[[i, j, k]] =
                    j_all[[i, j - 1, k]] * (*x_k * a3 + a2) - j_all[[i, j - 2, k]] * a4;
            }
            if i > 0 {
                for (k, _) in x.iter().enumerate() {
                    let add = T::from(i).unwrap() * a3 * j_all[[i - 1, j - 1, k]];
                    j_all[[i, j, k]] += add;
                }
            }
        }
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
fn compute_points<T: RlstScalar<Real = T> + FloatConst + PartialOrd>(a: T, m: usize) -> Vec<T> {
    let eps = T::from(1.0e-8).unwrap();
    let max_iter = 100;
    let two = T::from(2.0).unwrap();
    let one = T::from(1.0).unwrap();

    let mut x = vec![T::zero(); m];

    for k in 0..m {
        // Initial guess
        x[k] = -T::cos(T::from(2 * k + 1).unwrap() * T::PI() / T::from(2 * m).unwrap());
        if k > 0 {
            x[k] = (x[k] + x[k - 1]) / two;
        }

        for _ in 0..max_iter {
            let s = x.iter().take(k).map(|i| one / (x[k] - *i)).sum::<T>();
            let f = compute_deriv(a, m, 1, &x[k..k + 1]);
            let delta = f[[0, 0]] / (f[[1, 0]] - f[[0, 0]] * s);
            x[k] -= delta;
            if delta.abs() < eps {
                break;
            }
        }
    }
    x
}

/// Note: computes on [-1, 1]
fn compute_rule<T: RlstScalar<Real = T> + FloatConst + PartialOrd>(
    a: T,
    m: usize,
) -> (Vec<T>, Vec<T>) {
    let one = T::from(1.0).unwrap();
    let two = T::from(2.0).unwrap();

    let pts = compute_points(a, m);
    let j_d = compute_deriv(a, m, 1, &pts);
    let a1 = T::pow(two, a + one);
    let wts = pts
        .iter()
        .enumerate()
        .map(|(i, x)| a1 / (one - x.powi(2)) / j_d[[1, i]].powi(2))
        .collect::<Vec<_>>();

    (pts, wts)
}

fn make_quadrature_line<T: RlstScalar<Real = T> + FloatConst + PartialOrd>(
    m: usize,
) -> QuadratureRule<T> {
    let (mut pts, mut wts) = compute_rule(T::zero(), m);

    let half = T::from(0.5).unwrap();
    let one = T::from(1.0).unwrap();
    for p in pts.iter_mut() {
        *p = half * (*p + one);
    }
    for w in wts.iter_mut() {
        *w *= half;
    }
    QuadratureRule::new(pts, wts)
}

fn make_quadrature_triangle_collapsed<T: RlstScalar<Real = T> + FloatConst + PartialOrd>(
    m: usize,
) -> QuadratureRule<T> {
    let one = T::from(1.0).unwrap();

    let (ptx, wtx) = compute_rule(T::zero(), m);
    let (pty, wty) = compute_rule(T::from(1).unwrap(), m);

    let mut pts = vec![T::zero(); m.pow(2) * 2];
    let mut wts = vec![T::zero(); m.pow(2)];

    for (i, (px, wx)) in izip!(&ptx, &wtx).enumerate() {
        for (j, (py, wy)) in izip!(&pty, &wty).enumerate() {
            let index = i * wty.len() + j;
            pts[2 * index] = T::from(0.25).unwrap() * (one + *px) * (one - *py);
            pts[2 * index + 1] = T::from(0.5).unwrap() * (one + *py);
            wts[index] = *wx * *wy * T::from(0.125).unwrap();
        }
    }
    QuadratureRule::new(pts, wts)
}

fn make_quadrature_tetrahedron_collapsed<T: RlstScalar<Real = T> + FloatConst + PartialOrd>(
    m: usize,
) -> QuadratureRule<T> {
    let one = T::from(1.0).unwrap();

    let (ptx, wtx) = compute_rule(T::zero(), m);
    let (pty, wty) = compute_rule(T::from(1).unwrap(), m);
    let (ptz, wtz) = compute_rule(T::from(2).unwrap(), m);

    let mut pts = vec![T::zero(); m.pow(3) * 3];
    let mut wts = vec![T::zero(); m.pow(3)];

    for (i, (px, wx)) in izip!(&ptx, &wtx).enumerate() {
        for (j, (py, wy)) in izip!(&pty, &wty).enumerate() {
            for (k, (pz, wz)) in izip!(&ptz, &wtz).enumerate() {
                let index = i * wty.len() * wtz.len() + j * wtz.len() + k;
                pts[3 * index] = T::from(0.125).unwrap() * (one + *px) * (one - *py) * (one - *pz);
                pts[3 * index + 1] = T::from(0.25).unwrap() * (one + *py) * (one - *pz);
                pts[3 * index + 2] = T::from(0.5).unwrap() * (one + *pz);
                wts[index] = *wx * *wy * *wz * T::from(0.015625).unwrap();
            }
        }
    }
    QuadratureRule::new(pts, wts)
}

/// Get the number of quadrature points for a given rule
pub fn npoints(celltype: ReferenceCellType, m: usize) -> usize {
    let np = (m + 2) / 2;
    match celltype {
        ReferenceCellType::Interval => np,
        ReferenceCellType::Quadrilateral => np.pow(2),
        ReferenceCellType::Hexahedron => np.pow(3),
        ReferenceCellType::Triangle => np.pow(2),
        ReferenceCellType::Tetrahedron => np.pow(3),
        _ => {
            panic!("Unsupported cell type");
        }
    }
}

/// Get the points and weights for a Gauss-Jacobi quadrature rule
pub fn make_quadrature<T: RlstScalar<Real = T> + FloatConst + PartialOrd>(
    celltype: ReferenceCellType,
    m: usize,
) -> QuadratureRule<T> {
    let np = (m + 2) / 2;
    match celltype {
        ReferenceCellType::Interval => make_quadrature_line::<T>(np),
        ReferenceCellType::Quadrilateral => {
            let rule = make_quadrature_line::<T>(np);
            let mut pts = vec![T::zero(); np.pow(2) * 2];
            let mut wts = vec![T::zero(); np.pow(2)];

            for (i, (pi, wi)) in izip!(rule.points(), rule.weights()).enumerate() {
                for (j, (pj, wj)) in izip!(rule.points(), rule.weights()).enumerate() {
                    let index = i * np + j;
                    pts[2 * index] = *pi;
                    pts[2 * index + 1] = *pj;
                    wts[index] = *wi * *wj;
                }
            }
            QuadratureRule::new(pts, wts)
        }
        ReferenceCellType::Hexahedron => {
            let rule = make_quadrature_line::<T>(np);
            let mut pts = vec![T::zero(); np.pow(3) * 3];
            let mut wts = vec![T::zero(); np.pow(3)];

            for (i, (pi, wi)) in izip!(rule.points(), rule.weights()).enumerate() {
                for (j, (pj, wj)) in izip!(rule.points(), rule.weights()).enumerate() {
                    for (k, (pk, wk)) in izip!(rule.points(), rule.weights()).enumerate() {
                        let index = i * np.pow(2) + j * np + k;
                        pts[3 * index] = *pi;
                        pts[3 * index + 1] = *pj;
                        pts[3 * index + 2] = *pk;
                        wts[index] = *wi * *wj * *wk;
                    }
                }
            }
            QuadratureRule::new(pts, wts)
        }
        ReferenceCellType::Triangle => make_quadrature_triangle_collapsed::<T>(np),
        ReferenceCellType::Tetrahedron => make_quadrature_tetrahedron_collapsed::<T>(np),
        _ => {
            panic!("Unsupported cell type");
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::*;

    #[test]
    fn test_interval_3() {
        let rule = make_quadrature_line::<f64>(3);

        for (p, q) in izip!(rule.points(), [0.1127016653792583, 0.5, 0.8872983346207417]) {
            assert_relative_eq!(*p, q);
        }
        for (w, v) in izip!(
            rule.weights(),
            [0.2777777777777777, 0.4444444444444444, 0.2777777777777777]
        ) {
            assert_relative_eq!(*w, v);
        }
    }
}
