//! Binding for C
#![allow(missing_docs)]
#![allow(clippy::missing_safety_doc)]

pub mod reference_cell {
    use crate::reference_cell;
    use crate::types::ReferenceCellType;
    use rlst::RlstScalar;

    #[no_mangle]
    pub unsafe extern "C" fn dim(cell: u8) -> usize {
        reference_cell::dim(ReferenceCellType::from(cell).expect("Invalid cell type"))
    }
    #[no_mangle]
    pub unsafe extern "C" fn is_simplex(cell: u8) -> bool {
        reference_cell::is_simplex(ReferenceCellType::from(cell).expect("Invalid cell type"))
    }
    unsafe fn vertices<T: RlstScalar<Real = T>>(cell: u8, vs: *mut T) {
        let mut i = 0;
        for v in
            reference_cell::vertices::<T>(ReferenceCellType::from(cell).expect("Invalid cell type"))
        {
            for c in v {
                *vs.add(i) = c;
                i += 1;
            }
        }
    }
    #[no_mangle]
    pub unsafe extern "C" fn vertices_f32(cell: u8, vs: *mut f32) {
        vertices(cell, vs);
    }
    #[no_mangle]
    pub unsafe extern "C" fn vertices_f64(cell: u8, vs: *mut f64) {
        vertices(cell, vs);
    }
    unsafe fn midpoint<T: RlstScalar<Real = T>>(cell: u8, pt: *mut T) {
        for (i, c) in
            reference_cell::midpoint(ReferenceCellType::from(cell).expect("Invalid cell type"))
                .iter()
                .enumerate()
        {
            *pt.add(i) = *c;
        }
    }
    #[no_mangle]
    pub unsafe extern "C" fn midpoint_f32(cell: u8, pt: *mut f32) {
        midpoint(cell, pt);
    }
    #[no_mangle]
    pub unsafe extern "C" fn midpoint_f64(cell: u8, pt: *mut f64) {
        midpoint(cell, pt);
    }
    #[no_mangle]
    pub unsafe extern "C" fn edges(cell: u8, es: *mut usize) {
        let mut i = 0;
        for e in reference_cell::edges(ReferenceCellType::from(cell).expect("Invalid cell type")) {
            for v in e {
                *es.add(i) = v;
                i += 1
            }
        }
    }
    #[no_mangle]
    pub unsafe extern "C" fn faces(cell: u8, es: *mut usize) {
        let mut i = 0;
        for e in reference_cell::faces(ReferenceCellType::from(cell).expect("Invalid cell type")) {
            for v in e {
                *es.add(i) = v;
                i += 1
            }
        }
    }
    #[no_mangle]
    pub unsafe extern "C" fn volumes(cell: u8, es: *mut usize) {
        let mut i = 0;
        for e in reference_cell::volumes(ReferenceCellType::from(cell).expect("Invalid cell type"))
        {
            for v in e {
                *es.add(i) = v;
                i += 1
            }
        }
    }
    #[no_mangle]
    pub unsafe extern "C" fn entity_types(cell: u8, et: *mut u8) {
        let mut i = 0;
        for es in
            reference_cell::entity_types(ReferenceCellType::from(cell).expect("Invalid cell type"))
        {
            for e in es {
                *et.add(i) = e as u8;
                i += 1
            }
        }
    }
    #[no_mangle]
    pub unsafe extern "C" fn entity_counts(cell: u8, ec: *mut usize) {
        for (i, e) in
            reference_cell::entity_counts(ReferenceCellType::from(cell).expect("Invalid cell type"))
                .iter()
                .enumerate()
        {
            *ec.add(i) = *e;
        }
    }
    #[no_mangle]
    pub unsafe extern "C" fn connectivity_size(
        cell: u8,
        dim0: usize,
        index0: usize,
        dim1: usize,
    ) -> usize {
        reference_cell::connectivity(ReferenceCellType::from(cell).expect("Invalid cell type"))
            [dim0][index0][dim1]
            .len()
    }
    #[no_mangle]
    pub unsafe extern "C" fn connectivity(
        cell: u8,
        dim0: usize,
        index0: usize,
        dim1: usize,
        c: *mut usize,
    ) {
        for (i, j) in
            reference_cell::connectivity(ReferenceCellType::from(cell).expect("Invalid cell type"))
                [dim0][index0][dim1]
                .iter()
                .enumerate()
        {
            *c.add(i) = *j;
        }
    }
}

pub mod quadrature {
    use crate::quadrature;
    use crate::traits::QuadratureRule;
    use crate::types::ReferenceCellType;
    use num::traits::FloatConst;
    use rlst::RlstScalar;
    use std::cmp::PartialOrd;

    #[no_mangle]
    pub unsafe extern "C" fn gauss_jacobi_quadrature_npoints(cell: u8, m: usize) -> usize {
        quadrature::gauss_jacobi_quadrature_npoints(
            ReferenceCellType::from(cell).expect("Invalid cell type"),
            m,
        )
    }
    unsafe fn make_gauss_jacobi_quadrature<T: RlstScalar<Real = T> + FloatConst + PartialOrd>(
        cell: u8,
        m: usize,
        pts: *mut T,
        wts: *mut T,
    ) {
        let rule = quadrature::make_gauss_jacobi_quadrature::<T>(
            ReferenceCellType::from(cell).expect("Invalid cell type"),
            m,
        );
        for (i, p) in rule.points().iter().enumerate() {
            *pts.add(i) = *p;
        }
        for (i, w) in rule.weights().iter().enumerate() {
            *wts.add(i) = *w;
        }
    }
    #[no_mangle]
    pub unsafe extern "C" fn make_gauss_jacobi_quadrature_f32(
        cell: u8,
        m: usize,
        pts: *mut f32,
        wts: *mut f32,
    ) {
        make_gauss_jacobi_quadrature(cell, m, pts, wts);
    }
    #[no_mangle]
    pub unsafe extern "C" fn make_gauss_jacobi_quadrature_f64(
        cell: u8,
        m: usize,
        pts: *mut f64,
        wts: *mut f64,
    ) {
        make_gauss_jacobi_quadrature(cell, m, pts, wts);
    }
}

pub mod polynomials {
    use crate::types::ReferenceCellType;
    use crate::{polynomials, reference_cell};
    use rlst::{rlst_array_from_slice2, rlst_array_from_slice_mut3, RlstScalar};
    use std::slice::{from_raw_parts, from_raw_parts_mut};

    #[no_mangle]
    pub unsafe extern "C" fn legendre_polynomials_shape(
        cell: u8,
        npts: usize,
        degree: usize,
        derivatives: usize,
        shape: *mut usize,
    ) {
        let cell_type = ReferenceCellType::from(cell).expect("Invalid cell type");
        *shape.add(0) = polynomials::derivative_count(cell_type, derivatives);
        *shape.add(1) = polynomials::polynomial_count(cell_type, degree);
        *shape.add(2) = npts;
    }
    unsafe fn tabulate_legendre_polynomials<T: RlstScalar>(
        cell: u8,
        points: *const T::Real,
        npts: usize,
        degree: usize,
        derivatives: usize,
        data: *mut T,
    ) {
        let cell_type = ReferenceCellType::from(cell).expect("Invalid cell type");
        let tdim = reference_cell::dim(cell_type);
        let points = rlst_array_from_slice2!(from_raw_parts(points, npts * tdim), [tdim, npts]);
        let npoly = polynomials::polynomial_count(cell_type, degree);
        let nderiv = polynomials::derivative_count(cell_type, derivatives);
        let mut data = rlst_array_from_slice_mut3!(
            from_raw_parts_mut(data, npts * npoly * nderiv),
            [nderiv, npoly, npts]
        );
        polynomials::tabulate_legendre_polynomials(
            cell_type,
            &points,
            degree,
            derivatives,
            &mut data,
        );
    }
    #[no_mangle]
    pub unsafe extern "C" fn tabulate_legendre_polynomials_f32(
        cell: u8,
        points: *const f32,
        npts: usize,
        degree: usize,
        derivatives: usize,
        data: *mut f32,
    ) {
        tabulate_legendre_polynomials(cell, points, npts, degree, derivatives, data);
    }
    #[no_mangle]
    pub unsafe extern "C" fn tabulate_legendre_polynomials_f64(
        cell: u8,
        points: *const f64,
        npts: usize,
        degree: usize,
        derivatives: usize,
        data: *mut f64,
    ) {
        tabulate_legendre_polynomials(cell, points, npts, degree, derivatives, data);
    }
}
