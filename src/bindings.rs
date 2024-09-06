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

pub mod ciarlet {
    use crate::reference_cell;
    use crate::{ciarlet, ciarlet::CiarletElement};
    use crate::{
        traits::{ElementFamily, FiniteElement},
        types::{Continuity, ReferenceCellType},
    };
    use rlst::{
        c32, c64, rlst_array_from_slice2, rlst_array_from_slice_mut4, MatrixInverse, RawAccess,
        RlstScalar, Shape,
    };
    use std::ffi::c_void;
    use std::slice::{from_raw_parts, from_raw_parts_mut};

    #[derive(Debug, PartialEq, Clone, Copy)]
    #[repr(u8)]
    pub enum DType {
        F32 = 0,
        F64 = 1,
        C32 = 2,
        C64 = 3,
    }

    #[derive(Debug, PartialEq, Clone, Copy)]
    #[repr(u8)]
    pub enum ElementType {
        Lagrange = 0,
        RaviartThomas = 1,
    }

    #[repr(C)]
    pub struct CiarletElementWrapper {
        pub element: *const c_void,
        pub dtype: DType,
    }

    #[repr(C)]
    pub struct ElementFamilyWrapper {
        pub etype: ElementType,
        pub dtype: DType,
        pub family: *const c_void,
    }

    impl Drop for CiarletElementWrapper {
        fn drop(&mut self) {
            let Self { element, dtype } = self;
            match dtype {
                DType::F32 => {
                    drop(unsafe { Box::from_raw(*element as *mut ciarlet::CiarletElement<f32>) })
                }
                DType::F64 => {
                    drop(unsafe { Box::from_raw(*element as *mut ciarlet::CiarletElement<f64>) })
                }
                DType::C32 => {
                    drop(unsafe { Box::from_raw(*element as *mut ciarlet::CiarletElement<c32>) })
                }
                DType::C64 => {
                    drop(unsafe { Box::from_raw(*element as *mut ciarlet::CiarletElement<c64>) })
                }
            }
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn ciarlet_free_element(e: *mut CiarletElementWrapper) {
        assert!(!e.is_null());
        unsafe { drop(Box::from_raw(e)) }
    }

    impl Drop for ElementFamilyWrapper {
        fn drop(&mut self) {
            let Self {
                family,
                etype,
                dtype,
            } = self;
            match etype {
                ElementType::Lagrange => match dtype {
                    DType::F32 => drop(unsafe {
                        Box::from_raw(*family as *mut ciarlet::LagrangeElementFamily<f32>)
                    }),
                    DType::F64 => drop(unsafe {
                        Box::from_raw(*family as *mut ciarlet::LagrangeElementFamily<f64>)
                    }),
                    DType::C32 => drop(unsafe {
                        Box::from_raw(*family as *mut ciarlet::LagrangeElementFamily<c32>)
                    }),
                    DType::C64 => drop(unsafe {
                        Box::from_raw(*family as *mut ciarlet::LagrangeElementFamily<c64>)
                    }),
                },
                ElementType::RaviartThomas => match dtype {
                    DType::F32 => drop(unsafe {
                        Box::from_raw(*family as *mut ciarlet::RaviartThomasElementFamily<f32>)
                    }),
                    DType::F64 => drop(unsafe {
                        Box::from_raw(*family as *mut ciarlet::RaviartThomasElementFamily<f64>)
                    }),
                    DType::C32 => drop(unsafe {
                        Box::from_raw(*family as *mut ciarlet::RaviartThomasElementFamily<c32>)
                    }),
                    DType::C64 => drop(unsafe {
                        Box::from_raw(*family as *mut ciarlet::RaviartThomasElementFamily<c64>)
                    }),
                },
            }
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn ciarlet_free_family(f: *mut ElementFamilyWrapper) {
        assert!(!f.is_null());
        unsafe { drop(Box::from_raw(f)) }
    }

    unsafe fn extract_element<T: RlstScalar + MatrixInverse>(
        element: *const CiarletElementWrapper,
    ) -> *const CiarletElement<T> {
        (*element).element as *const CiarletElement<T>
    }

    #[no_mangle]
    pub unsafe extern "C" fn ciarlet_value_size(element: *const CiarletElementWrapper) -> usize {
        match (*element).dtype {
            DType::F32 => (*extract_element::<f32>(element)).value_size(),
            DType::F64 => (*extract_element::<f64>(element)).value_size(),
            DType::C32 => (*extract_element::<c32>(element)).value_size(),
            DType::C64 => (*extract_element::<c64>(element)).value_size(),
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn ciarlet_degree(element: *const CiarletElementWrapper) -> usize {
        match (*element).dtype {
            DType::F32 => (*extract_element::<f32>(element)).degree(),
            DType::F64 => (*extract_element::<f64>(element)).degree(),
            DType::C32 => (*extract_element::<c32>(element)).degree(),
            DType::C64 => (*extract_element::<c64>(element)).degree(),
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn ciarlet_embedded_superdegree(
        element: *const CiarletElementWrapper,
    ) -> usize {
        match (*element).dtype {
            DType::F32 => (*extract_element::<f32>(element)).embedded_superdegree(),
            DType::F64 => (*extract_element::<f64>(element)).embedded_superdegree(),
            DType::C32 => (*extract_element::<c32>(element)).embedded_superdegree(),
            DType::C64 => (*extract_element::<c64>(element)).embedded_superdegree(),
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn ciarlet_dim(element: *const CiarletElementWrapper) -> usize {
        match (*element).dtype {
            DType::F32 => (*extract_element::<f32>(element)).dim(),
            DType::F64 => (*extract_element::<f64>(element)).dim(),
            DType::C32 => (*extract_element::<c32>(element)).dim(),
            DType::C64 => (*extract_element::<c64>(element)).dim(),
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn ciarlet_continuity(element: *const CiarletElementWrapper) -> u8 {
        match (*element).dtype {
            DType::F32 => (*extract_element::<f32>(element)).continuity() as u8,
            DType::F64 => (*extract_element::<f64>(element)).continuity() as u8,
            DType::C32 => (*extract_element::<c32>(element)).continuity() as u8,
            DType::C64 => (*extract_element::<c64>(element)).continuity() as u8,
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn ciarlet_map_type(element: *const CiarletElementWrapper) -> u8 {
        match (*element).dtype {
            DType::F32 => (*extract_element::<f32>(element)).map_type() as u8,
            DType::F64 => (*extract_element::<f64>(element)).map_type() as u8,
            DType::C32 => (*extract_element::<c32>(element)).map_type() as u8,
            DType::C64 => (*extract_element::<c64>(element)).map_type() as u8,
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn ciarlet_cell_type(element: *const CiarletElementWrapper) -> u8 {
        match (*element).dtype {
            DType::F32 => (*extract_element::<f32>(element)).cell_type() as u8,
            DType::F64 => (*extract_element::<f64>(element)).cell_type() as u8,
            DType::C32 => (*extract_element::<c32>(element)).cell_type() as u8,
            DType::C64 => (*extract_element::<c64>(element)).cell_type() as u8,
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn ciarlet_value_shape(
        element: *const CiarletElementWrapper,
        shape: *mut usize,
    ) {
        for (i, j) in match (*element).dtype {
            DType::F32 => (*extract_element::<f32>(element)).value_shape(),
            DType::F64 => (*extract_element::<f64>(element)).value_shape(),
            DType::C32 => (*extract_element::<c32>(element)).value_shape(),
            DType::C64 => (*extract_element::<c64>(element)).value_shape(),
        }
        .iter()
        .enumerate()
        {
            *shape.add(i) = *j;
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn ciarlet_value_rank(element: *const CiarletElementWrapper) -> usize {
        match (*element).dtype {
            DType::F32 => (*extract_element::<f32>(element)).value_shape(),
            DType::F64 => (*extract_element::<f64>(element)).value_shape(),
            DType::C32 => (*extract_element::<c32>(element)).value_shape(),
            DType::C64 => (*extract_element::<c64>(element)).value_shape(),
        }
        .len()
    }

    #[no_mangle]
    pub unsafe extern "C" fn ciarlet_entity_dofs_size(
        element: *const CiarletElementWrapper,
        entity_dim: usize,
        entity_number: usize,
    ) -> usize {
        match (*element).dtype {
            DType::F32 => (*extract_element::<f32>(element)).entity_dofs(entity_dim, entity_number),
            DType::F64 => (*extract_element::<f64>(element)).entity_dofs(entity_dim, entity_number),
            DType::C32 => (*extract_element::<c32>(element)).entity_dofs(entity_dim, entity_number),
            DType::C64 => (*extract_element::<c64>(element)).entity_dofs(entity_dim, entity_number),
        }
        .unwrap()
        .len()
    }

    #[no_mangle]
    pub unsafe extern "C" fn ciarlet_entity_dofs(
        element: *const CiarletElementWrapper,
        entity_dim: usize,
        entity_number: usize,
        entity_dofs: *mut usize,
    ) {
        for (i, dof) in match (*element).dtype {
            DType::F32 => (*extract_element::<f32>(element)).entity_dofs(entity_dim, entity_number),
            DType::F64 => (*extract_element::<f64>(element)).entity_dofs(entity_dim, entity_number),
            DType::C32 => (*extract_element::<c32>(element)).entity_dofs(entity_dim, entity_number),
            DType::C64 => (*extract_element::<c64>(element)).entity_dofs(entity_dim, entity_number),
        }
        .unwrap()
        .iter()
        .enumerate()
        {
            *entity_dofs.add(i) = *dof;
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn ciarlet_entity_closure_dofs_size(
        element: *const CiarletElementWrapper,
        entity_dim: usize,
        entity_number: usize,
    ) -> usize {
        match (*element).dtype {
            DType::F32 => {
                (*extract_element::<f32>(element)).entity_closure_dofs(entity_dim, entity_number)
            }
            DType::F64 => {
                (*extract_element::<f64>(element)).entity_closure_dofs(entity_dim, entity_number)
            }
            DType::C32 => {
                (*extract_element::<c32>(element)).entity_closure_dofs(entity_dim, entity_number)
            }
            DType::C64 => {
                (*extract_element::<c64>(element)).entity_closure_dofs(entity_dim, entity_number)
            }
        }
        .unwrap()
        .len()
    }

    #[no_mangle]
    pub unsafe extern "C" fn ciarlet_entity_closure_dofs(
        element: *const CiarletElementWrapper,
        entity_dim: usize,
        entity_number: usize,
        entity_dofs: *mut usize,
    ) {
        for (i, dof) in
            match (*element).dtype {
                DType::F32 => (*extract_element::<f32>(element))
                    .entity_closure_dofs(entity_dim, entity_number),
                DType::F64 => (*extract_element::<f64>(element))
                    .entity_closure_dofs(entity_dim, entity_number),
                DType::C32 => (*extract_element::<c32>(element))
                    .entity_closure_dofs(entity_dim, entity_number),
                DType::C64 => (*extract_element::<c64>(element))
                    .entity_closure_dofs(entity_dim, entity_number),
            }
            .unwrap()
            .iter()
            .enumerate()
        {
            *entity_dofs.add(i) = *dof;
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn ciarlet_interpolation_npoints(
        element: *const CiarletElementWrapper,
        entity_dim: usize,
        entity_index: usize,
    ) -> usize {
        match (*element).dtype {
            DType::F32 => (*extract_element::<f32>(element)).interpolation_points()[entity_dim]
                [entity_index]
                .shape()[1],
            DType::F64 => (*extract_element::<f64>(element)).interpolation_points()[entity_dim]
                [entity_index]
                .shape()[1],
            DType::C32 => (*extract_element::<c32>(element)).interpolation_points()[entity_dim]
                [entity_index]
                .shape()[1],
            DType::C64 => (*extract_element::<c64>(element)).interpolation_points()[entity_dim]
                [entity_index]
                .shape()[1],
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn ciarlet_interpolation_ndofs(
        element: *const CiarletElementWrapper,
        entity_dim: usize,
        entity_index: usize,
    ) -> usize {
        match (*element).dtype {
            DType::F32 => (*extract_element::<f32>(element)).interpolation_weights()[entity_dim]
                [entity_index]
                .shape()[0],
            DType::F64 => (*extract_element::<f64>(element)).interpolation_weights()[entity_dim]
                [entity_index]
                .shape()[0],
            DType::C32 => (*extract_element::<c32>(element)).interpolation_weights()[entity_dim]
                [entity_index]
                .shape()[0],
            DType::C64 => (*extract_element::<c64>(element)).interpolation_weights()[entity_dim]
                [entity_index]
                .shape()[0],
        }
    }

    unsafe fn ciarlet_interpolation_points_internal<T: RlstScalar + MatrixInverse>(
        element: *const CiarletElementWrapper,
        entity_dim: usize,
        entity_index: usize,
        points: *mut T::Real,
    ) {
        for (i, j) in (*extract_element::<T>(element)).interpolation_points()[entity_dim]
            [entity_index]
            .data()
            .iter()
            .enumerate()
        {
            *points.add(i) = *j;
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn ciarlet_interpolation_points(
        element: *const CiarletElementWrapper,
        entity_dim: usize,
        entity_index: usize,
        points: *mut c_void,
    ) {
        match (*element).dtype {
            DType::F32 => ciarlet_interpolation_points_internal::<f32>(
                element,
                entity_dim,
                entity_index,
                points as *mut f32,
            ),
            DType::F64 => ciarlet_interpolation_points_internal::<f64>(
                element,
                entity_dim,
                entity_index,
                points as *mut f64,
            ),
            DType::C32 => ciarlet_interpolation_points_internal::<c32>(
                element,
                entity_dim,
                entity_index,
                points as *mut f32,
            ),
            DType::C64 => ciarlet_interpolation_points_internal::<c64>(
                element,
                entity_dim,
                entity_index,
                points as *mut f64,
            ),
        }
    }

    unsafe fn ciarlet_interpolation_weights_internal<T: RlstScalar + MatrixInverse>(
        element: *const CiarletElementWrapper,
        entity_dim: usize,
        entity_index: usize,
        weights: *mut T,
    ) {
        for (i, j) in (*extract_element::<T>(element)).interpolation_weights()[entity_dim]
            [entity_index]
            .data()
            .iter()
            .enumerate()
        {
            *weights.add(i) = *j;
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn ciarlet_interpolation_weights(
        element: *const CiarletElementWrapper,
        entity_dim: usize,
        entity_index: usize,
        weights: *mut c_void,
    ) {
        match (*element).dtype {
            DType::F32 => ciarlet_interpolation_weights_internal::<f32>(
                element,
                entity_dim,
                entity_index,
                weights as *mut f32,
            ),
            DType::F64 => ciarlet_interpolation_weights_internal::<f64>(
                element,
                entity_dim,
                entity_index,
                weights as *mut f64,
            ),
            DType::C32 => ciarlet_interpolation_weights_internal::<c32>(
                element,
                entity_dim,
                entity_index,
                weights as *mut c32,
            ),
            DType::C64 => ciarlet_interpolation_weights_internal::<c64>(
                element,
                entity_dim,
                entity_index,
                weights as *mut c64,
            ),
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn ciarlet_element_dtype(element: *const CiarletElementWrapper) -> u8 {
        (*element).dtype as u8
    }

    #[no_mangle]
    pub unsafe extern "C" fn ciarlet_tabulate_array_shape(
        element: *const CiarletElementWrapper,
        nderivs: usize,
        npoints: usize,
        shape: *mut usize,
    ) {
        for (i, j) in match (*element).dtype {
            DType::F32 => (*extract_element::<f32>(element)).tabulate_array_shape(nderivs, npoints),
            DType::F64 => (*extract_element::<f64>(element)).tabulate_array_shape(nderivs, npoints),
            DType::C32 => (*extract_element::<c32>(element)).tabulate_array_shape(nderivs, npoints),
            DType::C64 => (*extract_element::<c64>(element)).tabulate_array_shape(nderivs, npoints),
        }
        .iter()
        .enumerate()
        {
            *shape.add(i) = *j;
        }
    }

    unsafe fn ciarlet_tabulate_internal<T: RlstScalar + MatrixInverse>(
        element: *const CiarletElementWrapper,
        points: *const T::Real,
        npoints: usize,
        nderivs: usize,
        data: *mut T,
    ) {
        let element = extract_element::<T>(element);
        let tdim = reference_cell::dim((*element).cell_type());
        let points =
            rlst_array_from_slice2!(from_raw_parts(points, npoints * tdim), [tdim, npoints]);
        let shape = (*element).tabulate_array_shape(nderivs, npoints);
        let mut data = rlst_array_from_slice_mut4!(
            from_raw_parts_mut(data, shape[0] * shape[1] * shape[2] * shape[3]),
            shape
        );
        (*element).tabulate(&points, nderivs, &mut data);
    }
    #[no_mangle]
    pub unsafe extern "C" fn ciarlet_tabulate(
        element: *const CiarletElementWrapper,
        points: *const c_void,
        npoints: usize,
        nderivs: usize,
        data: *mut c_void,
    ) {
        match (*element).dtype {
            DType::F32 => ciarlet_tabulate_internal::<f32>(
                element,
                points as *const f32,
                npoints,
                nderivs,
                data as *mut f32,
            ),
            DType::F64 => ciarlet_tabulate_internal::<f64>(
                element,
                points as *const f64,
                npoints,
                nderivs,
                data as *mut f64,
            ),
            DType::C32 => ciarlet_tabulate_internal::<c32>(
                element,
                points as *const f32,
                npoints,
                nderivs,
                data as *mut c32,
            ),
            DType::C64 => ciarlet_tabulate_internal::<c64>(
                element,
                points as *const f64,
                npoints,
                nderivs,
                data as *mut c64,
            ),
        }
    }

    unsafe fn element_family_element_internal<
        T: RlstScalar + MatrixInverse,
        F: ElementFamily<T = T, CellType = ReferenceCellType>,
    >(
        family: *const ElementFamilyWrapper,
        cell: u8,
    ) -> *const CiarletElementWrapper {
        let element = Box::into_raw(Box::new(
            (*((*family).family as *const F))
                .element(ReferenceCellType::from(cell).expect("Invalid cell type")),
        )) as *mut c_void;

        Box::into_raw(Box::new(CiarletElementWrapper {
            element,
            dtype: (*family).dtype,
        }))
    }

    #[no_mangle]
    pub unsafe extern "C" fn element_family_element(
        family: *const ElementFamilyWrapper,
        cell: u8,
    ) -> *const CiarletElementWrapper {
        match (*family).etype {
            ElementType::Lagrange => match (*family).dtype {
                DType::F32 => element_family_element_internal::<
                    f32,
                    ciarlet::LagrangeElementFamily<f32>,
                >(family, cell),
                DType::F64 => element_family_element_internal::<
                    f64,
                    ciarlet::LagrangeElementFamily<f64>,
                >(family, cell),
                DType::C32 => element_family_element_internal::<
                    c32,
                    ciarlet::LagrangeElementFamily<c32>,
                >(family, cell),
                DType::C64 => element_family_element_internal::<
                    c64,
                    ciarlet::LagrangeElementFamily<c64>,
                >(family, cell),
            },
            ElementType::RaviartThomas => match (*family).dtype {
                DType::F32 => element_family_element_internal::<
                    f32,
                    ciarlet::RaviartThomasElementFamily<f32>,
                >(family, cell),
                DType::F64 => element_family_element_internal::<
                    f64,
                    ciarlet::RaviartThomasElementFamily<f64>,
                >(family, cell),
                DType::C32 => element_family_element_internal::<
                    c32,
                    ciarlet::RaviartThomasElementFamily<c32>,
                >(family, cell),
                DType::C64 => element_family_element_internal::<
                    c64,
                    ciarlet::RaviartThomasElementFamily<c64>,
                >(family, cell),
            },
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn lagrange_element_family_new_f32(
        degree: usize,
        continuity: u8,
    ) -> *const ElementFamilyWrapper {
        let family = Box::into_raw(Box::new(ciarlet::LagrangeElementFamily::<f32>::new(
            degree,
            Continuity::from(continuity).expect("Invalid continuity"),
        ))) as *mut c_void;

        Box::into_raw(Box::new(ElementFamilyWrapper {
            family,
            etype: ElementType::Lagrange,
            dtype: DType::F32,
        }))
    }

    #[no_mangle]
    pub unsafe extern "C" fn lagrange_element_family_new_f64(
        degree: usize,
        continuity: u8,
    ) -> *const ElementFamilyWrapper {
        let family = Box::into_raw(Box::new(ciarlet::LagrangeElementFamily::<f64>::new(
            degree,
            Continuity::from(continuity).expect("Invalid continuity"),
        ))) as *mut c_void;

        Box::into_raw(Box::new(ElementFamilyWrapper {
            family,
            etype: ElementType::Lagrange,
            dtype: DType::F64,
        }))
    }

    #[no_mangle]
    pub unsafe extern "C" fn raviart_thomas_element_family_new_f32(
        degree: usize,
        continuity: u8,
    ) -> *const ElementFamilyWrapper {
        let family = Box::into_raw(Box::new(ciarlet::RaviartThomasElementFamily::<f32>::new(
            degree,
            Continuity::from(continuity).expect("Invalid continuity"),
        ))) as *mut c_void;

        Box::into_raw(Box::new(ElementFamilyWrapper {
            family,
            etype: ElementType::RaviartThomas,
            dtype: DType::F32,
        }))
    }

    #[no_mangle]
    pub unsafe extern "C" fn raviart_thomas_element_family_new_f64(
        degree: usize,
        continuity: u8,
    ) -> *const ElementFamilyWrapper {
        let family = Box::into_raw(Box::new(ciarlet::RaviartThomasElementFamily::<f64>::new(
            degree,
            Continuity::from(continuity).expect("Invalid continuity"),
        ))) as *mut c_void;

        Box::into_raw(Box::new(ElementFamilyWrapper {
            family,
            etype: ElementType::RaviartThomas,
            dtype: DType::F64,
        }))
    }
}
