//! Binding for C
#![allow(missing_docs)]
#![allow(clippy::missing_safety_doc)]

pub mod reference_cell {
    use crate::reference_cell;
    use crate::types::ReferenceCellType;
    use rlst::RlstScalar;

    #[no_mangle]
    pub unsafe extern "C" fn dim(cell: ReferenceCellType) -> usize {
        reference_cell::dim(cell)
    }
    #[no_mangle]
    pub unsafe extern "C" fn is_simplex(cell: ReferenceCellType) -> bool {
        reference_cell::is_simplex(cell)
    }
    unsafe fn vertices<T: RlstScalar<Real = T>>(cell: ReferenceCellType, vs: *mut T) {
        let mut i = 0;
        for v in reference_cell::vertices::<T>(cell) {
            for c in v {
                *vs.add(i) = c;
                i += 1;
            }
        }
    }
    #[no_mangle]
    pub unsafe extern "C" fn vertices_f32(cell: ReferenceCellType, vs: *mut f32) {
        vertices(cell, vs);
    }
    #[no_mangle]
    pub unsafe extern "C" fn vertices_f64(cell: ReferenceCellType, vs: *mut f64) {
        vertices(cell, vs);
    }
    unsafe fn midpoint<T: RlstScalar<Real = T>>(cell: ReferenceCellType, pt: *mut T) {
        for (i, c) in reference_cell::midpoint(cell).iter().enumerate() {
            *pt.add(i) = *c;
        }
    }
    #[no_mangle]
    pub unsafe extern "C" fn midpoint_f32(cell: ReferenceCellType, pt: *mut f32) {
        midpoint(cell, pt);
    }
    #[no_mangle]
    pub unsafe extern "C" fn midpoint_f64(cell: ReferenceCellType, pt: *mut f64) {
        midpoint(cell, pt);
    }
    #[no_mangle]
    pub unsafe extern "C" fn edges(cell: ReferenceCellType, es: *mut usize) {
        let mut i = 0;
        for e in reference_cell::edges(cell) {
            for v in e {
                *es.add(i) = v;
                i += 1
            }
        }
    }
    #[no_mangle]
    pub unsafe extern "C" fn faces(cell: ReferenceCellType, es: *mut usize) {
        let mut i = 0;
        for e in reference_cell::faces(cell) {
            for v in e {
                *es.add(i) = v;
                i += 1
            }
        }
    }
    #[no_mangle]
    pub unsafe extern "C" fn volumes(cell: ReferenceCellType, es: *mut usize) {
        let mut i = 0;
        for e in reference_cell::volumes(cell) {
            for v in e {
                *es.add(i) = v;
                i += 1
            }
        }
    }
    #[no_mangle]
    pub unsafe extern "C" fn entity_types(cell: ReferenceCellType, et: *mut u8) {
        let mut i = 0;
        for es in reference_cell::entity_types(cell) {
            for e in es {
                *et.add(i) = e as u8;
                i += 1
            }
        }
    }
    #[no_mangle]
    pub unsafe extern "C" fn entity_counts(cell: ReferenceCellType, ec: *mut usize) {
        for (i, e) in reference_cell::entity_counts(cell).iter().enumerate() {
            *ec.add(i) = *e;
        }
    }
    #[no_mangle]
    pub unsafe extern "C" fn connectivity_size(
        cell: ReferenceCellType,
        dim0: usize,
        index0: usize,
        dim1: usize,
    ) -> usize {
        reference_cell::connectivity(cell)[dim0][index0][dim1].len()
    }
    #[no_mangle]
    pub unsafe extern "C" fn connectivity(
        cell: ReferenceCellType,
        dim0: usize,
        index0: usize,
        dim1: usize,
        c: *mut usize,
    ) {
        for (i, j) in reference_cell::connectivity(cell)[dim0][index0][dim1]
            .iter()
            .enumerate()
        {
            *c.add(i) = *j;
        }
    }
}

pub mod quadrature {
    use crate::quadrature;
    use crate::types::ReferenceCellType;

    #[no_mangle]
    pub unsafe extern "C" fn gauss_jacobi_quadrature_npoints(
        cell: ReferenceCellType,
        m: usize,
    ) -> usize {
        quadrature::gauss_jacobi_npoints(cell, m)
    }
    #[no_mangle]
    pub unsafe extern "C" fn make_gauss_jacobi_quadrature(
        cell: ReferenceCellType,
        m: usize,
        pts: *mut f64,
        wts: *mut f64,
    ) {
        let rule = quadrature::gauss_jacobi_rule(cell, m).unwrap();
        for (i, p) in rule.points.iter().enumerate() {
            *pts.add(i) = *p;
        }
        for (i, w) in rule.weights.iter().enumerate() {
            *wts.add(i) = *w;
        }
    }
}

pub mod polynomials {
    use crate::types::ReferenceCellType;
    use crate::{polynomials, reference_cell};
    use rlst::{rlst_array_from_slice2, rlst_array_from_slice_mut3, RlstScalar};
    use std::slice::{from_raw_parts, from_raw_parts_mut};

    #[no_mangle]
    pub unsafe extern "C" fn legendre_polynomials_shape(
        cell: ReferenceCellType,
        npts: usize,
        degree: usize,
        derivatives: usize,
        shape: *mut usize,
    ) {
        *shape.add(0) = polynomials::derivative_count(cell, derivatives);
        *shape.add(1) = polynomials::polynomial_count(cell, degree);
        *shape.add(2) = npts;
    }
    unsafe fn tabulate_legendre_polynomials<T: RlstScalar>(
        cell: ReferenceCellType,
        points: *const T::Real,
        npts: usize,
        degree: usize,
        derivatives: usize,
        data: *mut T,
    ) {
        let tdim = reference_cell::dim(cell);
        let points = rlst_array_from_slice2!(from_raw_parts(points, npts * tdim), [tdim, npts]);
        let npoly = polynomials::polynomial_count(cell, degree);
        let nderiv = polynomials::derivative_count(cell, derivatives);
        let mut data = rlst_array_from_slice_mut3!(
            from_raw_parts_mut(data, npts * npoly * nderiv),
            [nderiv, npoly, npts]
        );
        polynomials::tabulate_legendre_polynomials(cell, &points, degree, derivatives, &mut data);
    }
    #[no_mangle]
    pub unsafe extern "C" fn tabulate_legendre_polynomials_f32(
        cell: ReferenceCellType,
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
        cell: ReferenceCellType,
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
    use crate::{
        ciarlet,
        ciarlet::CiarletElement,
        map::{ContravariantPiolaMap, CovariantPiolaMap, IdentityMap},
        reference_cell,
        traits::{ElementFamily, FiniteElement, Map},
        types::{Continuity, ReferenceCellType},
    };
    use c_api_tools::{cfuncs, concretise_types, DType, DTypeIdentifier};
    use rlst::{
        c32, c64, rlst_array_from_slice2, rlst_array_from_slice3, rlst_array_from_slice4,
        rlst_array_from_slice_mut4, MatrixInverse, RawAccess, RlstScalar, Shape,
    };
    use std::ffi::c_void;
    use std::slice::{from_raw_parts, from_raw_parts_mut};

    #[derive(Debug, PartialEq, Clone, Copy)]
    #[repr(u8)]
    pub enum ElementType {
        Lagrange = 0,
        RaviartThomas = 1,
        NedelecFirstKind = 2,
    }

    #[cfuncs(name = "ciarlet_element_t", create, free, unwrap)]
    pub struct CiarletElementT;

    #[cfuncs(name = "element_family_t", create, free, unwrap)]
    pub struct ElementFamilyT;

    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        gen_type(name = "maptype", replace_with = ["IdentityMap", "CovariantPiolaMap", "ContravariantPiolaMap"]),
        field(arg = 0, name = "element", wrapper = "CiarletElementT", replace_with = ["CiarletElement<{{dtype}}, {{maptype}}>"])
    )]
    pub fn element_value_size<E: FiniteElement>(element: &E) -> usize {
        element.value_size()
    }

    #[no_mangle]
    pub extern "C" fn create_lagrange_family(
        degree: usize,
        continuity: Continuity,
        dtype: DType,
    ) -> *mut ElementFamilyT {
        let family = element_family_t_create();
        let family_inner = unsafe { element_family_t_unwrap(family).unwrap() };

        *family_inner = match dtype {
            DType::F32 => Box::new(ciarlet::LagrangeElementFamily::<f32>::new(
                degree, continuity,
            )),
            DType::F64 => Box::new(ciarlet::LagrangeElementFamily::<f64>::new(
                degree, continuity,
            )),
            DType::C32 => Box::new(ciarlet::LagrangeElementFamily::<c32>::new(
                degree, continuity,
            )),
            DType::C64 => Box::new(ciarlet::LagrangeElementFamily::<c64>::new(
                degree, continuity,
            )),
            _ => panic!("Unsupported dtype"),
        };

        family
    }

    #[no_mangle]
    pub extern "C" fn create_raviart_thomas_family(
        degree: usize,
        continuity: Continuity,
        dtype: DType,
    ) -> *mut ElementFamilyT {
        let family = element_family_t_create();
        let family_inner = unsafe { element_family_t_unwrap(family).unwrap() };

        *family_inner = match dtype {
            DType::F32 => Box::new(ciarlet::RaviartThomasElementFamily::<f32>::new(
                degree, continuity,
            )),
            DType::F64 => Box::new(ciarlet::RaviartThomasElementFamily::<f64>::new(
                degree, continuity,
            )),
            DType::C32 => Box::new(ciarlet::RaviartThomasElementFamily::<c32>::new(
                degree, continuity,
            )),
            DType::C64 => Box::new(ciarlet::RaviartThomasElementFamily::<c64>::new(
                degree, continuity,
            )),
            _ => panic!("Unsupported dtype"),
        };

        family
    }

    #[no_mangle]
    pub extern "C" fn create_nedelec_family(
        degree: usize,
        continuity: Continuity,
        dtype: DType,
    ) -> *mut ElementFamilyT {
        let family = element_family_t_create();
        let family_inner = unsafe { element_family_t_unwrap(family).unwrap() };

        *family_inner = match dtype {
            DType::F32 => Box::new(ciarlet::NedelecFirstKindElementFamily::<f32>::new(
                degree, continuity,
            )),
            DType::F64 => Box::new(ciarlet::NedelecFirstKindElementFamily::<f64>::new(
                degree, continuity,
            )),
            DType::C32 => Box::new(ciarlet::NedelecFirstKindElementFamily::<c32>::new(
                degree, continuity,
            )),
            DType::C64 => Box::new(ciarlet::NedelecFirstKindElementFamily::<c64>::new(
                degree, continuity,
            )),
            _ => panic!("Unsupported dtype"),
        };

        family
    }

    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        field(arg = 0, name = "element_family", wrapper = "ElementFamilyT", replace_with = ["crate::ciarlet::LagrangeElementFamily<{{dtype}}>", "ciarlet::RaviartThomasElementFamily<{{dtype}}>", "ciarlet::NedelecFirstKindElementFamily<{{dtype}}>"])
    )]
    pub fn element_family_create_element<F: ElementFamily<CellType = ReferenceCellType>>(
        family: &F,
        cell: ReferenceCellType,
    ) -> *mut CiarletElementT {
        let ciarlet_element = ciarlet_element_t_create();
        let inner = unsafe { ciarlet_element_t_unwrap(ciarlet_element).unwrap() };

        *inner = Box::new(family.element(cell));

        ciarlet_element
    }

    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        field(arg = 0, name = "element_family", wrapper = "ElementFamilyT", replace_with = ["crate::ciarlet::LagrangeElementFamily<{{dtype}}>", "ciarlet::RaviartThomasElementFamily<{{dtype}}>", "ciarlet::NedelecFirstKindElementFamily<{{dtype}}>"])
    )]
    pub fn element_family_dtype<
        T: RlstScalar + MatrixInverse + DTypeIdentifier,
        F: ElementFamily<CellType = ReferenceCellType, T = T>,
    >(
        _elem: &F,
    ) -> DType {
        <T as DTypeIdentifier>::dtype()
    }

    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        gen_type(name = "maptype", replace_with = ["IdentityMap", "CovariantPiolaMap", "ContravariantPiolaMap"]),
        field(arg = 0, name = "element", wrapper = "CiarletElementT", replace_with = ["CiarletElement<{{dtype}}, {{maptype}}>"])
    )]
    pub fn ciarlet_element_dtype<
        T: RlstScalar + MatrixInverse + DTypeIdentifier,
        E: FiniteElement<T = T>,
    >(
        _elem: &E,
    ) -> DType {
        <T as DTypeIdentifier>::dtype()
    }

    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        gen_type(name = "maptype", replace_with = ["IdentityMap", "CovariantPiolaMap", "ContravariantPiolaMap"]),
        field(arg = 0, name = "element", wrapper = "CiarletElementT", replace_with = ["CiarletElement<{{dtype}}, {{maptype}}>"])
    )]
    pub fn ciarlet_element_tabulate_array_shape<E: FiniteElement>(
        element: &E,
        nderivs: usize,
        npoints: usize,
        shape: *mut usize,
    ) {
        for (i, j) in element
            .tabulate_array_shape(nderivs, npoints)
            .iter()
            .enumerate()
        {
            unsafe {
                *shape.add(i) = *j;
            }
        }
    }

    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        gen_type(name = "maptype", replace_with = ["IdentityMap", "CovariantPiolaMap", "ContravariantPiolaMap"]),
        field(arg = 0, name = "element", wrapper = "CiarletElementT", replace_with = ["CiarletElement<{{dtype}}, {{maptype}}>"])
    )]
    pub fn ciarlet_element_tabulate<E: FiniteElement<CellType = ReferenceCellType>>(
        element: &E,
        points: *const c_void,
        npoints: usize,
        nderivs: usize,
        data: *mut c_void,
    ) {
        let tdim = reference_cell::dim(element.cell_type());
        let points = points as *mut <E::T as RlstScalar>::Real;
        let data = data as *mut E::T;
        let points = rlst_array_from_slice2!(
            unsafe { from_raw_parts(points, npoints * tdim) },
            [tdim, npoints]
        );
        let shape = element.tabulate_array_shape(nderivs, npoints);
        let mut data = rlst_array_from_slice_mut4!(
            unsafe { from_raw_parts_mut(data, shape[0] * shape[1] * shape[2] * shape[3]) },
            shape
        );
        element.tabulate(&points, nderivs, &mut data);
    }

    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        gen_type(name = "maptype", replace_with = ["IdentityMap", "CovariantPiolaMap", "ContravariantPiolaMap"]),
        field(arg = 0, name = "element", wrapper = "CiarletElementT", replace_with = ["CiarletElement<{{dtype}}, {{maptype}}>"])
    )]
    pub fn ciarlet_element_value_size<E: FiniteElement>(element: &E) -> usize {
        element.value_size()
    }

    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        gen_type(name = "maptype", replace_with = ["IdentityMap", "CovariantPiolaMap", "ContravariantPiolaMap"]),
        field(arg = 0, name = "element", wrapper = "CiarletElementT", replace_with = ["CiarletElement<{{dtype}}, {{maptype}}>"])
    )]
    pub fn ciarlet_element_value_rank<E: FiniteElement>(element: &E) -> usize {
        element.value_shape().len()
    }

    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        gen_type(name = "maptype", replace_with = ["IdentityMap", "CovariantPiolaMap", "ContravariantPiolaMap"]),
        field(arg = 0, name = "element", wrapper = "CiarletElementT", replace_with = ["CiarletElement<{{dtype}}, {{maptype}}>"])
    )]
    pub fn ciarlet_element_value_shape<E: FiniteElement>(element: &E, shape: *mut usize) {
        for (i, j) in element.value_shape().iter().enumerate() {
            unsafe {
                *shape.add(i) = *j;
            }
        }
    }

    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        gen_type(name = "maptype", replace_with = ["IdentityMap", "CovariantPiolaMap", "ContravariantPiolaMap"]),
        field(arg = 0, name = "element", wrapper = "CiarletElementT", replace_with = ["CiarletElement<{{dtype}}, {{maptype}}>"])
    )]
    pub fn ciarlet_element_physical_value_size<E: FiniteElement>(
        element: &E,
        gdim: usize,
    ) -> usize {
        element.physical_value_size(gdim)
    }

    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        gen_type(name = "maptype", replace_with = ["IdentityMap", "CovariantPiolaMap", "ContravariantPiolaMap"]),
        field(arg = 0, name = "element", wrapper = "CiarletElementT", replace_with = ["CiarletElement<{{dtype}}, {{maptype}}>"])
    )]
    pub fn ciarlet_element_physical_value_rank<E: FiniteElement>(
        element: &E,
        gdim: usize,
    ) -> usize {
        element.physical_value_shape(gdim).len()
    }

    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        gen_type(name = "maptype", replace_with = ["IdentityMap", "CovariantPiolaMap", "ContravariantPiolaMap"]),
        field(arg = 0, name = "element", wrapper = "CiarletElementT", replace_with = ["CiarletElement<{{dtype}}, {{maptype}}>"])
    )]
    pub fn ciarlet_element_physical_value_shape<E: FiniteElement>(
        element: &E,
        gdim: usize,
        shape: *mut usize,
    ) {
        for (i, j) in element.physical_value_shape(gdim).iter().enumerate() {
            unsafe {
                *shape.add(i) = *j;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        gen_type(name = "maptype", replace_with = ["IdentityMap", "CovariantPiolaMap", "ContravariantPiolaMap"]),
        field(arg = 0, name = "element", wrapper = "CiarletElementT", replace_with = ["CiarletElement<{{dtype}}, {{maptype}}>"])
    )]
    pub fn ciarlet_element_push_forward<E: FiniteElement<CellType = ReferenceCellType>>(
        element: &E,
        npoints: usize,
        nfunctions: usize,
        gdim: usize,
        reference_values: *const c_void,
        nderivs: usize,
        j: *const c_void,
        jdet: *const c_void,
        jinv: *const c_void,
        physical_values: *mut c_void,
    ) {
        let tdim = reference_cell::dim(element.cell_type());
        let deriv_size = element.tabulate_array_shape(nderivs, npoints)[0];
        let pvs = element.physical_value_size(gdim);
        let vs = element.value_size();
        let reference_values = rlst_array_from_slice4!(
            unsafe {
                from_raw_parts(
                    reference_values as *const E::T,
                    deriv_size * npoints * nfunctions * vs,
                )
            },
            [deriv_size, npoints, nfunctions, vs]
        );
        let j = rlst_array_from_slice3!(
            unsafe {
                from_raw_parts(
                    j as *const <E::T as RlstScalar>::Real,
                    npoints * gdim * tdim,
                )
            },
            [npoints, gdim, tdim]
        );
        let jdet = unsafe { from_raw_parts(jdet as *const <E::T as RlstScalar>::Real, npoints) };
        let jinv = rlst_array_from_slice3!(
            unsafe {
                from_raw_parts(
                    jinv as *const <E::T as RlstScalar>::Real,
                    npoints * tdim * gdim,
                )
            },
            [npoints, tdim, gdim]
        );
        let mut physical_values = rlst_array_from_slice_mut4!(
            unsafe {
                from_raw_parts_mut(
                    physical_values as *mut E::T,
                    deriv_size * npoints * nfunctions * pvs,
                )
            },
            [deriv_size, npoints, nfunctions, pvs]
        );
        element.push_forward(
            &reference_values,
            nderivs,
            &j,
            jdet,
            &jinv,
            &mut physical_values,
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        gen_type(name = "maptype", replace_with = ["IdentityMap", "CovariantPiolaMap", "ContravariantPiolaMap"]),
        field(arg = 0, name = "element", wrapper = "CiarletElementT", replace_with = ["CiarletElement<{{dtype}}, {{maptype}}>"])
    )]
    pub fn ciarlet_element_pull_back<E: FiniteElement<CellType = ReferenceCellType>>(
        element: &E,
        npoints: usize,
        nfunctions: usize,
        gdim: usize,
        physical_values: *const c_void,
        nderivs: usize,
        j: *const c_void,
        jdet: *const c_void,
        jinv: *const c_void,
        reference_values: *mut c_void,
    ) {
        let tdim = reference_cell::dim(element.cell_type());
        let deriv_size = element.tabulate_array_shape(nderivs, npoints)[0];
        let pvs = element.physical_value_size(gdim);
        let vs = element.value_size();
        let physical_values = rlst_array_from_slice4!(
            unsafe {
                from_raw_parts(
                    physical_values as *const E::T,
                    deriv_size * npoints * nfunctions * pvs,
                )
            },
            [deriv_size, npoints, nfunctions, pvs]
        );
        let j = rlst_array_from_slice3!(
            unsafe {
                from_raw_parts(
                    j as *const <E::T as RlstScalar>::Real,
                    npoints * gdim * tdim,
                )
            },
            [npoints, gdim, tdim]
        );
        let jdet = unsafe { from_raw_parts(jdet as *const <E::T as RlstScalar>::Real, npoints) };
        let jinv = rlst_array_from_slice3!(
            unsafe {
                from_raw_parts(
                    jinv as *const <E::T as RlstScalar>::Real,
                    npoints * tdim * gdim,
                )
            },
            [npoints, tdim, gdim]
        );
        let mut reference_values = rlst_array_from_slice_mut4!(
            unsafe {
                from_raw_parts_mut(
                    reference_values as *mut E::T,
                    deriv_size * npoints * nfunctions * vs,
                )
            },
            [deriv_size, npoints, nfunctions, vs]
        );
        element.pull_back(
            &physical_values,
            nderivs,
            &j,
            jdet,
            &jinv,
            &mut reference_values,
        );
    }

    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        gen_type(name = "maptype", replace_with = ["IdentityMap", "CovariantPiolaMap", "ContravariantPiolaMap"]),
        field(arg = 0, name = "element", wrapper = "CiarletElementT", replace_with = ["CiarletElement<{{dtype}}, {{maptype}}>"])
    )]
    pub fn ciarlet_element_degree<T: RlstScalar + MatrixInverse + DTypeIdentifier, M: Map>(
        element: &CiarletElement<T, M>,
    ) -> usize {
        element.degree()
    }

    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        gen_type(name = "maptype", replace_with = ["IdentityMap", "CovariantPiolaMap", "ContravariantPiolaMap"]),
        field(arg = 0, name = "element", wrapper = "CiarletElementT", replace_with = ["CiarletElement<{{dtype}}, {{maptype}}>"])
    )]
    pub fn ciarlet_element_embedded_superdegree<E: FiniteElement>(element: &E) -> usize {
        element.embedded_superdegree()
    }

    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        gen_type(name = "maptype", replace_with = ["IdentityMap", "CovariantPiolaMap", "ContravariantPiolaMap"]),
        field(arg = 0, name = "element", wrapper = "CiarletElementT", replace_with = ["CiarletElement<{{dtype}}, {{maptype}}>"])
    )]
    pub fn ciarlet_element_dim<E: FiniteElement>(element: &E) -> usize {
        element.dim()
    }

    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        gen_type(name = "maptype", replace_with = ["IdentityMap", "CovariantPiolaMap", "ContravariantPiolaMap"]),
        field(arg = 0, name = "element", wrapper = "CiarletElementT", replace_with = ["CiarletElement<{{dtype}}, {{maptype}}>"])
    )]
    pub fn ciarlet_element_continuity<T: RlstScalar + MatrixInverse + DTypeIdentifier, M: Map>(
        element: &CiarletElement<T, M>,
    ) -> Continuity {
        element.continuity()
    }

    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        gen_type(name = "maptype", replace_with = ["IdentityMap", "CovariantPiolaMap", "ContravariantPiolaMap"]),
        field(arg = 0, name = "element", wrapper = "CiarletElementT", replace_with = ["CiarletElement<{{dtype}}, {{maptype}}>"])
    )]
    pub fn ciarlet_element_cell_type<E: FiniteElement<CellType = ReferenceCellType>>(
        element: &E,
    ) -> ReferenceCellType {
        element.cell_type()
    }

    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        gen_type(name = "maptype", replace_with = ["IdentityMap", "CovariantPiolaMap", "ContravariantPiolaMap"]),
        field(arg = 0, name = "element", wrapper = "CiarletElementT", replace_with = ["CiarletElement<{{dtype}}, {{maptype}}>"])
    )]
    pub fn ciarlet_element_entity_dofs_size<E: FiniteElement>(
        element: &E,
        entity_dim: usize,
        entity_index: usize,
    ) -> usize {
        element.entity_dofs(entity_dim, entity_index).unwrap().len()
    }

    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        gen_type(name = "maptype", replace_with = ["IdentityMap", "CovariantPiolaMap", "ContravariantPiolaMap"]),
        field(arg = 0, name = "element", wrapper = "CiarletElementT", replace_with = ["CiarletElement<{{dtype}}, {{maptype}}>"])
    )]
    pub fn ciarlet_element_entity_dofs<E: FiniteElement>(
        element: &E,
        entity_dim: usize,
        entity_index: usize,
        entity_dofs: *mut usize,
    ) {
        for (i, dof) in element
            .entity_dofs(entity_dim, entity_index)
            .unwrap()
            .iter()
            .enumerate()
        {
            unsafe {
                *entity_dofs.add(i) = *dof;
            }
        }
    }

    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        gen_type(name = "maptype", replace_with = ["IdentityMap", "CovariantPiolaMap", "ContravariantPiolaMap"]),
        field(arg = 0, name = "element", wrapper = "CiarletElementT", replace_with = ["CiarletElement<{{dtype}}, {{maptype}}>"])
    )]
    pub fn ciarlet_element_entity_closure_dofs_size<
        T: RlstScalar + MatrixInverse + DTypeIdentifier,
        M: Map,
    >(
        element: &CiarletElement<T, M>,
        entity_dim: usize,
        entity_index: usize,
    ) -> usize {
        element
            .entity_closure_dofs(entity_dim, entity_index)
            .unwrap()
            .len()
    }

    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        gen_type(name = "maptype", replace_with = ["IdentityMap", "CovariantPiolaMap", "ContravariantPiolaMap"]),
        field(arg = 0, name = "element", wrapper = "CiarletElementT", replace_with = ["CiarletElement<{{dtype}}, {{maptype}}>"])
    )]
    pub fn ciarlet_element_entity_closure_dofs<
        T: RlstScalar + MatrixInverse + DTypeIdentifier,
        M: Map,
    >(
        element: &CiarletElement<T, M>,
        entity_dim: usize,
        entity_index: usize,
        entity_closure_dofs: *mut usize,
    ) {
        for (i, dof) in element
            .entity_closure_dofs(entity_dim, entity_index)
            .unwrap()
            .iter()
            .enumerate()
        {
            unsafe {
                *entity_closure_dofs.add(i) = *dof;
            }
        }
    }

    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        gen_type(name = "maptype", replace_with = ["IdentityMap", "CovariantPiolaMap", "ContravariantPiolaMap"]),
        field(arg = 0, name = "element", wrapper = "CiarletElementT", replace_with = ["CiarletElement<{{dtype}}, {{maptype}}>"])
    )]
    pub fn ciarlet_element_interpolation_npoints<
        T: RlstScalar + MatrixInverse + DTypeIdentifier,
        M: Map,
    >(
        element: &CiarletElement<T, M>,
        entity_dim: usize,
        entity_index: usize,
    ) -> usize {
        element.interpolation_points()[entity_dim][entity_index].shape()[1]
    }

    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        gen_type(name = "maptype", replace_with = ["IdentityMap", "CovariantPiolaMap", "ContravariantPiolaMap"]),
        field(arg = 0, name = "element", wrapper = "CiarletElementT", replace_with = ["CiarletElement<{{dtype}}, {{maptype}}>"])
    )]
    pub fn ciarlet_element_interpolation_ndofs<
        T: RlstScalar + MatrixInverse + DTypeIdentifier,
        M: Map,
    >(
        element: &CiarletElement<T, M>,
        entity_dim: usize,
        entity_index: usize,
    ) -> usize {
        element.interpolation_weights()[entity_dim][entity_index].shape()[0]
    }

    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        gen_type(name = "maptype", replace_with = ["IdentityMap", "CovariantPiolaMap", "ContravariantPiolaMap"]),
        field(arg = 0, name = "element", wrapper = "CiarletElementT", replace_with = ["CiarletElement<{{dtype}}, {{maptype}}>"])
    )]
    pub fn ciarlet_element_interpolation_points<
        T: RlstScalar + MatrixInverse + DTypeIdentifier,
        M: Map,
    >(
        element: &CiarletElement<T, M>,
        entity_dim: usize,
        entity_index: usize,
        points: *mut c_void,
    ) {
        let points = points as *mut T::Real;
        for (i, j) in element.interpolation_points()[entity_dim][entity_index]
            .data()
            .iter()
            .enumerate()
        {
            unsafe {
                *points.add(i) = *j;
            }
        }
    }

    #[concretise_types(
        gen_type(name = "dtype", replace_with = ["f32", "f64", "c32", "c64"]),
        gen_type(name = "maptype", replace_with = ["IdentityMap", "CovariantPiolaMap", "ContravariantPiolaMap"]),
        field(arg = 0, name = "element", wrapper = "CiarletElementT", replace_with = ["CiarletElement<{{dtype}}, {{maptype}}>"])
    )]
    pub fn ciarlet_element_interpolation_weights<
        T: RlstScalar + MatrixInverse + DTypeIdentifier,
        M: Map,
    >(
        element: &CiarletElement<T, M>,
        entity_dim: usize,
        entity_index: usize,
        weights: *mut c_void,
    ) {
        let weights = weights as *mut T;
        for (i, j) in element.interpolation_weights()[entity_dim][entity_index]
            .data()
            .iter()
            .enumerate()
        {
            unsafe {
                *weights.add(i) = *j;
            }
        }
    }
}
