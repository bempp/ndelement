//! Binding for C
#![allow(missing_docs)]

pub mod reference_cell {
    use crate::types::ReferenceCellType;
    use crate::reference_cell;
    use rlst::RlstScalar;

    #[no_mangle]
    pub unsafe extern "C" fn dim(cell: u8) -> usize {
        reference_cell::dim(ReferenceCellType::from(cell).expect("Invalid cell type"))
    }
    #[no_mangle]
    pub unsafe extern "C" fn is_simplex(cell: u8) -> bool {
        reference_cell::is_simplex(ReferenceCellType::from(cell).expect("Invalid cell type"))
    }
    unsafe fn vertices<T: RlstScalar<Real=T>>(cell: u8, vs: *mut T) {
        let mut i = 0;
        for v in reference_cell::vertices::<T>(ReferenceCellType::from(cell).expect("Invalid cell type")) {
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
    unsafe fn midpoint<T: RlstScalar<Real=T>>(cell: u8, pt: *mut T) {
        let pt = pt as *mut T;
        for (i, c) in reference_cell::midpoint(ReferenceCellType::from(cell).expect("Invalid cell type")).iter().enumerate() {
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
        for e in reference_cell::volumes(ReferenceCellType::from(cell).expect("Invalid cell type")) {
            for v in e {
                *es.add(i) = v;
                i += 1
            }
        }
    }
    #[no_mangle]
    pub unsafe extern "C" fn entity_types(cell: u8, et: *mut u8) {
        let mut i = 0;
        for es in reference_cell::entity_types(ReferenceCellType::from(cell).expect("Invalid cell type")) {
            for e in es {
                *et.add(i) = e as u8;
                i += 1
            }
        }
    }
    #[no_mangle]
    pub unsafe extern "C" fn entity_counts(cell: u8, ec: *mut usize) {
        for (i, e) in reference_cell::entity_counts(ReferenceCellType::from(cell).expect("Invalid cell type")).iter().enumerate() {
            *ec.add(i) = *e;
        }
    }
    pub unsafe extern "C" fn connetivity_size(cell: u8, dim0: usize, index0: usize, dim1: usize) -> usize {
        reference_cell::connectivity(ReferenceCellType::from(cell).expect("Invalid cell type"))[dim0][index0][dim1].len()
    }
    #[no_mangle]
    pub unsafe extern "C" fn connetivity(cell: u8, dim0: usize, index0: usize, dim1: usize, c: *mut usize) {
        for (i, j) in reference_cell::connectivity(ReferenceCellType::from(cell).expect("Invalid cell type"))[dim0][index0][dim1].iter().enumerate() {
            *c.add(i) = *j;
        }
    }
}
