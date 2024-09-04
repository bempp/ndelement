//! Binding for C
#![allow(missing_docs)]

pub mod reference_cell {
    use crate::types::ReferenceCellType;

    #[no_mangle]
    pub unsafe extern "C" fn dim(cell: u8) -> usize {
        crate::reference_cell::dim(ReferenceCellType::from(cell).expect("Invalid cell type"))
    }
    #[no_mangle]
    pub unsafe extern "C" fn is_simplex(cell: u8) -> bool {
        crate::reference_cell::is_simplex(ReferenceCellType::from(cell).expect("Invalid cell type"))
    }
}
