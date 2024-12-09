//! Finite element definitions

use ndelement::ciarlet::{lagrange, nedelec, raviart_thomas};
use ndelement::types::{Continuity, ReferenceCellType};
use paste::paste;

fn main() {
    macro_rules! construct_lagrange {
        ($cell:ident, $max_degree:expr) => {
            paste! {
                for d in 1..[<$max_degree>] {
                    println!("Constructing Lagrange(degree={d}, cell={:?})", ReferenceCellType::[<$cell>]);
                    let _e = lagrange::create::<f64>(ReferenceCellType::[<$cell>], d, Continuity::Standard);
                }
            }
        };
    }

    macro_rules! construct_raviart_thomas {
        ($cell:ident, $max_degree:expr) => {
            paste! {
                for d in 1..[<$max_degree>] {
                    println!("Constructing RaviartThomas(degree={d}, cell={:?})", ReferenceCellType::[<$cell>]);
                    let _e = raviart_thomas::create::<f64>(ReferenceCellType::[<$cell>], d, Continuity::Standard);
                }
            }
        };
    }

    macro_rules! construct_nedelec {
        ($cell:ident, $max_degree:expr) => {
            paste! {
                for d in 1..[<$max_degree>] {
                    println!("Constructing Nedelec(degree={d}, cell={:?})", ReferenceCellType::[<$cell>]);
                    let _e = nedelec::create::<f64>(ReferenceCellType::[<$cell>], d, Continuity::Standard);
                }
            }
        };
    }

    construct_lagrange!(Interval, 14);
    construct_lagrange!(Triangle, 8);
    construct_lagrange!(Quadrilateral, 8);
    construct_lagrange!(Tetrahedron, 6);
    construct_lagrange!(Hexahedron, 6);

    construct_raviart_thomas!(Triangle, 8);
    construct_raviart_thomas!(Quadrilateral, 8);
    construct_raviart_thomas!(Tetrahedron, 6);
    construct_raviart_thomas!(Hexahedron, 6);

    construct_nedelec!(Triangle, 8);
    construct_nedelec!(Quadrilateral, 8);
    construct_nedelec!(Tetrahedron, 6);
    construct_nedelec!(Hexahedron, 6);
}
