use ndelement::ciarlet::LagrangeElementFamily;
use ndelement::traits::{ElementFamily, FiniteElement};
use ndelement::types::{Continuity, ReferenceCellType};

fn main() {
    // Create the degree 2 Lagrange element family. A family is a set of finite elements with the
    // same family type, degree, and continuity across a set of cells
    let family = LagrangeElementFamily::<f64>::new(2, Continuity::Standard);

    // Get the element in the family on a triangle
    let element = family.element(ReferenceCellType::Triangle);
    println!("Cell: {:?}", element.cell_type());

    // Get the element in the family on a triangle
    let element = family.element(ReferenceCellType::Quadrilateral);
    println!("Cell: {:?}", element.cell_type());
}
