//! Quadrature
use crate::types::ReferenceCellType;
use bempp_quadrature::{
    gauss_jacobi, simplex_rules,
    types::{NumericalQuadratureDefinition, QuadratureError},
};

/// Return a simplex rule for a given number of points.
///
/// If the rule does not exist `Err(())` is returned.
pub fn simplex_rule(
    cell: ReferenceCellType,
    npoints: usize,
) -> Result<NumericalQuadratureDefinition, QuadratureError> {
    match cell {
        ReferenceCellType::Point => {
            if npoints == 1 {
                Ok(NumericalQuadratureDefinition {
                    dim: 0,
                    order: 1,
                    npoints,
                    weights: vec![1.0],
                    points: vec![],
                })
            } else {
                Err(QuadratureError::RuleNotFound)
            }
        }
        ReferenceCellType::Interval => simplex_rules::simplex_rule_interval(npoints),
        ReferenceCellType::Triangle => simplex_rules::simplex_rule_triangle(npoints),
        ReferenceCellType::Quadrilateral => simplex_rules::simplex_rule_quadrilateral(npoints),
        ReferenceCellType::Tetrahedron => simplex_rules::simplex_rule_tetrahedron(npoints),
        ReferenceCellType::Hexahedron => simplex_rules::simplex_rule_hexahedron(npoints),
        ReferenceCellType::Prism => simplex_rules::simplex_rule_prism(npoints),
        ReferenceCellType::Pyramid => simplex_rules::simplex_rule_pyramid(npoints),
    }
}

/// For a given cell type return a vector with the numbers of points for which simplex rules are available.
pub fn available_simplex_rules(cell: ReferenceCellType) -> Vec<usize> {
    match cell {
        ReferenceCellType::Point => vec![1],
        ReferenceCellType::Interval => simplex_rules::available_rules_interval(),
        ReferenceCellType::Triangle => simplex_rules::available_rules_triangle(),
        ReferenceCellType::Quadrilateral => simplex_rules::available_rules_quadrilateral(),
        ReferenceCellType::Tetrahedron => simplex_rules::available_rules_tetrahedron(),
        ReferenceCellType::Hexahedron => simplex_rules::available_rules_hexahedron(),
        ReferenceCellType::Prism => simplex_rules::available_rules_prism(),
        ReferenceCellType::Pyramid => simplex_rules::available_rules_pyramid(),
    }
}

/// Get the points and weights for a Gauss-Jacobi quadrature rule
pub fn gauss_jacobi_rule(
    celltype: ReferenceCellType,
    m: usize,
) -> Result<NumericalQuadratureDefinition, QuadratureError> {
    let np = (m + 2) / 2;
    match celltype {
        ReferenceCellType::Interval => Ok(gauss_jacobi::gauss_jacobi_interval(np)),
        ReferenceCellType::Triangle => Ok(gauss_jacobi::gauss_jacobi_triangle(np)),
        ReferenceCellType::Quadrilateral => Ok(gauss_jacobi::gauss_jacobi_quadrilateral(np)),
        ReferenceCellType::Tetrahedron => Ok(gauss_jacobi::gauss_jacobi_tetrahedron(np)),
        ReferenceCellType::Hexahedron => Ok(gauss_jacobi::gauss_jacobi_hexahedron(np)),
        _ => Err(QuadratureError::RuleNotFound),
    }
}

/// Get the number of quadrature points for a Gauss-Jacobi rule
pub fn gauss_jacobi_npoints(celltype: ReferenceCellType, m: usize) -> usize {
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
