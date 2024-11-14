//! Simplex rules
use crate::types::ReferenceCellType;
use bempp_quadrature::{
    simplex_rules,
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
pub fn available_rules(cell: ReferenceCellType) -> Vec<usize> {
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
