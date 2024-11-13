use bempp_quadrature::{types::{NumericalQuadratureDefinition, QuadratureError}, simplex_rules::simplex_rule};
use crate::types::ReferenceCellType;

/// Return a simplex rule that integrates polynomials up to the given degree exactly
pub(crate) fn simplex_rule_by_degree(
    cell_type: ReferenceCellType,
    degree: usize,
) -> Result<NumericalQuadratureDefinition, QuadratureError> {
    simplex_rule(cell_type,
    match cell_type {
        ReferenceCellType::Point => 1,
        ReferenceCellType::Interval => degree + 1,
        ReferenceCellType::Triangle => (degree + 1) * (degree + 2) / 2,
        ReferenceCellType::Quadrilateral => (degree + 1).pow(2),
        ReferenceCellType::Tetrahedron => (degree + 1) * (degree + 2) * (degree + 3) / 6,
        ReferenceCellType::Hexahedron => (degree + 1).pow(3),
        ReferenceCellType::Prism => (degree + 1).pow(2) * (degree + 2) / 2,
        ReferenceCellType::Pyramid => (degree + 1) * (degree + 2) * (2 * degree + 3) / 6,
    }
    )
}
