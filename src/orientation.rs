//! Cell orientation.

use crate::{reference_cell, types::ReferenceCellType};
use itertools::izip;

/// Compute a 32-bit integer that encodes the orientation differences between the reference cell and a cell with vertices numbered as input
///
/// From right to left, the bits of this function's output encode:
///  - 1 bit for each edge - set to 1 if the edge needs reversing
///  - 3 bits for each face - two rightmost bits encode number of rotations that need to be applied to face; left bit encodes whether it needs reflecting
pub fn compute_orientation(entity_type: ReferenceCellType, vertices: &[usize]) -> i32 {
    let mut orientation = 0;
    let mut n = 0;
    let dim = reference_cell::dim(entity_type);
    if dim > 1 {
        for v in reference_cell::edges(entity_type) {
            if vertices[v[0]] > vertices[v[1]] {
                orientation += 1 << n;
            }
            n += 1;
        }
    }
    if dim > 2 {
        for (t, v) in izip!(
            &reference_cell::entity_types(entity_type)[2],
            reference_cell::faces(entity_type)
        ) {
            match t {
                ReferenceCellType::Triangle => {
                    let m = v
                        .iter()
                        .map(|i| vertices[*i])
                        .enumerate()
                        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(index, _)| index)
                        .unwrap() as i32;
                    orientation += m << n;
                    n += 2;
                    let next = vertices[v[(m as usize + 1) % 3]];
                    let prev = vertices[v[if m == 0 { 2 } else { m as usize - 1 }]];
                    if next > prev {
                        orientation += 1 << n;
                    }
                    n += 1;
                }
                ReferenceCellType::Quadrilateral => {
                    let m = v
                        .iter()
                        .map(|i| vertices[*i])
                        .enumerate()
                        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(index, _)| index)
                        .unwrap() as i32;
                    if m == 2 {
                        orientation += 3 << n;
                    } else if m == 3 {
                        orientation += 2 << n;
                    } else {
                        orientation += m << n;
                    }
                    n += 2;
                    let next = vertices[v[[1, 3, 0, 2][m as usize]]];
                    let prev = vertices[v[[2, 0, 3, 1][m as usize]]];
                    if next > prev {
                        orientation += 1 << n;
                    }
                    n += 1;
                }
                _ => {
                    panic!("Unsupported face type: {t:?}");
                }
            }
        }
    }
    orientation
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_compute_orientations_interval() {
        for (v, o) in [([0, 1], 0), ([1, 0], 0)] {
            assert_eq!(compute_orientation(ReferenceCellType::Interval, &v), o);
        }
    }

    #[test]
    fn test_compute_orientations_triangle() {
        for (v, o) in [
            ([0, 1, 2], 0),
            ([0, 2, 1], 1),
            ([1, 0, 2], 4),
            ([1, 2, 0], 3),
            ([2, 0, 1], 6),
            ([2, 1, 0], 7),
        ] {
            assert_eq!(compute_orientation(ReferenceCellType::Triangle, &v), o);
        }
    }

    #[test]
    fn test_compute_orientations_quadrilateral() {
        for (v, o) in [
            ([0, 1, 2, 3], 0),
            ([0, 1, 3, 2], 8),
            ([0, 2, 1, 3], 0),
            ([0, 2, 3, 1], 12),
            ([0, 3, 1, 2], 4),
            ([0, 3, 2, 1], 12),
            ([1, 0, 2, 3], 1),
            ([1, 0, 3, 2], 9),
            ([1, 2, 0, 3], 2),
            ([1, 2, 3, 0], 12),
            ([1, 3, 0, 2], 6),
            ([1, 3, 2, 0], 12),
        ] {
            assert_eq!(compute_orientation(ReferenceCellType::Quadrilateral, &v), o);
        }
    }

    #[test]
    fn test_compute_orientations_tetrahedron() {
        for (v, o) in [
            ([0, 1, 2, 3], 0),
            ([0, 3, 2, 1], 149895), // 100 100 100 110 000111
            ([1, 2, 0, 3], 68436),  // 010 000 101 101 010100
            ([1, 3, 2, 0], 140687), // 100 010 010 110 001111
        ] {
            assert_eq!(compute_orientation(ReferenceCellType::Tetrahedron, &v), o);
        }
    }

    #[test]
    fn test_compute_orientations_hexahedron() {
        for (v, o) in [
            ([0, 1, 2, 3, 4, 5, 6, 7], 0),
            ([2, 5, 6, 1, 7, 0, 4, 3], 165899128), // 001 001 111 000 110 110 101101111000
            ([4, 1, 5, 0, 2, 3, 6, 7], 95215661),  // 000 101 101 011 001 110 000000101101
            ([6, 5, 1, 7, 2, 4, 3, 0], 306691223), // 010 010 010 001 111 011 110010010111
        ] {
            assert_eq!(compute_orientation(ReferenceCellType::Hexahedron, &v), o);
        }
    }
}
