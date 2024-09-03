//! Lagrange elements

use super::CiarletElement;
use crate::polynomials::polynomial_count;
use crate::reference_cell;
use crate::traits::ElementFamily;
use crate::types::{Continuity, MapType, ReferenceCellType};
use rlst::{rlst_dynamic_array2, rlst_dynamic_array3, MatrixInverse, RandomAccessMut, RlstScalar};
use std::marker::PhantomData;

/// Create a Lagrange element
pub fn create<T: RlstScalar + MatrixInverse>(
    cell_type: ReferenceCellType,
    degree: usize,
    continuity: Continuity,
) -> CiarletElement<T> {
    let dim = polynomial_count(cell_type, degree);
    let tdim = reference_cell::dim(cell_type);
    let mut wcoeffs = rlst_dynamic_array3!(T, [dim, 1, dim]);
    for i in 0..dim {
        *wcoeffs.get_mut([i, 0, i]).unwrap() = T::from(1.0).unwrap();
    }

    let mut x = [vec![], vec![], vec![], vec![]];
    let mut m = [vec![], vec![], vec![], vec![]];
    let entity_counts = reference_cell::entity_counts(cell_type);
    let vertices = reference_cell::vertices::<T::Real>(cell_type);
    if degree == 0 {
        if continuity == Continuity::Standard {
            panic!("Cannot create continuous degree 0 Lagrange element");
        }
        for (d, counts) in entity_counts.iter().enumerate() {
            for _e in 0..*counts {
                x[d].push(rlst_dynamic_array2!(T::Real, [tdim, 0]));
                m[d].push(rlst_dynamic_array3!(T, [0, 1, 0]));
            }
        }
        let mut midp = rlst_dynamic_array2!(T::Real, [tdim, 1]);
        let nvertices = entity_counts[0];
        for i in 0..tdim {
            for vertex in &vertices {
                *midp.get_mut([i, 0]).unwrap() += num::cast::<_, T::Real>(vertex[i]).unwrap();
            }
            *midp.get_mut([i, 0]).unwrap() /= num::cast::<_, T::Real>(nvertices).unwrap();
        }
        x[tdim].push(midp);
        let mut mentry = rlst_dynamic_array3!(T, [1, 1, 1]);
        *mentry.get_mut([0, 0, 0]).unwrap() = T::from(1.0).unwrap();
        m[tdim].push(mentry);
    } else {
        let edges = reference_cell::edges(cell_type);
        let faces = reference_cell::faces(cell_type);
        let volumes = reference_cell::volumes(cell_type);
        // TODO: GLL points
        for vertex in &vertices {
            let mut pts = rlst_dynamic_array2!(T::Real, [tdim, 1]);
            for (i, v) in vertex.iter().enumerate() {
                *pts.get_mut([i, 0]).unwrap() = num::cast::<_, T::Real>(*v).unwrap();
            }
            x[0].push(pts);
            let mut mentry = rlst_dynamic_array3!(T, [1, 1, 1]);
            *mentry.get_mut([0, 0, 0]).unwrap() = T::from(1.0).unwrap();
            m[0].push(mentry);
        }
        for e in &edges {
            let mut pts = rlst_dynamic_array2!(T::Real, [tdim, degree - 1]);
            let [vn0, vn1] = e[..] else {
                panic!();
            };
            let v0 = &vertices[vn0];
            let v1 = &vertices[vn1];
            let mut ident = rlst_dynamic_array3!(T, [degree - 1, 1, degree - 1]);

            for i in 1..degree {
                *ident.get_mut([i - 1, 0, i - 1]).unwrap() = T::from(1.0).unwrap();
                for j in 0..tdim {
                    *pts.get_mut([j, i - 1]).unwrap() = num::cast::<_, T::Real>(v0[j]).unwrap()
                        + num::cast::<_, T::Real>(i).unwrap()
                            / num::cast::<_, T::Real>(degree).unwrap()
                            * num::cast::<_, T::Real>(v1[j] - v0[j]).unwrap();
                }
            }
            x[1].push(pts);
            m[1].push(ident);
        }
        for (e, face_type) in reference_cell::entity_types(cell_type)[2]
            .iter()
            .enumerate()
        {
            let npts = match face_type {
                ReferenceCellType::Triangle => {
                    if degree > 2 {
                        (degree - 1) * (degree - 2) / 2
                    } else {
                        0
                    }
                }
                ReferenceCellType::Quadrilateral => (degree - 1).pow(2),
                _ => {
                    panic!("Unsupported face type");
                }
            };
            let mut pts = rlst_dynamic_array2!(T::Real, [tdim, npts]);

            let [vn0, vn1, vn2] = faces[e][..3] else {
                panic!();
            };
            let v0 = &vertices[vn0];
            let v1 = &vertices[vn1];
            let v2 = &vertices[vn2];

            match face_type {
                ReferenceCellType::Triangle => {
                    let mut n = 0;
                    for i0 in 1..degree {
                        for i1 in 1..degree - i0 {
                            for j in 0..tdim {
                                *pts.get_mut([j, n]).unwrap() = num::cast::<_, T::Real>(v0[j])
                                    .unwrap()
                                    + num::cast::<_, T::Real>(i0).unwrap()
                                        / num::cast::<_, T::Real>(degree).unwrap()
                                        * num::cast::<_, T::Real>(v1[j] - v0[j]).unwrap()
                                    + num::cast::<_, T::Real>(i1).unwrap()
                                        / num::cast::<_, T::Real>(degree).unwrap()
                                        * num::cast::<_, T::Real>(v2[j] - v0[j]).unwrap();
                            }
                            n += 1;
                        }
                    }
                }
                ReferenceCellType::Quadrilateral => {
                    let mut n = 0;
                    for i0 in 1..degree {
                        for i1 in 1..degree {
                            for j in 0..tdim {
                                *pts.get_mut([j, n]).unwrap() = num::cast::<_, T::Real>(v0[j])
                                    .unwrap()
                                    + num::cast::<_, T::Real>(i0).unwrap()
                                        / num::cast::<_, T::Real>(degree).unwrap()
                                        * num::cast::<_, T::Real>(v1[j] - v0[j]).unwrap()
                                    + num::cast::<_, T::Real>(i1).unwrap()
                                        / num::cast::<_, T::Real>(degree).unwrap()
                                        * num::cast::<_, T::Real>(v2[j] - v0[j]).unwrap();
                            }
                            n += 1;
                        }
                    }
                }
                _ => {
                    panic!("Unsupported face type.");
                }
            };

            let mut ident = rlst_dynamic_array3!(T, [npts, 1, npts]);
            for i in 0..npts {
                *ident.get_mut([i, 0, i]).unwrap() = T::from(1.0).unwrap();
            }
            x[2].push(pts);
            m[2].push(ident);
        }
        for (e, volume_type) in reference_cell::entity_types(cell_type)[3]
            .iter()
            .enumerate()
        {
            let npts = match volume_type {
                ReferenceCellType::Tetrahedron => {
                    if degree > 2 {
                        (degree - 1) * (degree - 2) * (degree - 3) / 6
                    } else {
                        0
                    }
                }
                ReferenceCellType::Hexahedron => (degree - 1).pow(3),
                _ => {
                    panic!("Unsupported face type");
                }
            };
            let mut pts = rlst_dynamic_array2!(T::Real, [tdim, npts]);

            match volume_type {
                ReferenceCellType::Tetrahedron => {
                    let [vn0, vn1, vn2, vn3] = volumes[e][..4] else {
                        panic!();
                    };
                    let v0 = &vertices[vn0];
                    let v1 = &vertices[vn1];
                    let v2 = &vertices[vn2];
                    let v3 = &vertices[vn3];

                    let mut n = 0;
                    for i0 in 1..degree {
                        for i1 in 1..degree - i0 {
                            for i2 in 1..degree - i0 - i1 {
                                for j in 0..tdim {
                                    *pts.get_mut([j, n]).unwrap() = num::cast::<_, T::Real>(v0[j])
                                        .unwrap()
                                        + num::cast::<_, T::Real>(i0).unwrap()
                                            / num::cast::<_, T::Real>(degree).unwrap()
                                            * num::cast::<_, T::Real>(v1[j] - v0[j]).unwrap()
                                        + num::cast::<_, T::Real>(i1).unwrap()
                                            / num::cast::<_, T::Real>(degree).unwrap()
                                            * num::cast::<_, T::Real>(v2[j] - v0[j]).unwrap()
                                        + num::cast::<_, T::Real>(i2).unwrap()
                                            / num::cast::<_, T::Real>(degree).unwrap()
                                            * num::cast::<_, T::Real>(v3[j] - v0[j]).unwrap();
                                }
                                n += 1;
                            }
                        }
                    }
                }
                ReferenceCellType::Hexahedron => {
                    let [vn0, vn1, vn2, _, vn3] = volumes[e][..5] else {
                        panic!();
                    };
                    let v0 = &vertices[vn0];
                    let v1 = &vertices[vn1];
                    let v2 = &vertices[vn2];
                    let v3 = &vertices[vn3];

                    let mut n = 0;
                    for i0 in 1..degree {
                        for i1 in 1..degree {
                            for i2 in 1..degree {
                                for j in 0..tdim {
                                    *pts.get_mut([j, n]).unwrap() = num::cast::<_, T::Real>(v0[j])
                                        .unwrap()
                                        + num::cast::<_, T::Real>(i0).unwrap()
                                            / num::cast::<_, T::Real>(degree).unwrap()
                                            * num::cast::<_, T::Real>(v1[j] - v0[j]).unwrap()
                                        + num::cast::<_, T::Real>(i1).unwrap()
                                            / num::cast::<_, T::Real>(degree).unwrap()
                                            * num::cast::<_, T::Real>(v2[j] - v0[j]).unwrap()
                                        + num::cast::<_, T::Real>(i2).unwrap()
                                            / num::cast::<_, T::Real>(degree).unwrap()
                                            * num::cast::<_, T::Real>(v3[j] - v0[j]).unwrap();
                                }
                                n += 1;
                            }
                        }
                    }
                }
                _ => {
                    panic!("Unsupported face type.");
                }
            };

            let mut ident = rlst_dynamic_array3!(T, [npts, 1, npts]);
            for i in 0..npts {
                *ident.get_mut([i, 0, i]).unwrap() = T::from(1.0).unwrap();
            }
            x[3].push(pts);
            m[3].push(ident);
        }
    }
    CiarletElement::<T>::create(
        "Lagrange".to_string(),
        cell_type,
        degree,
        vec![],
        wcoeffs,
        x,
        m,
        MapType::Identity,
        continuity,
        degree,
    )
}

/// Lagrange element family
pub struct LagrangeElementFamily<T: RlstScalar + MatrixInverse> {
    degree: usize,
    continuity: Continuity,
    _t: PhantomData<T>,
}

impl<T: RlstScalar + MatrixInverse> LagrangeElementFamily<T> {
    /// Create new family
    pub fn new(degree: usize, continuity: Continuity) -> Self {
        Self {
            degree,
            continuity,
            _t: PhantomData,
        }
    }
}

impl<T: RlstScalar + MatrixInverse> ElementFamily for LagrangeElementFamily<T> {
    type T = T;
    type FiniteElement = CiarletElement<T>;
    type CellType = ReferenceCellType;
    fn element(&self, cell_type: ReferenceCellType) -> CiarletElement<T> {
        create::<T>(cell_type, self.degree, self.continuity)
    }
}
