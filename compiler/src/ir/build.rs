//! AST→UIR builder.

use std::collections::HashMap;

use crate::ast::{Dim, TypeExpr};

use super::error::BuildError;
use super::types::Shape;

pub(crate) fn resolve_type(
    ty: &TypeExpr,
    params: &HashMap<&str, u64>,
) -> Result<Shape, BuildError> {
    let mut dims: Vec<u64> = Vec::with_capacity(ty.dims.len());
    for dim in &ty.dims {
        match dim {
            Dim::Integer(n) => dims.push(*n),
            Dim::Symbol(name) => {
                let v = params
                    .get(name.as_str())
                    .copied()
                    .ok_or_else(|| BuildError::unknown_dim(name, ty.span))?;
                dims.push(v);
            }
        }
    }
    Ok(Shape(dims))
}
