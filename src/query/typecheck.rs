//! Plan-time type checks for expressions whose bad input the row
//! evaluator (`physical::evaluate_expression`, which returns a plain
//! `Value` and has no error channel) could only turn into a silent NULL.
//! Run on every lowered expression before rows are evaluated, so the
//! mistake surfaces as an error naming the function and the types.
//!
//! Covers the bit-vector functions (`TANIMOTO`/`JACCARD`/`HAMMING` need two
//! `BitVector`s of the same length, `BIT_COUNT` one `BitVector`) and the
//! spectral ones (`SPEC_*`: two `Matrix(2, n)` peak lists, then numbers).
//! Problems only the data can show (unsorted m/z) go through `row_error`.

use super::logical::{infer_expr_type_full, Expr, VectorFnKind};
use crate::core::tuple::Schema;
use crate::core::value::ValueType;

pub fn check_expr(expr: &Expr, schema: &Schema) -> Result<(), String> {
    if let Expr::VectorFn { func, args } = expr {
        let name = match func {
            VectorFnKind::Tanimoto => Some(("TANIMOTO", 2)),
            VectorFnKind::Jaccard => Some(("JACCARD", 2)),
            VectorFnKind::Hamming => Some(("HAMMING", 2)),
            VectorFnKind::BitCount => Some(("BIT_COUNT", 1)),
            _ => None,
        };
        if let Some((name, arity)) = name {
            if args.len() != arity {
                return Err(format!(
                    "{} takes {} argument(s), got {}",
                    name,
                    arity,
                    args.len()
                ));
            }
            let mut lengths = Vec::new();
            for arg in args {
                match infer_expr_type_full(arg, schema) {
                    ValueType::BitVector(n) => lengths.push(n),
                    // Unknown at plan time (e.g. an unresolved column, which
                    // other validation reports) -- nothing to check here.
                    ValueType::Null => {}
                    other => {
                        return Err(format!(
                            "{} expects BitVector arguments, got {}",
                            name, other
                        ))
                    }
                }
            }
            // 0 = length known only at runtime (CAST of a string column).
            let known: Vec<usize> = lengths.into_iter().filter(|&n| n > 0).collect();
            if known.len() == 2 && known[0] != known[1] {
                return Err(format!(
                    "{}: BitVector lengths differ ({} vs {})",
                    name, known[0], known[1]
                ));
            }
        }
    }
    if let Expr::VectorFn { func, args } = expr {
        let spec = match func {
            VectorFnKind::SpecCosine => Some(("SPEC_COSINE", 3, 5)),
            VectorFnKind::SpecCosineMod => Some(("SPEC_COSINE_MOD", 4, 6)),
            VectorFnKind::SpecMatches => Some(("SPEC_MATCHES", 3, 4)),
            _ => None,
        };
        if let Some((name, min, max)) = spec {
            if args.len() < min || args.len() > max {
                return Err(format!(
                    "{} takes {} to {} arguments, got {}",
                    name,
                    min,
                    max,
                    args.len()
                ));
            }
            for (i, arg) in args.iter().enumerate() {
                let t = infer_expr_type_full(arg, schema);
                let ok = if i < 2 {
                    matches!(t, ValueType::Matrix(2, _) | ValueType::Null)
                } else {
                    matches!(
                        t,
                        ValueType::Int | ValueType::Float | ValueType::Float64 | ValueType::Null
                    )
                };
                if !ok {
                    let expected = if i < 2 {
                        "a peak list Matrix(2, n)"
                    } else {
                        "a number"
                    };
                    return Err(format!(
                        "{}: argument {} must be {}, got {}",
                        name,
                        i + 1,
                        expected,
                        t
                    ));
                }
            }
        }
    }
    if let Expr::VectorFn { func, args } = expr {
        // Similarity on a SparseVector: the other side must be a Vector or
        // SparseVector of the same dimension. (Dense-only calls keep their
        // existing behavior.)
        if matches!(func, VectorFnKind::CosineSim | VectorFnKind::Dot) && args.len() == 2 {
            let name = if matches!(func, VectorFnKind::CosineSim) {
                "COSINE_SIM"
            } else {
                "DOT"
            };
            let (a, b) = (
                infer_expr_type_full(&args[0], schema),
                infer_expr_type_full(&args[1], schema),
            );
            let dim = |t: &ValueType| match t {
                ValueType::Vector(d) | ValueType::SparseVector(d) => Some(*d),
                _ => None,
            };
            if matches!(a, ValueType::SparseVector(_)) || matches!(b, ValueType::SparseVector(_)) {
                for t in [&a, &b] {
                    if dim(t).is_none() && *t != ValueType::Null {
                        return Err(format!(
                            "{} expects Vector or SparseVector arguments, got {}",
                            name, t
                        ));
                    }
                }
                if let (Some(x), Some(y)) = (dim(&a), dim(&b)) {
                    if x != 0 && y != 0 && x != y {
                        return Err(format!("{}: dimensions differ ({} vs {})", name, x, y));
                    }
                }
            }
        }
        if matches!(func, VectorFnKind::SparseNew) && args.len() != 3 {
            return Err(format!(
                "SPARSE takes 3 arguments (dim, [indices], [values]), got {}",
                args.len()
            ));
        }
    }
    for child in children(expr) {
        check_expr(child, schema)?;
    }
    Ok(())
}

fn children(expr: &Expr) -> Vec<&Expr> {
    match expr {
        Expr::Column(_) | Expr::Literal(_) | Expr::VecLiteral(_) | Expr::MatLiteral(_) => vec![],
        Expr::BinaryExpr { left, right, .. } | Expr::And(left, right) | Expr::Or(left, right) => {
            vec![left, right]
        }
        Expr::Nullif(a, b) => vec![a, b],
        Expr::Not(e)
        | Expr::IsNull(e)
        | Expr::IsNotNull(e)
        | Expr::Cast { expr: e, .. }
        | Expr::AggregateExpr { expr: e, .. } => vec![e],
        Expr::In { expr, list } => std::iter::once(expr.as_ref()).chain(list).collect(),
        Expr::Between { expr, low, high } => vec![expr, low, high],
        Expr::Case {
            operand,
            branches,
            else_expr,
        } => operand
            .iter()
            .map(|e| e.as_ref())
            .chain(branches.iter().flat_map(|(c, r)| [c, r]))
            .chain(else_expr.iter().map(|e| e.as_ref()))
            .collect(),
        Expr::Coalesce(args) | Expr::ScalarFn { args, .. } | Expr::VectorFn { args, .. } => {
            args.iter().collect()
        }
    }
}
