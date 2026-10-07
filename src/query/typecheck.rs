//! Plan-time type checks for expressions whose bad input the row
//! evaluator (`physical::evaluate_expression`, which returns a plain
//! `Value` and has no error channel) could only turn into a silent NULL.
//! Run on every lowered expression before rows are evaluated, so the
//! mistake surfaces as an error naming the function and the types.
//!
//! Covers the bit-vector functions: `TANIMOTO`/`JACCARD`/`HAMMING` need two
//! `BitVector`s of the same length, `BIT_COUNT` one `BitVector`.

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
