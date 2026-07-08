use std::sync::atomic::{AtomicU64, Ordering};

use crate::core::tensor::Shape;
use crate::dsl::ast::*;
use crate::dsl::{DslError, DslOutput};
use crate::engine::context::ExecutionContext;
use crate::engine::{BinaryOp, TensorDb, UnaryOp};

// ─── Let / expression evaluation ──────────────────────────────────────────────

pub(super) fn eval_let(
    db: &mut TensorDb,
    ctx: &mut ExecutionContext,
    output_name: &str,
    lazy: bool,
    expr: &Expr,
    line_no: usize,
) -> Result<DslOutput, DslError> {
    let result = eval_expr_to_name(db, ctx, output_name, expr, lazy, line_no)?;
    Ok(DslOutput::Message(if lazy {
        format!("Defined lazy variable: {}", result)
    } else {
        format!("Defined variable: {}", result)
    }))
}

fn eval_expr_to_name(
    db: &mut TensorDb,
    ctx: &mut ExecutionContext,
    desired_name: &str,
    expr: &Expr,
    lazy: bool,
    line_no: usize,
) -> Result<String, DslError> {
    let eng = |e| DslError::Engine {
        line: line_no,
        source: e,
    };

    match expr {
        Expr::Ref(name) => Ok(name.clone()),

        Expr::Int(n) => {
            db.insert_named(desired_name, Shape::new(vec![]), vec![*n as f32])
                .map_err(eng)?;
            Ok(desired_name.to_string())
        }
        Expr::Scalar(n) => {
            db.insert_named(desired_name, Shape::new(vec![]), vec![*n as f32])
                .map_err(eng)?;
            Ok(desired_name.to_string())
        }

        Expr::StringLit(_) => Err(DslError::Parse {
            line: line_no,
            msg: "string literal is not a valid tensor expression".into(),
        }),

        Expr::Infix { op, lhs, rhs } => {
            let l_tmp = fresh_temp("l");
            let r_tmp = fresh_temp("r");
            let l = eval_expr_to_name(db, ctx, &l_tmp, lhs, false, line_no)?;
            let r = eval_expr_to_name(db, ctx, &r_tmp, rhs, false, line_no)?;
            let bin_op = infix_to_binary_op(*op);
            if lazy {
                db.eval_lazy_binary(ctx, desired_name, &l, &r, bin_op)
            } else {
                db.eval_binary(ctx, desired_name, &l, &r, bin_op)
            }
            .map_err(eng)?;
            Ok(desired_name.to_string())
        }

        Expr::Call(call) => {
            eval_call(db, ctx, desired_name, call, lazy, line_no)?;
            Ok(desired_name.to_string())
        }

        Expr::Index { base, indices } => {
            let base_tmp = fresh_temp("base");
            let base_name = eval_expr_to_name(db, ctx, &base_tmp, base, false, line_no)?;
            apply_index(db, ctx, desired_name, &base_name, indices, line_no)?;
            Ok(desired_name.to_string())
        }

        Expr::Field { base, field } => {
            let base_tmp = fresh_temp("base");
            let base_name = eval_expr_to_name(db, ctx, &base_tmp, base, false, line_no)?;
            if db.get_dataset(&base_name).is_ok() || db.get_tensor_dataset(&base_name).is_some() {
                db.eval_column_access(ctx, desired_name, &base_name, field)
            } else {
                db.eval_field_access(ctx, desired_name, &base_name, field)
            }
            .map_err(eng)?;
            Ok(desired_name.to_string())
        }

        Expr::DatasetRef(name) => {
            let ds = crate::core::dataset::Dataset::new(name);
            db.register_tensor_dataset(ds.clone());
            db.register_dataset_var(desired_name.to_string(), name.clone());
            Ok(desired_name.to_string())
        }

        Expr::And(_, _) | Expr::Or(_, _) | Expr::Not(_) => Err(DslError::Parse {
            line: line_no,
            msg: "AND/OR/NOT are not valid tensor expressions".into(),
        }),
    }
}

fn eval_call(
    db: &mut TensorDb,
    ctx: &mut ExecutionContext,
    output: &str,
    call: &CallExpr,
    lazy: bool,
    line_no: usize,
) -> Result<(), DslError> {
    let eng = |e| DslError::Engine {
        line: line_no,
        source: e,
    };

    macro_rules! operand {
        ($expr:expr, $hint:expr) => {{
            let tmp = fresh_temp($hint);
            eval_expr_to_name(db, ctx, &tmp, $expr, false, line_no)?
        }};
    }

    match call {
        CallExpr::Add(a, b) => {
            let (a, b) = (operand!(a, "a"), operand!(b, "b"));
            if lazy {
                db.eval_lazy_binary(ctx, output, &a, &b, BinaryOp::Add)
            } else {
                db.eval_binary(ctx, output, &a, &b, BinaryOp::Add)
            }
            .map_err(eng)
        }
        CallExpr::Subtract(a, b) => {
            let (a, b) = (operand!(a, "a"), operand!(b, "b"));
            if lazy {
                db.eval_lazy_binary(ctx, output, &a, &b, BinaryOp::Subtract)
            } else {
                db.eval_binary(ctx, output, &a, &b, BinaryOp::Subtract)
            }
            .map_err(eng)
        }
        CallExpr::Multiply(a, b) => {
            let (a, b) = (operand!(a, "a"), operand!(b, "b"));
            if lazy {
                db.eval_lazy_binary(ctx, output, &a, &b, BinaryOp::Multiply)
            } else {
                db.eval_binary(ctx, output, &a, &b, BinaryOp::Multiply)
            }
            .map_err(eng)
        }
        CallExpr::Divide(a, b) => {
            let (a, b) = (operand!(a, "a"), operand!(b, "b"));
            if lazy {
                db.eval_lazy_binary(ctx, output, &a, &b, BinaryOp::Divide)
            } else {
                db.eval_binary(ctx, output, &a, &b, BinaryOp::Divide)
            }
            .map_err(eng)
        }
        CallExpr::Correlate(a, b) => {
            let (a, b) = (operand!(a, "a"), operand!(b, "b"));
            db.eval_binary(ctx, output, &a, &b, BinaryOp::Correlate)
                .map_err(eng)
        }
        CallExpr::Similarity(a, b) => {
            let (a, b) = (operand!(a, "a"), operand!(b, "b"));
            db.eval_binary(ctx, output, &a, &b, BinaryOp::Similarity)
                .map_err(eng)
        }
        CallExpr::Distance(a, b) => {
            let (a, b) = (operand!(a, "a"), operand!(b, "b"));
            db.eval_binary(ctx, output, &a, &b, BinaryOp::Distance)
                .map_err(eng)
        }
        CallExpr::Matmul(a, b) => {
            let (a, b) = (operand!(a, "a"), operand!(b, "b"));
            if lazy {
                db.eval_lazy_matmul(ctx, output, &a, &b)
            } else {
                db.eval_matmul(ctx, output, &a, &b)
            }
            .map_err(eng)
        }
        CallExpr::Normalize(a) => {
            let a = operand!(a, "a");
            if lazy {
                db.eval_lazy_unary(ctx, output, &a, UnaryOp::Normalize)
            } else {
                db.eval_unary(ctx, output, &a, UnaryOp::Normalize)
            }
            .map_err(eng)
        }
        CallExpr::Transpose(a) => {
            let a = operand!(a, "a");
            db.eval_unary(ctx, output, &a, UnaryOp::Transpose)
                .map_err(eng)
        }
        CallExpr::Flatten(a) => {
            let a = operand!(a, "a");
            if lazy {
                db.eval_lazy_unary(ctx, output, &a, UnaryOp::Flatten)
            } else {
                db.eval_unary(ctx, output, &a, UnaryOp::Flatten)
            }
            .map_err(eng)
        }
        CallExpr::Sum(a) => {
            let a = operand!(a, "a");
            if lazy {
                db.eval_lazy_unary(ctx, output, &a, UnaryOp::Sum)
            } else {
                db.eval_unary(ctx, output, &a, UnaryOp::Sum)
            }
            .map_err(eng)
        }
        CallExpr::Mean(a) => {
            let a = operand!(a, "a");
            if lazy {
                db.eval_lazy_unary(ctx, output, &a, UnaryOp::Mean)
            } else {
                db.eval_unary(ctx, output, &a, UnaryOp::Mean)
            }
            .map_err(eng)
        }
        CallExpr::Stdev(a) => {
            let a = operand!(a, "a");
            if lazy {
                db.eval_lazy_unary(ctx, output, &a, UnaryOp::Stdev)
            } else {
                db.eval_unary(ctx, output, &a, UnaryOp::Stdev)
            }
            .map_err(eng)
        }
        CallExpr::Scale { input, factor } => {
            let a = operand!(input, "a");
            let op = UnaryOp::Scale(*factor as f32);
            if lazy {
                db.eval_lazy_unary(ctx, output, &a, op)
            } else {
                db.eval_unary(ctx, output, &a, op)
            }
            .map_err(eng)
        }
        CallExpr::Reshape { input, shape } => {
            let a = operand!(input, "a");
            let new_shape = Shape::new(shape.clone());
            db.eval_reshape(ctx, output, &a, new_shape).map_err(eng)
        }
        CallExpr::Stack(operands) => {
            let names: Vec<String> = operands
                .iter()
                .enumerate()
                .map(|(i, e)| {
                    let tmp = fresh_temp(&format!("s{}", i));
                    eval_expr_to_name(db, ctx, &tmp, e, false, line_no)
                })
                .collect::<Result<_, _>>()?;
            let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();
            db.eval_stack(ctx, output, name_refs, 0).map_err(eng)
        }
    }
}

fn apply_index(
    db: &mut TensorDb,
    ctx: &mut ExecutionContext,
    output: &str,
    base_name: &str,
    indices: &[IndexSpec],
    line_no: usize,
) -> Result<(), DslError> {
    use crate::engine::kernels::SliceSpec;
    let eng = |e| DslError::Engine {
        line: line_no,
        source: e,
    };

    let specs: Vec<SliceSpec> = indices
        .iter()
        .map(|i| match i {
            IndexSpec::All => SliceSpec::All,
            IndexSpec::Index(n) => SliceSpec::Index(*n),
            IndexSpec::Range(s, e) => SliceSpec::Range(*s, *e),
        })
        .collect();

    let all_single = specs.iter().all(|s| matches!(s, SliceSpec::Index(_)));
    if all_single {
        let idx: Vec<usize> = specs
            .iter()
            .filter_map(|s| {
                if let SliceSpec::Index(n) = s {
                    Some(*n)
                } else {
                    None
                }
            })
            .collect();
        db.eval_index(ctx, output, base_name, idx).map_err(eng)
    } else {
        db.eval_slice(ctx, output, base_name, specs).map_err(eng)
    }
}

pub(super) fn infix_to_binary_op(op: InfixOp) -> BinaryOp {
    match op {
        InfixOp::Add => BinaryOp::Add,
        InfixOp::Subtract => BinaryOp::Subtract,
        InfixOp::Multiply => BinaryOp::Multiply,
        InfixOp::Divide => BinaryOp::Divide,
        _ => unreachable!("comparison operators cannot appear in tensor expressions"),
    }
}

pub(super) fn fresh_temp(hint: &str) -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("_t_{}_{}", hint, n)
}

// ─── Expression serialization ─────────────────────────────────────────────────

pub fn expr_to_string(expr: &Expr) -> String {
    match expr {
        Expr::Ref(n) => n.clone(),
        Expr::Int(n) => format!("{}", n),
        Expr::Scalar(n) => format!("{}", n),
        Expr::StringLit(s) => format!("\"{}\"", s),
        Expr::Infix { op, lhs, rhs } => {
            let sym = match op {
                InfixOp::Add => "+",
                InfixOp::Subtract => "-",
                InfixOp::Multiply => "*",
                InfixOp::Divide => "/",
                InfixOp::Eq => "=",
                InfixOp::NotEq => "!=",
                InfixOp::Gt => ">",
                InfixOp::Lt => "<",
                InfixOp::GtEq => ">=",
                InfixOp::LtEq => "<=",
            };
            format!("{} {} {}", expr_to_string(lhs), sym, expr_to_string(rhs))
        }
        Expr::Field { base, field } => format!("{}.{}", expr_to_string(base), field),
        Expr::Index { base, indices } => {
            let idx: Vec<String> = indices
                .iter()
                .map(|i| match i {
                    IndexSpec::All => "*".into(),
                    IndexSpec::Index(n) => n.to_string(),
                    IndexSpec::Range(s, e) => format!("{}:{}", s, e),
                })
                .collect();
            format!("{}[{}]", expr_to_string(base), idx.join(", "))
        }
        Expr::Call(c) => call_to_string(c),
        Expr::DatasetRef(name) => format!("dataset(\"{}\")", name),
        Expr::And(l, r) => format!("({} AND {})", expr_to_string(l), expr_to_string(r)),
        Expr::Or(l, r) => format!("({} OR {})", expr_to_string(l), expr_to_string(r)),
        Expr::Not(inner) => format!("(NOT {})", expr_to_string(inner)),
    }
}

fn call_to_string(c: &CallExpr) -> String {
    match c {
        CallExpr::Add(a, b) => format!("ADD {} {}", expr_to_string(a), expr_to_string(b)),
        CallExpr::Subtract(a, b) => {
            format!("SUBTRACT {} {}", expr_to_string(a), expr_to_string(b))
        }
        CallExpr::Multiply(a, b) => {
            format!("MULTIPLY {} {}", expr_to_string(a), expr_to_string(b))
        }
        CallExpr::Divide(a, b) => format!("DIVIDE {} {}", expr_to_string(a), expr_to_string(b)),
        CallExpr::Correlate(a, b) => {
            format!("CORRELATE {} WITH {}", expr_to_string(a), expr_to_string(b))
        }
        CallExpr::Similarity(a, b) => format!(
            "SIMILARITY {} WITH {}",
            expr_to_string(a),
            expr_to_string(b)
        ),
        CallExpr::Distance(a, b) => {
            format!("DISTANCE {} TO {}", expr_to_string(a), expr_to_string(b))
        }
        CallExpr::Matmul(a, b) => format!("MATMUL {} {}", expr_to_string(a), expr_to_string(b)),
        CallExpr::Normalize(a) => format!("NORMALIZE {}", expr_to_string(a)),
        CallExpr::Transpose(a) => format!("TRANSPOSE {}", expr_to_string(a)),
        CallExpr::Flatten(a) => format!("FLATTEN {}", expr_to_string(a)),
        CallExpr::Sum(a) => format!("SUM {}", expr_to_string(a)),
        CallExpr::Mean(a) => format!("MEAN {}", expr_to_string(a)),
        CallExpr::Stdev(a) => format!("STDEV {}", expr_to_string(a)),
        CallExpr::Scale { input, factor } => {
            format!("SCALE {} BY {}", expr_to_string(input), factor)
        }
        CallExpr::Reshape { input, shape } => {
            let d: Vec<String> = shape.iter().map(|n| n.to_string()).collect();
            format!("RESHAPE {} TO [{}]", expr_to_string(input), d.join(", "))
        }
        CallExpr::Stack(ops) => {
            let names: Vec<String> = ops.iter().map(expr_to_string).collect();
            format!("STACK {}", names.join(" "))
        }
    }
}
