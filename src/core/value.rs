// src/value.rs

//use super::tensor::Tensor;
//use crate::core::tensor::Shape;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Re-exported so every other module reaches `Complex64` via
/// `crate::core::value::Complex64` -- the same module `Value::Complex`
/// itself lives in -- rather than each needing its own `num_complex`
/// import.
pub use num_complex::Complex64;

/// Represents a value in the database - supports heterogeneous types
/// Represents a value in the database - supports heterogeneous types
/// Represents a value in the database - supports heterogeneous types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Value {
    Float(f32),
    /// Full double-precision scalar. Distinct from `Float` so a DSL user can
    /// explicitly opt into f64 (via a `DOUBLE` column or `CAST(... AS DOUBLE)`)
    /// where f32's ~7 significant digits aren't enough (e.g. GPS timestamps).
    Float64(f64),
    Int(i64),
    String(String),
    Bool(bool),
    Vector(Vec<f32>),      // Embedding vector
    Matrix(Vec<Vec<f32>>), // Matrix (2D Tensor)
    /// Scalar complex number, `f64` precision throughout (matching
    /// `Float64`'s rationale). Deliberately **scalar-only** -- a genuine
    /// `Tensor<Complex>` type is a separate, larger initiative (the same way
    /// this engine still has no `f64` tensor storage, only `f64` scalars).
    /// `EIGENVALUES_GENERAL`/`EIGEN_GENERAL` (`core::linalg`) are this
    /// type's first producer; `FFT`'s spectrum output stays the existing
    /// `Matrix(2, N)` (re/im row) convention, unchanged -- see
    /// SCIENTIFIC_ENGINE_EXPANSION_PLAN.md Phase 3.
    Complex(Complex64),
    Null,
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Float(a), Value::Float(b)) => a.to_bits() == b.to_bits(),
            (Value::Float64(a), Value::Float64(b)) => a.to_bits() == b.to_bits(),
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Vector(a), Value::Vector(b)) => {
                if a.len() != b.len() {
                    return false;
                }
                a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
            }
            (Value::Matrix(a), Value::Matrix(b)) => {
                if a.len() != b.len() {
                    return false;
                }
                for i in 0..a.len() {
                    if a[i].len() != b[i].len() {
                        return false;
                    }
                    if !a[i]
                        .iter()
                        .zip(&b[i])
                        .all(|(x, y)| x.to_bits() == y.to_bits())
                    {
                        return false;
                    }
                }
                true
            }
            (Value::Complex(a), Value::Complex(b)) => {
                a.re.to_bits() == b.re.to_bits() && a.im.to_bits() == b.im.to_bits()
            }
            (Value::Null, Value::Null) => true,
            _ => false,
        }
    }
}

impl Eq for Value {}

impl std::hash::Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Value::Float(v) => v.to_bits().hash(state),
            Value::Float64(v) => v.to_bits().hash(state),
            Value::Int(v) => v.hash(state),
            Value::String(v) => v.hash(state),
            Value::Bool(v) => v.hash(state),
            Value::Vector(v) => {
                v.len().hash(state);
                for f in v {
                    f.to_bits().hash(state);
                }
            }
            Value::Matrix(m) => {
                m.len().hash(state);
                if !m.is_empty() {
                    m[0].len().hash(state);
                }
                for row in m {
                    for f in row {
                        f.to_bits().hash(state);
                    }
                }
            }
            Value::Complex(v) => {
                v.re.to_bits().hash(state);
                v.im.to_bits().hash(state);
            }
            Value::Null => {}
        }
    }
}

/// Type descriptor for values
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ValueType {
    Float,
    Float64,
    Int,
    String,
    Bool,
    Vector(usize),        // Vector with fixed dimension
    Matrix(usize, usize), // Matrix (rows, cols)
    Complex,
    Null,
}

impl Value {
    /// Get the type of this value
    pub fn value_type(&self) -> ValueType {
        match self {
            Value::Float(_) => ValueType::Float,
            Value::Float64(_) => ValueType::Float64,
            Value::Int(_) => ValueType::Int,
            Value::String(_) => ValueType::String,
            Value::Bool(_) => ValueType::Bool,
            Value::Vector(v) => ValueType::Vector(v.len()),
            Value::Matrix(m) => {
                if m.is_empty() {
                    ValueType::Matrix(0, 0)
                } else {
                    ValueType::Matrix(m.len(), m[0].len())
                }
            }
            Value::Complex(_) => ValueType::Complex,
            Value::Null => ValueType::Null,
        }
    }

    // ... existing impls ...

    /// Check if this value is null
    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    /// Try to convert to f32
    pub fn as_float(&self) -> Option<f32> {
        match self {
            Value::Float(f) => Some(*f),
            Value::Float64(f) => Some(*f as f32),
            Value::Int(i) => Some(*i as f32),
            _ => None,
        }
    }

    /// Try to convert to f64, preserving full precision when the value is
    /// already a `Float64`.
    pub fn as_float64(&self) -> Option<f64> {
        match self {
            Value::Float64(f) => Some(*f),
            Value::Float(f) => Some(*f as f64),
            Value::Int(i) => Some(*i as f64),
            _ => None,
        }
    }

    /// Try to convert to i64
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            Value::Float(f) => Some(*f as i64),
            Value::Float64(f) => Some(*f as i64),
            _ => None,
        }
    }

    /// Try to get string reference
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    /// Try to get bool
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Try to convert to a `Complex64`, promoting any real numeric `Value`
    /// (`Int`/`Float`/`Float64`) to a zero-imaginary-part complex number --
    /// the same "mixed arithmetic always promotes" convention `Float64`'s
    /// own doc comment describes for `Float`/`Int`.
    pub fn as_complex(&self) -> Option<Complex64> {
        match self {
            Value::Complex(c) => Some(*c),
            Value::Float64(f) => Some(Complex64::new(*f, 0.0)),
            Value::Float(f) => Some(Complex64::new(*f as f64, 0.0)),
            Value::Int(i) => Some(Complex64::new(*i as f64, 0.0)),
            _ => None,
        }
    }

    /// Try to get vector reference
    pub fn as_vector(&self) -> Option<&[f32]> {
        match self {
            Value::Vector(v) => Some(v),
            _ => None,
        }
    }

    /// Equality for `=`/`!=`/`IN` predicate evaluation -- distinct from
    /// `compare()` because `Complex` has real equality but no total order
    /// (`compare()` correctly returns `None` for it, since there's no
    /// meaningful answer to `>`/`<`, but `None` also means "incomparable"
    /// to every `=`/`!=` call site, which would make `WHERE z = 1+2i`
    /// silently never match and `WHERE z != 1+2i` silently never match
    /// either -- wrong for a type that *does* have well-defined equality).
    /// Every other type's equality still comes from `compare()`'s existing
    /// cross-type numeric/bool promotions (Int vs Float, Bool vs 0/1, ...)
    /// unchanged -- only `Complex` gets a real-`PartialEq`-based answer
    /// instead of `None`.
    pub fn equals(&self, other: &Value) -> Option<bool> {
        if matches!(self, Value::Complex(_)) || matches!(other, Value::Complex(_)) {
            return match (self.as_complex(), other.as_complex()) {
                (Some(a), Some(b)) => Some(a == b),
                _ => None,
            };
        }
        self.compare(other)
            .map(|ord| ord == std::cmp::Ordering::Equal)
    }

    /// Compare values (for sorting and filtering)
    pub fn compare(&self, other: &Value) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;

        match (self, other) {
            (Value::Float(a), Value::Float(b)) => a.partial_cmp(b),
            (Value::Float64(a), Value::Float64(b)) => a.partial_cmp(b),
            (Value::Int(a), Value::Int(b)) => Some(a.cmp(b)),
            (Value::String(a), Value::String(b)) => Some(a.cmp(b)),
            (Value::Bool(a), Value::Bool(b)) => Some(a.cmp(b)),
            (Value::Null, Value::Null) => Some(Ordering::Equal),
            (Value::Null, _) => Some(Ordering::Less),
            (_, Value::Null) => Some(Ordering::Greater),
            // Cross-type numeric comparison
            (Value::Float(a), Value::Int(b)) => a.partial_cmp(&(*b as f32)),
            (Value::Int(a), Value::Float(b)) => (*a as f32).partial_cmp(b),
            // Any pairing touching Float64 compares at full f64 precision.
            (Value::Float64(a), Value::Float(b)) => a.partial_cmp(&(*b as f64)),
            (Value::Float(a), Value::Float64(b)) => (*a as f64).partial_cmp(b),
            (Value::Float64(a), Value::Int(b)) => a.partial_cmp(&(*b as f64)),
            (Value::Int(a), Value::Float64(b)) => (*a as f64).partial_cmp(b),
            // Bool vs Int: only 0/1 are treated as false/true (the common
            // "boolean-as-integer" SQL idiom, and what `DSL_REFERENCE.md`'s
            // own `WHERE active = 1` example assumes) -- any other integer
            // is incomparable rather than guessing a truthiness rule for it.
            (Value::Bool(a), Value::Int(b)) if *b == 0 || *b == 1 => Some((*a as i64).cmp(b)),
            (Value::Int(a), Value::Bool(b)) if *a == 0 || *a == 1 => Some(a.cmp(&(*b as i64))),
            _ => None, // Vectors and Matrices not comparable for sorting currently
        }
    }

    /// Check if this value matches the given type
    pub fn matches_type(&self, value_type: &ValueType) -> bool {
        match (self, value_type) {
            (Value::Float(_), ValueType::Float) => true,
            (Value::Float64(_), ValueType::Float64) => true,
            (Value::Int(_), ValueType::Int) => true,
            (Value::String(_), ValueType::String) => true,
            (Value::Bool(_), ValueType::Bool) => true,
            (Value::Vector(v), ValueType::Vector(dim)) => v.len() == *dim,
            (Value::Matrix(m), ValueType::Matrix(r, c)) => {
                m.len() == *r && (m.is_empty() || m[0].len() == *c)
            }
            (Value::Complex(_), ValueType::Complex) => true,
            (Value::Null, _) => true, // Null matches any type if nullable
            _ => false,
        }
    }
}

/// Real scientific data routinely has magnitudes far outside typical
/// relational values (e.g. LIGO strain ~1e-21, astronomical distances
/// ~1e20+): plain decimal `Display` on f32 never switches to scientific
/// notation, so these print as 20+ digits of leading/trailing zeros instead
/// of a readable number. Rust's `{:?}` (Debug) on f32 already does this
/// switch (which is why Tensor's SHOW output, going through Debug on a
/// Vec<f32>, looks fine) -- this brings `Value`'s own `Display` in line for
/// the same magnitudes, only for values plain decimal notation renders
/// unreadable, leaving normal-range values unchanged.
fn format_f32(v: f32) -> String {
    if v != 0.0 && (v.abs() < 1e-4 || v.abs() >= 1e15) {
        format!("{:e}", v)
    } else {
        format!("{}", v)
    }
}

/// f64 counterpart of `format_f32`, kept as a separate function (not a
/// generic) so `Value::Float`'s existing f32 rendering is never routed
/// through a widened formatter and can't drift.
fn format_f64(v: f64) -> String {
    if v != 0.0 && (v.abs() < 1e-4 || v.abs() >= 1e15) {
        format!("{:e}", v)
    } else {
        format!("{}", v)
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Float(v) => write!(f, "{}", format_f32(*v)),
            Value::Float64(v) => write!(f, "{}", format_f64(*v)),
            Value::Int(v) => write!(f, "{}", v),
            Value::String(v) => write!(f, "\"{}\"", v),
            Value::Bool(v) => write!(f, "{}", v),
            Value::Vector(v) => {
                write!(f, "[")?;
                for (i, val) in v.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", format_f32(*val))?;
                }
                write!(f, "]")
            }
            Value::Matrix(m) => {
                write!(f, "[")?;
                for (i, row) in m.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "[")?;
                    for (j, val) in row.iter().enumerate() {
                        if j > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", format_f32(*val))?;
                    }
                    write!(f, "]")?;
                }
                write!(f, "]")
            }
            Value::Complex(v) => write!(f, "{}", format_complex(*v)),
            Value::Null => write!(f, "NULL"),
        }
    }
}

/// `a+bi` / `a-bi`, matching the conventional mathematical notation (not
/// `num_complex::Complex`'s own `Display`, which renders `a+bi` too but
/// without `format_f64`'s scientific-notation switch for extreme
/// magnitudes -- consistent with every other numeric `Value` variant here).
fn format_complex(v: Complex64) -> String {
    if v.im < 0.0 {
        format!("{}-{}i", format_f64(v.re), format_f64(-v.im))
    } else {
        format!("{}+{}i", format_f64(v.re), format_f64(v.im))
    }
}

impl fmt::Display for ValueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValueType::Float => write!(f, "FLOAT"),
            ValueType::Float64 => write!(f, "DOUBLE"),
            ValueType::Int => write!(f, "INT"),
            ValueType::String => write!(f, "STRING"),
            ValueType::Bool => write!(f, "BOOL"),
            ValueType::Vector(dim) => write!(f, "VECTOR[{}]", dim),
            ValueType::Matrix(r, c) => write!(f, "MATRIX[{}, {}]", r, c),
            ValueType::Complex => write!(f, "COMPLEX"),
            ValueType::Null => write!(f, "NULL"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_types() {
        assert_eq!(Value::Float(1.5).value_type(), ValueType::Float);
        assert_eq!(Value::Int(42).value_type(), ValueType::Int);
        assert_eq!(
            Value::String("hello".to_string()).value_type(),
            ValueType::String
        );
        assert_eq!(Value::Bool(true).value_type(), ValueType::Bool);
        assert_eq!(
            Value::Vector(vec![1.0, 2.0, 3.0]).value_type(),
            ValueType::Vector(3)
        );
        assert_eq!(Value::Null.value_type(), ValueType::Null);
    }

    // ...
}
