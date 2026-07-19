// src/value.rs

//use super::tensor::Tensor;
//use crate::core::tensor::Shape;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Represents a value in the database - supports heterogeneous types
/// Represents a value in the database - supports heterogeneous types
/// Represents a value in the database - supports heterogeneous types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Value {
    Float(f32),
    Int(i64),
    String(String),
    Bool(bool),
    Vector(Vec<f32>),      // Embedding vector
    Matrix(Vec<Vec<f32>>), // Matrix (2D Tensor)
    Null,
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Float(a), Value::Float(b)) => a.to_bits() == b.to_bits(),
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
            Value::Null => {}
        }
    }
}

/// Type descriptor for values
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ValueType {
    Float,
    Int,
    String,
    Bool,
    Vector(usize),        // Vector with fixed dimension
    Matrix(usize, usize), // Matrix (rows, cols)
    Null,
}

impl Value {
    /// Get the type of this value
    pub fn value_type(&self) -> ValueType {
        match self {
            Value::Float(_) => ValueType::Float,
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
            Value::Int(i) => Some(*i as f32),
            _ => None,
        }
    }

    /// Try to convert to i64
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            Value::Float(f) => Some(*f as i64),
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

    /// Try to get vector reference
    pub fn as_vector(&self) -> Option<&[f32]> {
        match self {
            Value::Vector(v) => Some(v),
            _ => None,
        }
    }

    /// Compare values (for sorting and filtering)
    pub fn compare(&self, other: &Value) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;

        match (self, other) {
            (Value::Float(a), Value::Float(b)) => a.partial_cmp(b),
            (Value::Int(a), Value::Int(b)) => Some(a.cmp(b)),
            (Value::String(a), Value::String(b)) => Some(a.cmp(b)),
            (Value::Bool(a), Value::Bool(b)) => Some(a.cmp(b)),
            (Value::Null, Value::Null) => Some(Ordering::Equal),
            (Value::Null, _) => Some(Ordering::Less),
            (_, Value::Null) => Some(Ordering::Greater),
            // Cross-type numeric comparison
            (Value::Float(a), Value::Int(b)) => a.partial_cmp(&(*b as f32)),
            (Value::Int(a), Value::Float(b)) => (*a as f32).partial_cmp(b),
            _ => None, // Vectors and Matrices not comparable for sorting currently
        }
    }

    /// Check if this value matches the given type
    pub fn matches_type(&self, value_type: &ValueType) -> bool {
        match (self, value_type) {
            (Value::Float(_), ValueType::Float) => true,
            (Value::Int(_), ValueType::Int) => true,
            (Value::String(_), ValueType::String) => true,
            (Value::Bool(_), ValueType::Bool) => true,
            (Value::Vector(v), ValueType::Vector(dim)) => v.len() == *dim,
            (Value::Matrix(m), ValueType::Matrix(r, c)) => {
                m.len() == *r && (m.is_empty() || m[0].len() == *c)
            }
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

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Float(v) => write!(f, "{}", format_f32(*v)),
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
            Value::Null => write!(f, "NULL"),
        }
    }
}

impl fmt::Display for ValueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValueType::Float => write!(f, "FLOAT"),
            ValueType::Int => write!(f, "INT"),
            ValueType::String => write!(f, "STRING"),
            ValueType::Bool => write!(f, "BOOL"),
            ValueType::Vector(dim) => write!(f, "VECTOR[{}]", dim),
            ValueType::Matrix(r, c) => write!(f, "MATRIX[{}, {}]", r, c),
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
