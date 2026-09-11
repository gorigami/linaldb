use std::fmt;

/// Operaciones binarias de alto nivel
#[derive(Debug, Clone)]
pub enum BinaryOp {
    /// a + b (element-wise)
    Add,
    /// a - b (element-wise)
    Subtract,
    /// a * b (element-wise)
    Multiply,
    /// a / b (element-wise)
    Divide,
    /// CORRELATE a WITH b  -> Pearson correlation coefficient(a, b) (rank-1)
    Correlate,
    /// SIMILARITY a WITH b -> cosine_similarity(a, b) (rank-1)
    Similarity,
    /// DISTANCE a TO b -> distancia L2 (rank-1)
    Distance,
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            BinaryOp::Add => "ADD",
            BinaryOp::Subtract => "SUBTRACT",
            BinaryOp::Multiply => "MULTIPLY",
            BinaryOp::Divide => "DIVIDE",
            BinaryOp::Correlate => "CORRELATE",
            BinaryOp::Similarity => "SIMILARITY",
            BinaryOp::Distance => "DISTANCE",
        };
        write!(f, "{}", s)
    }
}

/// Operaciones unarias
#[derive(Debug, Clone)]
pub enum UnaryOp {
    /// SCALE a BY s
    Scale(f32),
    /// NORMALIZE a
    Normalize,
    /// TRANSPOSE a (matrix transpose)
    Transpose,
    /// FLATTEN a (flatten to 1D)
    Flatten,
    /// SUM a (sum of all elements)
    Sum,
    /// MEAN a (average of all elements)
    Mean,
    /// STDEV a (standard deviation)
    Stdev,
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnaryOp::Scale(fac) => write!(f, "SCALE(by={:.4})", fac),
            UnaryOp::Normalize => write!(f, "NORMALIZE"),
            UnaryOp::Transpose => write!(f, "TRANSPOSE"),
            UnaryOp::Flatten => write!(f, "FLATTEN"),
            UnaryOp::Sum => write!(f, "SUM"),
            UnaryOp::Mean => write!(f, "MEAN"),
            UnaryOp::Stdev => write!(f, "STDEV"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorKind {
    /// Comportamiento por defecto (permite operaciones relajadas)
    Normal,
    /// Comportamiento estricto (shapes deben coincidir para element-wise)
    Strict,
    /// Tensor perezoso (almacena una expresión)
    Lazy,
}
