use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use uuid::Uuid;

/// Unique identifier for a tensor
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorId(pub Uuid);

impl Default for TensorId {
    fn default() -> Self {
        Self::new()
    }
}

impl TensorId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

/// Unique identifier for an execution/query
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExecutionId(pub Uuid);

impl Default for ExecutionId {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

/// Information about how a tensor was derived
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lineage {
    pub execution_id: ExecutionId,
    pub operation: String,
    pub inputs: Vec<TensorId>,
}

/// Metadata about a tensor
#[derive(Debug, Clone)]
pub struct TensorMetadata {
    pub id: TensorId,
    pub created_at: DateTime<Utc>,
    pub creator: Option<String>,
    pub lineage: Option<Lineage>,
    pub(crate) data_hash: std::sync::OnceLock<String>,
}

impl Serialize for TensorMetadata {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let has_hash = self.data_hash.get().is_some();
        let num_fields =
            3 + (if has_hash { 1 } else { 0 }) + (if self.lineage.is_some() { 1 } else { 0 });

        let mut state = serializer.serialize_struct("TensorMetadata", num_fields)?;
        state.serialize_field("id", &self.id)?;
        state.serialize_field("created_at", &self.created_at)?;
        state.serialize_field("creator", &self.creator)?;

        if let Some(lineage) = &self.lineage {
            state.serialize_field("lineage", lineage)?;
        }

        if let Some(hash) = self.data_hash.get() {
            state.serialize_field("data_hash", hash)?;
        }

        state.end()
    }
}

impl<'de> Deserialize<'de> for TensorMetadata {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct MetadataFields {
            id: TensorId,
            created_at: DateTime<Utc>,
            creator: Option<String>,
            lineage: Option<Lineage>,
            data_hash: Option<String>,
        }

        let fields = MetadataFields::deserialize(deserializer)?;
        let lock = std::sync::OnceLock::new();
        if let Some(hash) = fields.data_hash {
            let _ = lock.set(hash);
        }

        Ok(Self {
            id: fields.id,
            created_at: fields.created_at,
            creator: fields.creator,
            lineage: fields.lineage,
            data_hash: lock,
        })
    }
}

impl TensorMetadata {
    pub fn new(id: TensorId, creator: Option<String>) -> Self {
        Self {
            id,
            created_at: Utc::now(),
            creator,
            lineage: None,
            data_hash: std::sync::OnceLock::new(),
        }
    }

    pub fn new_with_timestamp(
        id: TensorId,
        creator: Option<String>,
        timestamp: DateTime<Utc>,
    ) -> Self {
        Self {
            id,
            created_at: timestamp,
            creator,
            lineage: None,
            data_hash: std::sync::OnceLock::new(),
        }
    }

    pub fn with_lineage(mut self, lineage: Lineage) -> Self {
        self.lineage = Some(lineage);
        self
    }

    /// Internal method to compute hash if not already set.
    /// Usually called by Tensor::data_hash()
    ///
    /// Hashes `shape_dims` *and* `data` together, not `data` alone: this is
    /// used as `ProvenanceStore`'s content-addressing key
    /// (`LINEAGE_AND_LINALG_PLAN.md` Phase 0), and `data` here must already
    /// be the tensor's *logical* (shape/stride-aware) values, not its raw
    /// storage buffer -- a zero-copy `TRANSPOSE`/slice shares the exact same
    /// underlying `Arc<Vec<f32>>` as its input (this engine's storage model,
    /// see `CLAUDE.md`), so hashing the raw buffer alone made a transposed
    /// tensor and its untransposed source hash *identically*, silently
    /// misattributing ancestry for any workflow that transposes a matrix
    /// before combining it with something else (found via a real showcase
    /// notebook -- `A @ Aᵗ` before an eigendecomposition -- this project's
    /// recurring bug-finding pattern, again). Including `shape_dims` also
    /// covers the reshape case: two different shapes can iterate to the
    /// same flat value sequence, but never to the same `(shape, values)` pair.
    pub(crate) fn compute_hash(&self, shape_dims: &[usize], data: &[f32]) -> &str {
        self.data_hash.get_or_init(|| {
            let mut hasher = Sha256::new();
            for dim in shape_dims {
                hasher.update(dim.to_le_bytes());
            }
            // Convert f32 slice to byte slice for hashing
            // Safety: f32 has no padding bits and we are just reading bits for hashing
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
            };
            hasher.update(bytes);
            format!("{:x}", hasher.finalize())
        })
    }
}

/// Representa la forma (shape) de un tensor.
/// []        -> escalar (rank 0)
/// [3]       -> vector 3D (rank 1)
/// [2, 3]    -> matriz 2x3 (rank 2)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shape {
    pub dims: Vec<usize>,
}

impl Shape {
    /// Crea un nuevo shape a partir de una lista de dimensiones
    pub fn new<D: Into<Vec<usize>>>(dims: D) -> Self {
        Self { dims: dims.into() }
    }

    /// Número de dimensiones (rank)
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Número total de elementos
    pub fn num_elements(&self) -> usize {
        self.dims.iter().product()
    }

    /// Verifica si este shape es compatible con otro para operaciones elemento a elemento.
    /// Actualmente requiere igualdad exacta, pero sentamos la base para broadcasting.
    pub fn is_compatible(&self, other: &Self) -> bool {
        self.dims == other.dims
    }

    /// Comprueba si este shape puede ser broadcasted al shape objetivo.
    /// Siguiendo reglas estilo NumPy: las dimensiones deben ser iguales o una de ellas debe ser 1.
    pub fn can_broadcast_to(&self, target: &Self) -> bool {
        if self.rank() > target.rank() {
            return false;
        }

        let rank_diff = target.rank() - self.rank();
        for (i, &dim) in self.dims.iter().enumerate() {
            let target_dim = target.dims[i + rank_diff];
            if dim != 1 && dim != target_dim {
                return false;
            }
        }
        true
    }
}

/// Tensor denso de f32 con layout row-major
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor {
    pub id: TensorId,
    pub shape: Shape,
    /// Primary storage using Arc for zero-copy sharing
    pub data: Arc<Vec<f32>>,
    pub metadata: Arc<TensorMetadata>,
    /// Strides for multi-dimensional indexing (counters for each dimension)
    pub strides: Vec<usize>,
    /// Offset from the start of the data buffer
    pub offset: usize,
}

impl Tensor {
    /// Helper to compute default row-major strides for a shape
    pub fn compute_default_strides(shape: &Shape) -> Vec<usize> {
        let mut strides = vec![0; shape.rank()];
        let mut stride = 1;
        for i in (0..shape.rank()).rev() {
            strides[i] = stride;
            stride *= shape.dims[i];
        }
        strides
    }

    /// Creates a new tensor, wrapping data in an Arc
    pub fn new(
        id: TensorId,
        shape: Shape,
        data: Vec<f32>,
        metadata: TensorMetadata,
    ) -> Result<Self, String> {
        let expected = shape.num_elements();
        if data.len() != expected {
            return Err(format!(
                "Data length {} does not match shape {:?} (expected {})",
                data.len(),
                shape.dims,
                expected
            ));
        }

        let strides = Self::compute_default_strides(&shape);

        Ok(Self {
            id,
            shape,
            data: Arc::new(data),
            metadata: Arc::new(metadata),
            strides,
            offset: 0,
        })
    }

    /// Creates a tensor from shared data (zero-copy)
    pub fn from_shared(
        id: TensorId,
        shape: Shape,
        data: Arc<Vec<f32>>,
        metadata: Arc<TensorMetadata>,
    ) -> Result<Self, String> {
        let expected = shape.num_elements();
        // With strides/offset, the data buffer can be larger than the shape
        // checking equality is too strict for views.
        // We should check bounds instead, but for now assuming full-buffer views unless offset is specified
        // For from_shared (contiguous default), we might want to keep the check loosely
        // or relax it if we expect to use this for slices later?
        // Phase 8 plan says from_shared is default contiguous.
        if data.len() < expected {
            return Err(format!(
                "Data length {} is too small for shape {:?} (expected {})",
                data.len(),
                shape.dims,
                expected
            ));
        }

        let strides = Self::compute_default_strides(&shape);

        Ok(Self {
            id,
            shape,
            data,
            metadata,
            strides,
            offset: 0,
        })
    }

    /// Creates a tensor from shared data with explicit strides/offset (Advanced)
    pub fn from_shared_strided(
        id: TensorId,
        shape: Shape,
        data: Arc<Vec<f32>>,
        metadata: Arc<TensorMetadata>,
        strides: Vec<usize>,
        offset: usize,
    ) -> Result<Self, String> {
        // Basic validation
        // Ideally ensure max_index < data.len()
        // let max_index = offset + (dims[i]-1)*strides[i] ...

        Ok(Self {
            id,
            shape,
            data,
            metadata,
            strides,
            offset,
        })
    }

    /// Get reference to data slice
    pub fn data_ref(&self) -> &[f32] {
        &self.data
    }

    /// Get shared reference to data (zero-copy)
    pub fn share(&self) -> Arc<Vec<f32>> {
        self.data.clone()
    }

    /// Rank del tensor (0 = escalar, 1 = vector, 2 = matriz...)
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Número de elementos lógicos (basado en shape)
    pub fn len(&self) -> usize {
        self.shape.num_elements()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Checks if the tensor data is contiguous in memory (standard row-major layout)
    pub fn is_contiguous(&self) -> bool {
        if self.offset != 0 {
            return false;
        }

        // Check strides without allocating a new vector
        let mut expected_stride = 1;
        for i in (0..self.shape.rank()).rev() {
            if self.strides[i] != expected_stride {
                // Special case: dimension size 1 strides don't strictly matter for density,
                // but standard row-major convention implies they follow the pattern.
                // However, for contiguous raw buffer access, we just need to know if
                // the memory layout matches what we expect for a single slice.
                // If dim is 1, stride doesn't affect packing, but let's be strict for now
                // to match compute_default_strides logic.
                // Actually, if dim is 1, any stride is "valid" for reaching the next element (there isn't one),
                // but for `as_contiguous_slice` to return the whole range, we usually expect standard packing.
                return false;
            }
            expected_stride *= self.shape.dims[i];
        }
        true
    }

    /// Returns a slice of the data if the tensor is contiguous in memory.
    /// This handles cases where the tensor is a strict view (slice) of a larger buffer.
    pub fn as_contiguous_slice(&self) -> Option<&[f32]> {
        if self.is_contiguous() {
            let len = self.len();
            if self.data_ref().len() >= len {
                return Some(&self.data_ref()[0..len]);
            }
        }
        None
    }

    /// Get or compute the cryptographic hash of the tensor's *logical* data
    /// (shape + values as `to_logical_vec()` produces them) -- not the raw
    /// storage buffer, which a zero-copy view (transpose/slice) can share
    /// byte-for-byte with a differently-shaped tensor. This is a lazy
    /// operation; the logical materialization only runs once per tensor
    /// (cached in `TensorMetadata.data_hash`), same as before this
    /// distinction existed.
    pub fn data_hash(&self) -> &str {
        if let Some(contiguous) = self.as_contiguous_slice() {
            self.metadata.compute_hash(&self.shape.dims, contiguous)
        } else {
            self.metadata
                .compute_hash(&self.shape.dims, &self.to_logical_vec())
        }
    }

    /// Returns the logical data as a contiguous vector.
    /// If the tensor is already contiguous, returns a copy of the slice.
    /// If strided, materializes the view.
    pub fn to_logical_vec(&self) -> Vec<f32> {
        if let Some(slice) = self.as_contiguous_slice() {
            return slice.to_vec();
        }

        let len = self.len();
        let mut data = Vec::with_capacity(len);
        let rank = self.shape.rank();

        if rank == 0 {
            if !self.data.is_empty() {
                data.push(self.data[self.offset]);
            }
            return data;
        }

        // Optimized path for common ranks
        match rank {
            1 => {
                let dim = self.shape.dims[0];
                let stride = self.strides[0];
                for i in 0..dim {
                    data.push(self.data[self.offset + i * stride]);
                }
            }
            2 => {
                let rows = self.shape.dims[0];
                let cols = self.shape.dims[1];
                let s0 = self.strides[0];
                let s1 = self.strides[1];
                for i in 0..rows {
                    let row_base = self.offset + i * s0;
                    for j in 0..cols {
                        data.push(self.data[row_base + j * s1]);
                    }
                }
            }
            _ => {
                // Efficient N-dim traversal without re-calculating full offset every step
                let mut current_indices = vec![0; rank];
                let mut current_offset = self.offset;

                for _ in 0..len {
                    data.push(self.data[current_offset]);

                    // Advance indices and update offset incrementally
                    for j in (0..rank).rev() {
                        current_indices[j] += 1;
                        if current_indices[j] < self.shape.dims[j] {
                            current_offset += self.strides[j];
                            break;
                        } else {
                            // Reset dimension and adjust offset backwards
                            current_offset -= (current_indices[j] - 1) * self.strides[j];
                            current_indices[j] = 0;
                        }
                    }
                }
            }
        }
        data
    }
}

// ============================================================================
// LAZY EVALUATION STRUCTURES
// ============================================================================

/// Representa una operación pendiente en el grafo de computación
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Expression {
    /// Un tensor ya materializado
    Literal(Tensor),
    /// Suma de dos expresiones
    Add(Box<Expression>, Box<Expression>),
    /// Resta de dos expresiones
    Sub(Box<Expression>, Box<Expression>),
    /// Multiplicación elemento a elemento
    Multiply(Box<Expression>, Box<Expression>),
    /// División elemento a elemento
    Divide(Box<Expression>, Box<Expression>),
    /// Multiplicación de matrices
    MatMul(Box<Expression>, Box<Expression>),
    /// Multiplicación por escalar
    ScalarMul(Box<Expression>, f32),
    /// Normalización (L2)
    Normalize(Box<Expression>),
    /// Suma total
    Sum(Box<Expression>),
    /// Media aritmética
    Mean(Box<Expression>),
    /// Desviación estándar
    Stdev(Box<Expression>),
}

/// Un tensor que aún no ha sido evaluado
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LazyTensor {
    pub id: TensorId,
    pub expr: Expression,
    pub metadata: Arc<TensorMetadata>,
}

impl LazyTensor {
    pub fn new(id: TensorId, expr: Expression, metadata: TensorMetadata) -> Self {
        Self {
            id,
            expr,
            metadata: Arc::new(metadata),
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_metadata_serialization() {
        let id = TensorId::new();
        let metadata = TensorMetadata::new(id, Some("test_creator".to_string()));
        let serialized = serde_json::to_string(&metadata).unwrap();

        // Initially, data_hash should not be present in serialized JSON
        assert!(!serialized.contains("data_hash"));

        // Compute hash
        let data = vec![1.0, 2.0, 3.0];
        metadata.compute_hash(&[3], &data);

        let serialized_with_hash = serde_json::to_string(&metadata).unwrap();
        assert!(serialized_with_hash.contains("data_hash"));
    }

    #[test]
    fn test_lazy_hashing() {
        let id = TensorId::new();
        let data = vec![1.0, 2.0, 3.0];
        let metadata = TensorMetadata::new(id, None);
        let tensor = Tensor::new(id, Shape::new(vec![3]), data, metadata).unwrap();

        // Hash should not be set yet
        assert!(tensor.metadata.data_hash.get().is_none());

        // Request hash
        let hash1 = tensor.data_hash();
        assert!(!hash1.is_empty());

        // Hash should be cached now
        assert!(tensor.metadata.data_hash.get().is_some());

        // Request again, should be the same instance (cached)
        let hash2 = tensor.data_hash();
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn data_hash_distinguishes_a_zero_copy_transpose_from_its_source() {
        // A zero-copy transpose shares the exact same underlying `Arc<Vec<f32>>`
        // as its source, differing only in shape/strides -- data_hash() must
        // still tell them apart (see its own doc comment for the real bug this
        // regression-tests, found via a genuine A @ Aᵗ workflow).
        let data = Arc::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let original_id = TensorId::new();
        let original = Tensor::from_shared(
            original_id,
            Shape::new(vec![2, 3]),
            data.clone(),
            Arc::new(TensorMetadata::new(original_id, None)),
        )
        .unwrap();

        let transposed_id = TensorId::new();
        let transposed = Tensor::from_shared_strided(
            transposed_id,
            Shape::new(vec![3, 2]),
            data,
            Arc::new(TensorMetadata::new(transposed_id, None)),
            vec![1, 3],
            0,
        )
        .unwrap();

        assert_ne!(
            original.data_hash(),
            transposed.data_hash(),
            "a transposed view must not hash identically to its untransposed source"
        );
    }
}
