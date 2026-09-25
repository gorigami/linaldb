// src/store.rs

use crate::core::tensor::{Shape, Tensor, TensorId};
use std::collections::HashMap;

#[derive(Debug)]
pub enum StoreError {
    ShapeMismatch(String),
    TensorNotFound(TensorId),
    InvalidTensor(String),
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StoreError::ShapeMismatch(msg) => write!(f, "Shape mismatch: {}", msg),
            StoreError::TensorNotFound(id) => write!(f, "Tensor not found: {:?}", id),
            StoreError::InvalidTensor(msg) => write!(f, "Invalid tensor: {}", msg),
        }
    }
}

impl std::error::Error for StoreError {}

/// Motor en memoria: guarda tensores en una lista.
///
/// `tensors` keeps insertion order; `index` maps each id to its position so
/// `get` is O(1) instead of a linear scan over every tensor ever inserted.
#[derive(Debug)]
pub struct InMemoryTensorStore {
    tensors: Vec<Tensor>,
    index: HashMap<TensorId, usize>,
}

impl Default for InMemoryTensorStore {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryTensorStore {
    pub fn new() -> Self {
        Self {
            tensors: Vec::new(),
            index: HashMap::new(),
        }
    }

    /// Genera un nuevo ID interno
    pub fn gen_id(&mut self) -> TensorId {
        TensorId::new()
    }

    /// Inserta un tensor a partir de shape + data
    pub fn insert_tensor(&mut self, shape: Shape, data: Vec<f32>) -> Result<TensorId, StoreError> {
        let id = self.gen_id();
        let metadata = crate::core::tensor::TensorMetadata::new(id, None);
        let tensor = Tensor::new(id, shape, data, metadata).map_err(StoreError::InvalidTensor)?;
        self.push(tensor);
        Ok(id)
    }

    /// Inserta un Tensor ya construido
    pub fn insert_existing_tensor(&mut self, tensor: Tensor) -> Result<TensorId, StoreError> {
        if tensor.data.len() != tensor.shape.num_elements() {
            return Err(StoreError::InvalidTensor(format!(
                "Tensor data length {} does not match shape {:?}",
                tensor.data.len(),
                tensor.shape.dims
            )));
        }

        let id = tensor.id;
        self.push(tensor);
        Ok(id)
    }

    /// Appends a tensor and indexes it. If the same id is inserted twice,
    /// the index keeps pointing at the first copy -- the same first-match
    /// result the previous linear scan in `get` returned.
    fn push(&mut self, tensor: Tensor) {
        self.index.entry(tensor.id).or_insert(self.tensors.len());
        self.tensors.push(tensor);
    }

    pub fn get(&self, id: TensorId) -> Result<&Tensor, StoreError> {
        self.index
            .get(&id)
            .map(|&i| &self.tensors[i])
            .ok_or(StoreError::TensorNotFound(id))
    }

    /// Every stored tensor, in insertion order.
    pub fn tensors(&self) -> &[Tensor] {
        &self.tensors
    }

    /// Removes a tensor by ID. Returns true if it was found and removed.
    pub fn remove(&mut self, id: TensorId) -> bool {
        let len_before = self.tensors.len();
        self.tensors.retain(|t| t.id != id);
        let removed = self.tensors.len() < len_before;
        if removed {
            self.reindex();
        }
        removed
    }

    fn reindex(&mut self) {
        self.index.clear();
        for (i, t) in self.tensors.iter().enumerate() {
            self.index.entry(t.id).or_insert(i);
        }
    }

    /// Clears the store
    pub fn clear(&mut self) {
        self.tensors.clear();
        self.index.clear();
    }
}
