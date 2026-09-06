use super::{Index, IndexType};
use crate::core::tensor::Tensor;
use crate::core::value::Value;
use std::collections::HashMap;

/// An index based on a hash map for exact value lookups
#[derive(Debug)]
pub struct HashIndex {
    /// Maps values to a list of row IDs that contain that value
    /// Note: Value must ensure it is hashable and comparable suitable for keys
    map: HashMap<String, Vec<usize>>,
}

impl Default for HashIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl HashIndex {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    /// Helper to convert Value to a string key for the HashMap
    /// We use String keys because f32 isn't Hash/Eq, and Value derived traits can be tricky
    fn get_key(value: &Value) -> String {
        match value {
            Value::Int(i) => i.to_string(),
            Value::Float(f) => f.to_string(), // MVP: simple float string rep (beware precision)
            Value::Float64(f) => f.to_string(),
            Value::String(s) => s.clone(),
            Value::Bool(b) => b.to_string(),
            Value::Vector(v) => format!("{:?}", v),
            Value::Matrix(m) => format!("{:?}", m),
            Value::Null => "NULL".to_string(),
        }
    }
}

impl Index for HashIndex {
    fn add(&mut self, row_id: usize, value: &Value) -> Result<(), String> {
        let key = Self::get_key(value);
        self.map.entry(key).or_default().push(row_id);
        Ok(())
    }

    fn lookup(&self, value: &Value) -> Result<Vec<usize>, String> {
        let key = Self::get_key(value);
        Ok(self.map.get(&key).cloned().unwrap_or_default())
    }

    fn search(&self, _query: &Tensor, _k: usize) -> Result<Vec<(usize, f32)>, String> {
        Err("HashIndex does not support vector similarity search".to_string())
    }

    fn index_type(&self) -> IndexType {
        IndexType::Hash
    }

    fn box_clone(&self) -> Box<dyn Index> {
        Box::new(Self {
            map: self.map.clone(),
        })
    }
}
