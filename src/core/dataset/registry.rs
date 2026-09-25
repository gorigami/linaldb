use super::Dataset;
use std::collections::HashMap;

/// Registry to track datasets within the runtime scope.
/// This registry is typically owned by a DatabaseInstance or ExecutionContext.
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct DatasetRegistry {
    datasets: HashMap<String, Dataset>,
}

impl DatasetRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, dataset: Dataset) -> Result<(), String> {
        if self.datasets.contains_key(&dataset.name) {
            return Err(format!("Dataset '{}' already exists", dataset.name));
        }
        self.datasets.insert(dataset.name.clone(), dataset);
        Ok(())
    }

    pub fn unregister(&mut self, name: &str) -> Option<Dataset> {
        self.datasets.remove(name)
    }

    pub fn get(&self, name: &str) -> Option<&Dataset> {
        self.datasets.get(name)
    }

    pub fn get_mut(&mut self, name: &str) -> Option<&mut Dataset> {
        self.datasets.get_mut(name)
    }

    pub fn exists(&self, name: &str) -> bool {
        self.datasets.contains_key(name)
    }

    pub fn list_names(&self) -> Vec<String> {
        self.datasets.keys().cloned().collect()
    }

    pub fn datasets(&self) -> &HashMap<String, Dataset> {
        &self.datasets
    }

    pub fn clear(&mut self) {
        self.datasets.clear();
    }

    /// Find a dataset by its metadata hash.
    pub fn get_by_hash(&self, hash: &str) -> Option<&Dataset> {
        self.datasets
            .values()
            .find(|ds| ds.metadata.as_ref().is_some_and(|m| m.hash == hash))
    }
}
