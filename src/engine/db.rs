use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::core::dataset_legacy::{Dataset, DatasetId};
use crate::core::store::{DatasetStore, InMemoryTensorStore};
use crate::core::tensor::{Lineage, Shape, Tensor, TensorId, TensorMetadata};
use crate::core::tuple::{Schema, Tuple};

use super::error::EngineError;
use super::operations::{BinaryOp, TensorKind, UnaryOp};
use crate::engine::context::ExecutionContext;

struct NameEntry {
    id: TensorId,
    kind: TensorKind,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LineageNode {
    pub tensor_id: TensorId,
    pub name: Option<String>,
    pub operation: String,
    pub inputs: Vec<LineageNode>,
}

/// Individual database instance containing its own stores and name mappings
pub struct DatabaseInstance {
    pub name: String,
    pub store: InMemoryTensorStore,
    names: HashMap<String, NameEntry>,
    dataset_store: DatasetStore,
    pub tensor_datasets: crate::core::dataset::DatasetRegistry,
    pub dataset_vars: HashMap<String, String>,
    pub backend: Box<dyn crate::core::backend::ComputeBackend>,
    /// Almacenamiento de expresiones para tensores perezosos
    pub lazy_store: HashMap<TensorId, crate::core::tensor::Expression>,
}

impl DatabaseInstance {
    pub fn new(name: String) -> Self {
        Self {
            name,
            store: InMemoryTensorStore::new(),
            names: HashMap::new(),
            dataset_store: DatasetStore::new(),
            tensor_datasets: crate::core::dataset::DatasetRegistry::new(),
            dataset_vars: HashMap::new(),
            backend: Box::new(crate::core::backend::CpuBackend::new()),
            lazy_store: HashMap::new(),
        }
    }

    // ... all existing methods of the old TensorDb ...

    pub fn set_dataset_metadata(
        &mut self,
        name: &str,
        key: String,
        value: String,
    ) -> Result<(), EngineError> {
        let dataset = self
            .dataset_store
            .get_mut_by_name(name)
            .map_err(|_| EngineError::NameNotFound(name.to_string()))?;

        dataset.metadata.extra.insert(key, value);
        dataset.metadata.updated_at = chrono::Utc::now();
        Ok(())
    }

    pub fn get_tensor_id(&self, name: &str) -> Option<TensorId> {
        self.names.get(name).map(|e| e.id)
    }

    pub fn remove_tensor(&mut self, name: &str) -> bool {
        if let Some(entry) = self.names.remove(name) {
            self.store.remove(entry.id)
        } else {
            false
        }
    }

    pub fn register_tensor_dataset(&mut self, ds: crate::core::dataset::Dataset) {
        let _ = self.tensor_datasets.register(ds);
    }

    pub fn register_dataset_var(&mut self, var_name: String, ds_name: String) {
        self.dataset_vars.insert(var_name, ds_name);
    }

    pub fn add_column_to_tensor_dataset(
        &mut self,
        ds_var_or_name: &str,
        col_name: &str,
        tensor_var: &str,
    ) -> Result<(), EngineError> {
        use crate::core::value::ValueType;
        // 1. Get tensor_id from names
        let entry = self
            .names
            .get(tensor_var)
            .ok_or_else(|| EngineError::NameNotFound(tensor_var.to_string()))?;
        let tensor_id = entry.id;

        // 2. Get tensor to check shape/type
        let tensor = self.store.get(tensor_id).map_err(|_| {
            EngineError::InvalidOp(format!("Tensor '{}' not found in store", tensor_var))
        })?;

        // 3. Get dataset name (resolve variable if needed)
        let ds_name = self
            .dataset_vars
            .get(ds_var_or_name)
            .map(|s| s.as_str())
            .unwrap_or(ds_var_or_name)
            .to_string();
        let ds_name = ds_name.as_str();

        // 4. Validate row count consistency (using immutable borrow first)
        let rows_in_new_col = match tensor.shape.rank() {
            0 => 1,
            _ => tensor.shape.dims[0],
        };

        if let Some(ds) = self.tensor_datasets.get(ds_name) {
            if let Some((_, first_ref)) = ds.columns.iter().next() {
                let graph = crate::core::dataset::DatasetGraph::new(&self.tensor_datasets);
                let first_tensor_id = graph.resolve_to_tensor(first_ref).map_err(|e| {
                    EngineError::InvalidOp(format!("Dependency resolution error: {}", e))
                })?;

                let first_tensor = self.store.get(first_tensor_id)?;
                let rows_in_ds = match first_tensor.shape.rank() {
                    0 => 1,
                    _ => first_tensor.shape.dims[0],
                };

                if rows_in_new_col != rows_in_ds {
                    return Err(EngineError::InvalidOp(format!(
                        "Column '{}' has {} rows, but dataset '{}' has {} rows",
                        col_name, rows_in_new_col, ds_name, rows_in_ds
                    )));
                }
            }
        } else {
            return Err(EngineError::InvalidOp(format!(
                "Tensor dataset '{}' not found",
                ds_name
            )));
        }

        // 5. Update schema and columns (mutable borrow now safe)
        let ds = self.tensor_datasets.get_mut(ds_name).unwrap();

        let value_type = match tensor.shape.rank() {
            1 => ValueType::Vector(tensor.shape.dims[0]),
            2 => {
                if tensor.shape.dims.len() >= 2 {
                    ValueType::Matrix(tensor.shape.dims[0], tensor.shape.dims[1])
                } else {
                    ValueType::Vector(tensor.shape.dims[0])
                }
            }
            0 => ValueType::Float,
            _ => ValueType::Vector(tensor.shape.num_elements()),
        };

        let schema = crate::core::dataset::ColumnSchema::new(
            col_name.to_string(),
            value_type,
            tensor.shape.clone(),
        );

        ds.add_column(
            col_name.to_string(),
            crate::core::dataset::ResourceReference::tensor(tensor_id),
            schema,
        );
        self.sync_tensor_dataset_to_legacy(ds_name)?;
        Ok(())
    }
    /// Verify that all columns in a tensor-first dataset point to existing tensors.
    /// Returns a list of column names with missing tensors.
    pub fn verify_tensor_dataset(&self, ds_name_or_var: &str) -> Result<Vec<String>, EngineError> {
        let ds_name = self
            .dataset_vars
            .get(ds_name_or_var)
            .map(|s| s.as_str())
            .unwrap_or(ds_name_or_var);

        let ds = self.tensor_datasets.get(ds_name).ok_or_else(|| {
            EngineError::InvalidOp(format!("Tensor dataset '{}' not found", ds_name))
        })?;

        let mut missing_cols = Vec::new();
        let graph = crate::core::dataset::DatasetGraph::new(&self.tensor_datasets);
        for (col_name, reference) in &ds.columns {
            let tensor_id = graph.resolve_to_tensor(reference).map_err(|e| {
                EngineError::InvalidOp(format!("Dependency resolution error: {}", e))
            })?;
            if self.store.get(tensor_id).is_err() {
                missing_cols.push(col_name.clone());
            }
        }
        Ok(missing_cols)
    }

    pub fn materialize_tensor_dataset(
        &self,
        name: &str,
    ) -> Result<crate::core::dataset_legacy::Dataset, EngineError> {
        // Resolve name via vars if needed
        let ds_name = self
            .dataset_vars
            .get(name)
            .map(|s| s.as_str())
            .unwrap_or(name);

        let ds = self
            .tensor_datasets
            .get(ds_name)
            .ok_or_else(|| EngineError::DatasetNotFound(ds_name.to_string()))?;

        if ds.columns.is_empty() {
            return Err(EngineError::InvalidOp(format!(
                "Cannot materialize empty tensor dataset '{}'",
                ds_name
            )));
        }

        // 1. Determine number of rows and column schemas
        let mut row_count = 0;
        let mut fields = Vec::new();
        let mut col_data = Vec::new();

        // Sort column names for deterministic schema
        let mut col_names: Vec<_> = ds.columns.keys().cloned().collect();
        col_names.sort();

        for col_name in col_names {
            let reference = ds.columns.get(&col_name).unwrap();
            let graph = crate::core::dataset::DatasetGraph::new(&self.tensor_datasets);
            let tensor_id = graph.resolve_to_tensor(reference).map_err(|e| {
                EngineError::InvalidOp(format!("Dependency resolution error: {}", e))
            })?;
            let tensor = self.store.get(tensor_id)?;

            let (rows_in_col, vt) = match tensor.shape.rank() {
                0 => (1, crate::core::value::ValueType::Float), // One row, one scalar
                1 => (
                    tensor.shape.dims[0],
                    crate::core::value::ValueType::Float, // N rows, each a scalar
                ),
                2 => (
                    tensor.shape.dims[0],
                    crate::core::value::ValueType::Vector(tensor.shape.dims[1]), // N rows, each a vector
                ),
                _ => {
                    return Err(EngineError::InvalidOp(format!(
                        "Cannot materialize tensor with rank > 2 (rank: {})",
                        tensor.shape.rank()
                    )))
                }
            };

            if row_count == 0 {
                row_count = rows_in_col;
            } else if rows_in_col != row_count {
                return Err(EngineError::InvalidOp(format!(
                    "Column '{}' has {} rows, but previous columns had {}",
                    col_name, rows_in_col, row_count
                )));
            }

            fields.push(crate::core::tuple::Field::new(&col_name, vt));
            col_data.push(tensor);
        }

        let schema = std::sync::Arc::new(crate::core::tuple::Schema::new(fields));
        let mut rows = Vec::with_capacity(row_count);

        // 2. Build rows
        for i in 0..row_count {
            let mut values = Vec::with_capacity(col_data.len());
            for tensor in &col_data {
                let val = match tensor.shape.rank() {
                    0 => crate::core::value::Value::Float(tensor.data[0]),
                    1 => crate::core::value::Value::Float(tensor.data[i]),
                    2 => {
                        let dim = tensor.shape.dims[1];
                        let start = i * dim;
                        let end = (i + 1) * dim;
                        crate::core::value::Value::Vector(tensor.data[start..end].to_vec())
                    }
                    _ => unreachable!(),
                };
                values.push(val);
            }
            rows.push(crate::core::tuple::Tuple::new(schema.clone(), values).unwrap());
        }

        let legacy_id = crate::core::dataset_legacy::DatasetId(0);
        crate::core::dataset_legacy::Dataset::with_rows(
            legacy_id,
            schema,
            rows,
            Some(ds_name.to_string()),
        )
        .map_err(EngineError::InvalidOp)
    }

    /// Materializes a tensor-first dataset (see `materialize_tensor_dataset`)
    /// and, unlike that method, also stores/refreshes the result in the
    /// queryable legacy dataset store under the same name.
    ///
    /// Without this, a dataset built via `dataset()` + `.add_column()` (or
    /// ingested via `USE DATASET FROM`) was only ever materialized as a
    /// one-shot `DslOutput::Table` for immediate display -- `SHOW ALL
    /// DATASETS`, `SELECT ... FROM <name>`, and `MATERIALIZE <name>` all
    /// look exclusively in the legacy store, which nothing ever populated,
    /// so every subsequent reference to the dataset failed with "Dataset
    /// not found" even though `SHOW <name>` (the tensor-first health-check
    /// view) worked fine. Called from every mutation site that changes a
    /// tensor-first dataset's columns, so the legacy copy never goes stale.
    pub fn sync_tensor_dataset_to_legacy(&mut self, ds_name: &str) -> Result<Dataset, EngineError> {
        let materialized = self.materialize_tensor_dataset(ds_name)?;
        if let Ok(existing) = self.dataset_store.get_mut_by_name(ds_name) {
            existing.schema = materialized.schema.clone();
            existing.rows = materialized.rows.clone();
            existing
                .metadata
                .update_stats(&existing.schema, &existing.rows);
        } else {
            let id = self.dataset_store.gen_id();
            let mut fresh = materialized.clone();
            fresh.id = id;
            self.dataset_store
                .insert(fresh, Some(ds_name.to_string()))
                .map_err(EngineError::from)?;
        }
        Ok(materialized)
    }

    /// Resets the database biological state (clear all tensors and datasets)
    pub fn reset(&mut self) {
        self.store.clear();
        self.names.clear();
        self.dataset_store.clear();
        self.tensor_datasets.clear();
        self.dataset_vars.clear();
    }
}

/// High-level engine that manages multiple DatabaseInstances
pub struct TensorDb {
    pub config: crate::core::config::EngineConfig,
    databases: HashMap<String, DatabaseInstance>,
    active_db: String,
    /// Session-wide named pipeline registry.
    pub pipelines: HashMap<String, crate::dsl::ast::StoredPipeline>,
}

impl Default for TensorDb {
    fn default() -> Self {
        Self::new()
    }
}

impl TensorDb {
    pub fn new() -> Self {
        let config = crate::core::config::EngineConfig::load();
        Self::with_config(config)
    }

    pub fn with_config(config: crate::core::config::EngineConfig) -> Self {
        let default_name = config.storage.default_db.clone();
        let mut dbs = HashMap::new();
        dbs.insert(
            default_name.clone(),
            DatabaseInstance::new(default_name.clone()),
        );

        let mut db = Self {
            databases: dbs,
            active_db: default_name,
            config,
            pipelines: HashMap::new(),
        };

        // Try to recover existing databases
        let _ = db.recover_databases();

        db
    }

    fn recover_databases(&mut self) -> Result<(), EngineError> {
        let data_dir = &self.config.storage.data_dir;
        if !data_dir.exists() {
            return Ok(());
        }

        // Scan data_dir for subdirectories (each is a database)
        if let Ok(entries) = std::fs::read_dir(data_dir) {
            for entry in entries.flatten() {
                if let Ok(file_type) = entry.file_type() {
                    if file_type.is_dir() {
                        let db_name = entry.file_name().to_string_lossy().into_owned();
                        if !self.databases.contains_key(&db_name) {
                            self.databases
                                .insert(db_name.clone(), DatabaseInstance::new(db_name));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Get reference to the active database
    pub fn active_instance(&self) -> &DatabaseInstance {
        self.databases
            .get(&self.active_db)
            .expect("Active DB must exist")
    }

    /// Get mutable reference to the active database
    pub fn active_instance_mut(&mut self) -> &mut DatabaseInstance {
        self.databases
            .get_mut(&self.active_db)
            .expect("Active DB must exist")
    }

    /// Create a new database
    pub fn create_database(&mut self, name: String) -> Result<(), EngineError> {
        if self.databases.contains_key(&name) {
            return Err(EngineError::InvalidOp(format!(
                "Database '{}' already exists",
                name
            )));
        }

        // Create directory for persistence
        let db_path = self.config.storage.data_dir.join(&name);
        if !db_path.exists() {
            std::fs::create_dir_all(&db_path).map_err(|e| {
                EngineError::InvalidOp(format!("Failed to create DB directory: {}", e))
            })?;
        }

        self.databases
            .insert(name.clone(), DatabaseInstance::new(name));
        Ok(())
    }

    /// Switch active database
    pub fn use_database(&mut self, name: &str) -> Result<(), EngineError> {
        if !self.databases.contains_key(name) {
            return Err(EngineError::InvalidOp(format!(
                "Database '{}' not found",
                name
            )));
        }
        self.active_db = name.to_string();
        Ok(())
    }

    /// Get active database name
    pub fn active_db(&self) -> &str {
        &self.active_db
    }

    /// Drop a database
    pub fn drop_database(&mut self, name: &str) -> Result<(), EngineError> {
        if name == "default" {
            return Err(EngineError::InvalidOp(
                "Cannot drop the 'default' database".to_string(),
            ));
        }
        if !self.databases.contains_key(name) {
            return Err(EngineError::InvalidOp(format!(
                "Database '{}' not found",
                name
            )));
        }

        // Remove from memory
        if self.active_db == name {
            self.active_db = "default".to_string();
        }
        self.databases.remove(name);

        // Remove from disk
        let db_path = self.config.storage.data_dir.join(name);
        if db_path.exists() {
            std::fs::remove_dir_all(&db_path).map_err(|e| {
                EngineError::InvalidOp(format!("Failed to remove DB directory: {}", e))
            })?;
        }

        Ok(())
    }

    /// List all databases
    pub fn list_databases(&self) -> Vec<String> {
        self.databases.keys().cloned().collect()
    }

    // Delegate methods to active instance
    pub fn insert_named(
        &mut self,
        name: impl Into<String>,
        shape: Shape,
        data: Vec<f32>,
    ) -> Result<(), EngineError> {
        self.active_instance_mut().insert_named(name, shape, data)
    }

    pub fn insert_named_with_kind(
        &mut self,
        name: impl Into<String>,
        shape: Shape,
        data: Vec<f32>,
        kind: TensorKind,
    ) -> Result<(), EngineError> {
        self.active_instance_mut()
            .insert_named_with_kind(name, shape, data, kind)
    }

    pub fn get(&self, name: &str) -> Result<&Tensor, EngineError> {
        self.active_instance().get(name)
    }

    pub fn register_tensor_dataset(&mut self, ds: crate::core::dataset::Dataset) {
        self.active_instance_mut().register_tensor_dataset(ds);
    }

    pub fn register_dataset_var(&mut self, var_name: String, ds_name: String) {
        self.active_instance_mut()
            .register_dataset_var(var_name, ds_name);
    }

    pub fn add_column_to_tensor_dataset(
        &mut self,
        ds_name: &str,
        col_name: &str,
        tensor_var: &str,
    ) -> Result<(), EngineError> {
        self.active_instance_mut()
            .add_column_to_tensor_dataset(ds_name, col_name, tensor_var)
    }

    pub fn get_tensor_dataset(&self, var_or_name: &str) -> Option<&crate::core::dataset::Dataset> {
        let instance = self.active_instance();
        let ds_name = instance
            .dataset_vars
            .get(var_or_name)
            .map(|s| s.as_str())
            .unwrap_or(var_or_name);
        instance.tensor_datasets.get(ds_name)
    }

    pub fn materialize_tensor_dataset(
        &self,
        name: &str,
    ) -> Result<crate::core::dataset_legacy::Dataset, EngineError> {
        self.active_instance().materialize_tensor_dataset(name)
    }

    pub fn verify_tensor_dataset(&self, ds_name_or_var: &str) -> Result<Vec<String>, EngineError> {
        self.active_instance().verify_tensor_dataset(ds_name_or_var)
    }

    pub fn get_lineage_tree(&self, name: &str) -> Result<LineageNode, EngineError> {
        self.active_instance().get_lineage_tree(name)
    }

    pub fn remove_tensor(&mut self, name: &str) -> bool {
        self.active_instance_mut().remove_tensor(name)
    }

    pub fn eval_unary(
        &mut self,
        ctx: &mut ExecutionContext,
        output_name: impl Into<String>,
        input_name: &str,
        op: UnaryOp,
    ) -> Result<(), EngineError> {
        self.active_instance_mut()
            .eval_unary(ctx, output_name, input_name, op)
    }

    pub fn eval_binary(
        &mut self,
        ctx: &mut ExecutionContext,
        output_name: impl Into<String>,
        left_name: &str,
        right_name: &str,
        op: BinaryOp,
    ) -> Result<(), EngineError> {
        self.active_instance_mut()
            .eval_binary(ctx, output_name, left_name, right_name, op)
    }

    pub fn list_names(&self) -> Vec<String> {
        self.active_instance().list_names()
    }

    pub fn eval_matmul(
        &mut self,
        ctx: &mut ExecutionContext,
        output_name: impl Into<String>,
        left_name: &str,
        right_name: &str,
    ) -> Result<(), EngineError> {
        self.active_instance_mut()
            .eval_matmul(ctx, output_name, left_name, right_name)
    }

    pub fn eval_lazy_binary(
        &mut self,
        ctx: &mut ExecutionContext,
        output_name: &str,
        left_name: &str,
        right_name: &str,
        op: BinaryOp,
    ) -> Result<(), EngineError> {
        self.active_instance_mut()
            .eval_lazy_binary(ctx, output_name, left_name, right_name, op)
    }

    pub fn eval_lazy_unary(
        &mut self,
        ctx: &mut ExecutionContext,
        output_name: &str,
        input_name: &str,
        op: UnaryOp,
    ) -> Result<(), EngineError> {
        self.active_instance_mut()
            .eval_lazy_unary(ctx, output_name, input_name, op)
    }

    pub fn eval_lazy_matmul(
        &mut self,
        ctx: &mut ExecutionContext,
        output_name: &str,
        left_name: &str,
        right_name: &str,
    ) -> Result<(), EngineError> {
        self.active_instance_mut()
            .eval_lazy_matmul(ctx, output_name, left_name, right_name)
    }

    pub fn evaluate(&mut self, name: &str) -> Result<(), EngineError> {
        self.active_instance_mut().evaluate(name)
    }

    pub fn eval_reshape(
        &mut self,
        ctx: &mut ExecutionContext,
        output_name: impl Into<String>,
        input_name: &str,
        new_shape: Shape,
    ) -> Result<(), EngineError> {
        self.active_instance_mut()
            .eval_reshape(ctx, output_name, input_name, new_shape)
    }

    pub fn eval_stack(
        &mut self,
        ctx: &mut ExecutionContext,
        output_name: impl Into<String>,
        input_names: Vec<&str>,
        axis: usize,
    ) -> Result<(), EngineError> {
        self.active_instance_mut()
            .eval_stack(ctx, output_name, input_names, axis)
    }

    pub fn create_dataset(
        &mut self,
        name: String,
        schema: Arc<Schema>,
    ) -> Result<DatasetId, EngineError> {
        self.active_instance_mut().create_dataset(name, schema)
    }

    pub fn get_dataset(&self, name: &str) -> Result<&Dataset, EngineError> {
        self.active_instance().get_dataset(name)
    }

    pub fn get_dataset_mut(&mut self, name: &str) -> Result<&mut Dataset, EngineError> {
        self.active_instance_mut().get_dataset_mut(name)
    }

    pub fn insert_row(&mut self, dataset_name: &str, tuple: Tuple) -> Result<(), EngineError> {
        self.active_instance_mut().insert_row(dataset_name, tuple)
    }

    pub fn list_dataset_names(&self) -> Vec<String> {
        self.active_instance().list_dataset_names()
    }

    /// Remove a dataset by name (used for CTE cleanup)
    pub fn remove_dataset(&mut self, name: &str) -> Result<(), EngineError> {
        self.active_instance_mut().remove_dataset(name)
    }

    pub fn alter_dataset_add_column(
        &mut self,
        dataset_name: &str,
        column_name: String,
        value_type: crate::core::value::ValueType,
        default_value: crate::core::value::Value,
        nullable: bool,
    ) -> Result<(), EngineError> {
        self.active_instance_mut().alter_dataset_add_column(
            dataset_name,
            column_name,
            value_type,
            default_value,
            nullable,
        )
    }

    pub fn alter_dataset_add_computed_column(
        &mut self,
        dataset_name: &str,
        column_name: String,
        value_type: crate::core::value::ValueType,
        computed_values: Vec<crate::core::value::Value>,
        expression: crate::query::logical::Expr,
        lazy: bool,
    ) -> Result<(), EngineError> {
        self.active_instance_mut()
            .alter_dataset_add_computed_column(
                dataset_name,
                column_name,
                value_type,
                computed_values,
                expression,
                lazy,
            )
    }

    pub fn materialize_lazy_columns(&mut self, dataset_name: &str) -> Result<(), EngineError> {
        self.active_instance_mut()
            .materialize_lazy_columns(dataset_name)
    }

    pub fn eval_index(
        &mut self,
        ctx: &mut ExecutionContext,
        output_name: impl Into<String>,
        tensor_name: &str,
        indices: Vec<usize>,
    ) -> Result<(), EngineError> {
        self.active_instance_mut()
            .eval_index(ctx, output_name, tensor_name, indices)
    }

    pub fn eval_slice(
        &mut self,
        ctx: &mut ExecutionContext,
        output_name: impl Into<String>,
        tensor_name: &str,
        specs: Vec<super::kernels::SliceSpec>,
    ) -> Result<(), EngineError> {
        self.active_instance_mut()
            .eval_slice(ctx, output_name, tensor_name, specs)
    }

    pub fn eval_field_access(
        &mut self,
        ctx: &mut ExecutionContext,
        output_name: impl Into<String>,
        tuple_name: &str,
        field_name: &str,
    ) -> Result<(), EngineError> {
        self.active_instance_mut()
            .eval_field_access(ctx, output_name, tuple_name, field_name)
    }

    pub fn eval_column_access(
        &mut self,
        ctx: &mut ExecutionContext,
        output_name: impl Into<String>,
        dataset_name: &str,
        column_name: &str,
    ) -> Result<(), EngineError> {
        self.active_instance_mut()
            .eval_column_access(ctx, output_name, dataset_name, column_name)
    }

    pub fn create_index(
        &mut self,
        dataset_name: &str,
        column_name: &str,
    ) -> Result<(), EngineError> {
        self.active_instance_mut()
            .create_index(dataset_name, column_name)
    }

    pub fn create_vector_index(
        &mut self,
        dataset_name: &str,
        column_name: &str,
    ) -> Result<(), EngineError> {
        self.active_instance_mut()
            .create_vector_index(dataset_name, column_name)
    }

    pub fn list_indices(&self) -> Vec<(String, String, String)> {
        self.active_instance().list_indices()
    }

    pub fn set_dataset_metadata(
        &mut self,
        name: &str,
        key: String,
        value: String,
    ) -> Result<(), EngineError> {
        self.active_instance_mut()
            .set_dataset_metadata(name, key, value)
    }

    /// Execute a DSL command with an execution context for resource management
    /// This is an opt-in API that provides arena allocation and automatic cleanup
    pub fn execute_with_context(
        &mut self,
        ctx: &mut crate::engine::context::ExecutionContext,
        command: &str,
    ) -> Result<crate::dsl::DslOutput, crate::dsl::DslError> {
        use crate::dsl::execute_line_with_context;

        // For Phase 1, just call existing implementation
        // Phase 2 will use ctx for arena allocation
        let result = execute_line_with_context(self, command, 1, Some(ctx))?;

        // Cleanup any tracked resources
        self.cleanup_context_resources(ctx);

        Ok(result)
    }

    /// Clean up resources tracked by an execution context
    /// Note: For Phase 1, we just clear the tracking. Full cleanup will be implemented
    /// in Phase 2 when we add proper resource management to the stores.
    pub(crate) fn cleanup_context_resources(
        &mut self,
        ctx: &mut crate::engine::context::ExecutionContext,
    ) {
        // For now, just clear the tracked resources
        // In Phase 2, we'll implement proper removal when stores support it
        ctx.clear_tracked();
    }

    /// Resets the active session (clear all data in the active database)
    pub fn reset_session(&mut self) {
        self.active_instance_mut().reset();
        self.pipelines.clear();
    }
}

impl DatabaseInstance {
    /// Inserta un tensor y lo asocia a un nombre (modo NORMAL por defecto)
    pub fn insert_named(
        &mut self,
        name: impl Into<String>,
        shape: Shape,
        data: Vec<f32>,
    ) -> Result<(), EngineError> {
        self.insert_named_with_kind(name, shape, data, TensorKind::Normal)
    }

    /// Inserta un tensor con un "kind" explícito (NORMAL o STRICT)
    pub fn insert_named_with_kind(
        &mut self,
        name: impl Into<String>,
        shape: Shape,
        data: Vec<f32>,
        kind: TensorKind,
    ) -> Result<(), EngineError> {
        let id = self.store.insert_tensor(shape, data)?;
        self.names.insert(name.into(), NameEntry { id, kind });
        Ok(())
    }

    /// Inserta un objeto Tensor completo (preserva metadata)
    pub fn insert_tensor_object(
        &mut self,
        name: impl Into<String>,
        tensor: Tensor,
    ) -> Result<(), EngineError> {
        let name_str = name.into();
        let id = tensor.id;
        self.store.insert_existing_tensor(tensor)?;
        self.names.insert(
            name_str,
            NameEntry {
                id,
                kind: TensorKind::Normal,
            },
        );
        Ok(())
    }

    /// Vincular un nuevo nombre a un recurso existente (alias)
    pub fn bind_resource(&mut self, alias: &str, source: &str) -> Result<(), EngineError> {
        // Intentar resolver como tensor
        if let Some(entry) = self.names.get(source) {
            let id = entry.id;
            let kind = entry.kind;
            self.names.insert(alias.to_string(), NameEntry { id, kind });
            return Ok(());
        }

        // Intentar resolver como dataset (legacy)
        if let Ok(ds) = self.dataset_store.get_by_name(source) {
            let mut ds_new = ds.clone();
            // Generar nuevo ID para el alias en el store legacy (o mantener el mismo?
            // El store legacy asocia 1 ID a 1 nombre en su mapa `names`.
            // Para tener 2 nombres para el mismo dataset, necesitamos insertarlo de nuevo con otro nombre.
            // Opcionalmente podemos generar un nuevo ID o usar el mismo si el store lo permite.
            // El store legacy usa DatasetId como llave primaria.
            let new_id = self.dataset_store.gen_id();
            ds_new.id = new_id;
            self.dataset_store
                .insert(ds_new, Some(alias.to_string()))
                .map_err(EngineError::DatasetError)?;
            return Ok(());
        }

        // Intentar resolver como tensor-first dataset
        if let Some(ds) = self.tensor_datasets.get(source) {
            let mut new_ds = ds.clone();
            new_ds.name = alias.to_string();
            let _ = self.tensor_datasets.register(new_ds);
            return Ok(());
        }

        Err(EngineError::NameNotFound(source.to_string()))
    }

    pub fn get_lineage_tree(&self, name: &str) -> Result<LineageNode, EngineError> {
        let entry = self
            .names
            .get(name)
            .ok_or_else(|| EngineError::NameNotFound(name.to_string()))?;
        self.resolve_lineage_node(entry.id, Some(name.to_string()))
    }

    fn resolve_lineage_node(
        &self,
        id: TensorId,
        name: Option<String>,
    ) -> Result<LineageNode, EngineError> {
        let tensor = self.store.get(id)?;
        let mut inputs = Vec::new();

        let operation = if let Some(lineage) = &tensor.metadata.lineage {
            for input_id in &lineage.inputs {
                // Find name for input if exists in this instance
                let input_name = self
                    .names
                    .iter()
                    .find(|(_, entry)| entry.id == *input_id)
                    .map(|(n, _)| n.clone());
                inputs.push(self.resolve_lineage_node(*input_id, input_name)?);
            }
            lineage.operation.clone()
        } else {
            "ROOT".to_string()
        };

        Ok(LineageNode {
            tensor_id: id,
            name,
            operation,
            inputs,
        })
    }

    pub fn find_referencing_datasets(&self, tensor_id: TensorId) -> Vec<String> {
        let mut referencers = Vec::new();
        for (ds_name, ds) in self.tensor_datasets.datasets() {
            for reference in ds.columns.values() {
                if let crate::core::dataset::ResourceReference::Tensor { id } = reference {
                    if *id == tensor_id {
                        referencers.push(ds_name.clone());
                        break;
                    }
                }
            }
        }
        referencers
    }

    /// Obtiene un tensor por nombre
    pub fn get(&self, name: &str) -> Result<&Tensor, EngineError> {
        let entry = self
            .names
            .get(name)
            .ok_or_else(|| EngineError::NameNotFound(name.to_string()))?;
        Ok(self.store.get(entry.id)?)
    }

    /// Obtiene (tensor, kind) por nombre (para decisiones de ejecución)
    pub(crate) fn get_with_kind(&self, name: &str) -> Result<(&Tensor, TensorKind), EngineError> {
        let entry = self
            .names
            .get(name)
            .ok_or_else(|| EngineError::NameNotFound(name.to_string()))?;
        let t = self.store.get(entry.id)?;
        Ok((t, entry.kind))
    }

    /// Evalúa operación unaria: SCALE, etc.
    pub fn eval_unary(
        &mut self,
        ctx: &mut ExecutionContext,
        output_name: impl Into<String>,
        input_name: &str,
        op: UnaryOp,
    ) -> Result<(), EngineError> {
        let (in_tensor_ref, in_kind) = self.get_with_kind(input_name)?;
        let in_tensor = in_tensor_ref.clone();
        let new_id = self.store.gen_id();

        let mut result = match op {
            UnaryOp::Scale(s) => self
                .backend
                .scale(ctx, &in_tensor, s, new_id)
                .map_err(EngineError::InvalidOp)?,
            UnaryOp::Normalize => self
                .backend
                .normalize(ctx, &in_tensor, new_id)
                .map_err(EngineError::InvalidOp)?,
            UnaryOp::Transpose => self
                .backend
                .transpose(ctx, &in_tensor, new_id)
                .map_err(EngineError::InvalidOp)?,
            UnaryOp::Flatten => self
                .backend
                .flatten(ctx, &in_tensor, new_id)
                .map_err(EngineError::InvalidOp)?,
            UnaryOp::Sum => self
                .backend
                .sum(ctx, &in_tensor, new_id)
                .map_err(EngineError::InvalidOp)?,
            UnaryOp::Mean => self
                .backend
                .mean(ctx, &in_tensor, new_id)
                .map_err(EngineError::InvalidOp)?,
            UnaryOp::Stdev => self
                .backend
                .stdev(ctx, &in_tensor, new_id)
                .map_err(EngineError::InvalidOp)?,
        };

        // Attach lineage
        let lineage = Lineage {
            execution_id: ctx.execution_id(),
            operation: op.to_string(),
            inputs: vec![in_tensor.id],
        };
        result.metadata = Arc::new(TensorMetadata::new(new_id, None).with_lineage(lineage));

        let out_id = self.store.insert_existing_tensor(result)?;
        self.names.insert(
            output_name.into(),
            NameEntry {
                id: out_id,
                kind: in_kind, // hereda el modo del input
            },
        );
        Ok(())
    }

    /// Evalúa operación binaria: ADD, SUBTRACT, CORRELATE, SIMILARITY, DISTANCE
    pub fn eval_binary(
        &mut self,
        ctx: &mut crate::engine::context::ExecutionContext,
        output_name: impl Into<String>,
        left_name: &str,
        right_name: &str,
        op: BinaryOp,
    ) -> Result<(), EngineError> {
        let (a_ref, kind_a) = self.get_with_kind(left_name)?;
        let (b_ref, kind_b) = self.get_with_kind(right_name)?;
        let a = a_ref.clone();
        let b = b_ref.clone();
        let new_id = self.store.gen_id();

        // Si alguno es STRICT, el resultado también es STRICT.
        let out_kind = match (kind_a, kind_b) {
            (TensorKind::Strict, _) | (_, TensorKind::Strict) => TensorKind::Strict,
            _ => TensorKind::Normal,
        };

        let mut result_tensor = match op {
            BinaryOp::Add => self
                .backend
                .add(ctx, &a, &b, new_id)
                .map_err(EngineError::InvalidOp)?,
            BinaryOp::Subtract => self
                .backend
                .sub(ctx, &a, &b, new_id)
                .map_err(EngineError::InvalidOp)?,
            BinaryOp::Multiply => self
                .backend
                .multiply(ctx, &a, &b, new_id)
                .map_err(EngineError::InvalidOp)?,
            BinaryOp::Divide => self
                .backend
                .divide(ctx, &a, &b, new_id)
                .map_err(EngineError::InvalidOp)?,
            BinaryOp::Correlate => {
                let value = self
                    .backend
                    .dot(ctx, &a, &b)
                    .map_err(EngineError::InvalidOp)?;
                let shape = Shape::new(Vec::<usize>::new());
                let data = vec![value];
                let metadata = crate::core::tensor::TensorMetadata::new(new_id, None);
                Tensor::new(new_id, shape, data, metadata).map_err(EngineError::InvalidOp)?
            }
            BinaryOp::Similarity => {
                let value = self
                    .backend
                    .cosine_similarity(ctx, &a, &b)
                    .map_err(EngineError::InvalidOp)?;
                let shape = Shape::new(Vec::<usize>::new());
                let data = vec![value];
                let metadata = crate::core::tensor::TensorMetadata::new(new_id, None);
                Tensor::new(new_id, shape, data, metadata).map_err(EngineError::InvalidOp)?
            }
            BinaryOp::Distance => {
                let value = self
                    .backend
                    .distance(ctx, &a, &b)
                    .map_err(EngineError::InvalidOp)?;
                let shape = Shape::new(Vec::<usize>::new());
                let data = vec![value];
                let lineage = Lineage {
                    execution_id: ctx.execution_id(),
                    operation: op.to_string(),
                    inputs: vec![a.id, b.id],
                };
                let metadata =
                    crate::core::tensor::TensorMetadata::new(new_id, None).with_lineage(lineage);
                Tensor::new(new_id, shape, data, metadata).map_err(EngineError::InvalidOp)?
            }
        };

        // Attach lineage for backend-returned tensors
        if result_tensor.metadata.lineage.is_none() {
            let lineage = Lineage {
                execution_id: ctx.execution_id(),
                operation: op.to_string(),
                inputs: vec![a.id, b.id],
            };
            result_tensor.metadata =
                Arc::new(TensorMetadata::new(new_id, None).with_lineage(lineage));
        }

        let out_id = self.store.insert_existing_tensor(result_tensor)?;
        self.names.insert(
            output_name.into(),
            NameEntry {
                id: out_id,
                kind: out_kind,
            },
        );
        Ok(())
    }

    /// Para debug: todos los nombres registrados
    pub fn list_names(&self) -> Vec<String> {
        self.names.keys().cloned().collect()
    }

    /// Matrix multiplication: C = MATMUL A B
    pub fn eval_matmul(
        &mut self,
        ctx: &mut crate::engine::context::ExecutionContext,
        output_name: impl Into<String>,
        left_name: &str,
        right_name: &str,
    ) -> Result<(), EngineError> {
        let (a_ref, kind_a) = self.get_with_kind(left_name)?;
        let (b_ref, kind_b) = self.get_with_kind(right_name)?;
        let a = a_ref.clone();
        let b = b_ref.clone();
        let new_id = self.store.gen_id();

        let mut result = self
            .backend
            .matmul(ctx, &a, &b, new_id)
            .map_err(EngineError::InvalidOp)?;

        // Attach lineage
        let lineage = Lineage {
            execution_id: ctx.execution_id(),
            operation: "MATMUL".to_string(),
            inputs: vec![a.id, b.id],
        };
        result.metadata = Arc::new(TensorMetadata::new(new_id, None).with_lineage(lineage));

        let out_kind = match (kind_a, kind_b) {
            (TensorKind::Strict, _) | (_, TensorKind::Strict) => TensorKind::Strict,
            _ => TensorKind::Normal,
        };

        let out_id = self.store.insert_existing_tensor(result)?;
        self.names.insert(
            output_name.into(),
            NameEntry {
                id: out_id,
                kind: out_kind,
            },
        );
        Ok(())
    }

    /// Reshape tensor: B = RESHAPE A TO [new_shape]
    pub fn eval_reshape(
        &mut self,
        ctx: &mut ExecutionContext,
        output_name: impl Into<String>,
        input_name: &str,
        new_shape: Shape,
    ) -> Result<(), EngineError> {
        let (in_tensor_ref, in_kind) = self.get_with_kind(input_name)?;
        let in_tensor = in_tensor_ref.clone();
        let new_id = self.store.gen_id();

        let mut result = self
            .backend
            .reshape(ctx, &in_tensor, new_shape, new_id)
            .map_err(EngineError::InvalidOp)?;

        // Attach lineage
        let lineage = Lineage {
            execution_id: ctx.execution_id(),
            operation: "RESHAPE".to_string(),
            inputs: vec![in_tensor.id],
        };
        result.metadata = Arc::new(TensorMetadata::new(new_id, None).with_lineage(lineage));

        let out_id = self.store.insert_existing_tensor(result)?;
        self.names.insert(
            output_name.into(),
            NameEntry {
                id: out_id,
                kind: in_kind,
            },
        );
        Ok(())
    }

    /// Stack tensors: C = STACK A B
    pub fn eval_stack(
        &mut self,
        ctx: &mut ExecutionContext,
        output_name: impl Into<String>,
        input_names: Vec<&str>,
        axis: usize,
    ) -> Result<(), EngineError> {
        // Collect tensors
        let mut tensors = Vec::with_capacity(input_names.len());
        let mut kind = TensorKind::Normal;

        for name in input_names {
            let (t, k) = self.get_with_kind(name)?;
            if matches!(k, TensorKind::Strict) {
                kind = TensorKind::Strict;
            }
            tensors.push(t.clone());
        }

        let tensor_refs: Vec<&Tensor> = tensors.iter().collect();
        let new_id = self.store.gen_id();

        let mut result = self
            .backend
            .stack(ctx, &tensor_refs, axis, new_id)
            .map_err(EngineError::InvalidOp)?;

        // Attach lineage
        let lineage = Lineage {
            execution_id: ctx.execution_id(),
            operation: format!("STACK(axis={})", axis),
            inputs: tensors.iter().map(|t| t.id).collect(),
        };
        result.metadata = Arc::new(TensorMetadata::new(new_id, None).with_lineage(lineage));

        let out_id = self.store.insert_existing_tensor(result)?;
        self.names
            .insert(output_name.into(), NameEntry { id: out_id, kind });
        Ok(())
    }

    // ===== Dataset Management Methods =====

    /// Create a new dataset with schema
    pub fn create_dataset(
        &mut self,
        name: String,
        schema: Arc<Schema>,
    ) -> Result<DatasetId, EngineError> {
        let id = self.dataset_store.gen_id();
        let dataset = Dataset::new(id, schema, Some(name.clone()));
        self.dataset_store
            .insert(dataset, Some(name))
            .map_err(EngineError::from)
    }

    /// Get dataset by name
    pub fn get_dataset(&self, name: &str) -> Result<&Dataset, EngineError> {
        self.dataset_store
            .get_by_name(name)
            .map_err(|_| EngineError::DatasetNotFound(name.to_string()))
    }

    /// Get mutable dataset by name
    pub fn get_dataset_mut(&mut self, name: &str) -> Result<&mut Dataset, EngineError> {
        self.dataset_store
            .get_mut_by_name(name)
            .map_err(|_| EngineError::DatasetNotFound(name.to_string()))
    }

    /// Insert row into dataset
    pub fn insert_row(&mut self, dataset_name: &str, tuple: Tuple) -> Result<(), EngineError> {
        let dataset = self.get_dataset_mut(dataset_name)?;
        dataset.add_row(tuple).map_err(EngineError::InvalidOp)
    }

    /// List all dataset names
    pub fn list_dataset_names(&self) -> Vec<String> {
        self.dataset_store.list_names()
    }

    /// Remove a dataset by name (used for CTE cleanup)
    pub fn remove_dataset(&mut self, name: &str) -> Result<(), EngineError> {
        self.dataset_store
            .remove_by_name(name)
            .map(|_| ())
            .map_err(|_| EngineError::DatasetNotFound(name.to_string()))
    }

    /// Add a column to an existing dataset
    pub fn alter_dataset_add_column(
        &mut self,
        dataset_name: &str,
        column_name: String,
        value_type: crate::core::value::ValueType,
        default_value: crate::core::value::Value,
        nullable: bool,
    ) -> Result<(), EngineError> {
        let dataset = self.get_dataset_mut(dataset_name)?;
        dataset
            .add_column(column_name, value_type, default_value, nullable)
            .map_err(EngineError::InvalidOp)
    }

    /// Add a computed column to an existing dataset
    pub fn alter_dataset_add_computed_column(
        &mut self,
        dataset_name: &str,
        column_name: String,
        value_type: crate::core::value::ValueType,
        computed_values: Vec<crate::core::value::Value>,
        expression: crate::query::logical::Expr,
        lazy: bool,
    ) -> Result<(), EngineError> {
        let dataset = self.get_dataset_mut(dataset_name)?;
        dataset
            .add_computed_column(column_name, value_type, computed_values, expression, lazy)
            .map_err(EngineError::InvalidOp)
    }

    /// Materialize lazy columns in a dataset
    pub fn materialize_lazy_columns(&mut self, dataset_name: &str) -> Result<(), EngineError> {
        let dataset = self.get_dataset_mut(dataset_name)?;
        dataset
            .materialize_lazy_columns()
            .map_err(EngineError::InvalidOp)
    }

    /// Index into a tensor: output = tensor[indices]
    pub fn eval_index(
        &mut self,
        ctx: &mut ExecutionContext,
        output_name: impl Into<String>,
        tensor_name: &str,
        indices: Vec<usize>,
    ) -> Result<(), EngineError> {
        let (tensor_ref, kind) = self.get_with_kind(tensor_name)?;
        let tensor = tensor_ref.clone();
        let new_id = self.store.gen_id();

        let mut result = super::kernels::index_to_scalar(&tensor, &indices, new_id)
            .map_err(EngineError::InvalidOp)?;

        // Attach lineage
        let lineage = Lineage {
            execution_id: ctx.execution_id(),
            operation: format!("INDEX{:?}", indices),
            inputs: vec![tensor.id],
        };
        result.metadata = Arc::new(TensorMetadata::new(new_id, None).with_lineage(lineage));

        let out_id = self.store.insert_existing_tensor(result)?;
        self.names
            .insert(output_name.into(), NameEntry { id: out_id, kind });
        Ok(())
    }

    /// Slice a tensor: output = tensor[slice_specs]
    pub fn eval_slice(
        &mut self,
        ctx: &mut ExecutionContext,
        output_name: impl Into<String>,
        tensor_name: &str,
        specs: Vec<super::kernels::SliceSpec>,
    ) -> Result<(), EngineError> {
        let (tensor_ref, kind) = self.get_with_kind(tensor_name)?;
        let tensor = tensor_ref.clone();
        let new_id = self.store.gen_id();

        let mut result =
            super::kernels::slice_multi(&tensor, &specs, new_id).map_err(EngineError::InvalidOp)?;

        // Attach lineage
        let lineage = Lineage {
            execution_id: ctx.execution_id(),
            operation: format!("SLICE{:?}", specs),
            inputs: vec![tensor.id],
        };
        result.metadata = Arc::new(TensorMetadata::new(new_id, None).with_lineage(lineage));

        let out_id = self.store.insert_existing_tensor(result)?;
        self.names
            .insert(output_name.into(), NameEntry { id: out_id, kind });
        Ok(())
    }

    /// Access a tuple field: output = tuple.field
    /// Returns the field value as a scalar tensor
    pub fn eval_field_access(
        &mut self,
        ctx: &mut ExecutionContext,
        output_name: impl Into<String>,
        tuple_name: &str,
        field_name: &str,
    ) -> Result<(), EngineError> {
        // For now, we'll store tuples as datasets with a single row
        // This is a simplification - in the future we might have dedicated tuple storage
        let dataset = self.get_dataset(tuple_name)?;

        if dataset.rows.is_empty() {
            return Err(EngineError::InvalidOp(format!(
                "Cannot access field of empty dataset '{}'",
                tuple_name
            )));
        }

        // Get the first row (treating dataset as tuple)
        let row = &dataset.rows[0];
        let value = row
            .get(field_name)
            .ok_or_else(|| EngineError::InvalidOp(format!("Field '{}' not found", field_name)))?
            .clone(); // Clone to avoid borrow issues

        // Convert value to scalar tensor
        let new_id = self.store.gen_id();
        let shape = crate::core::tensor::Shape::new(vec![]);

        let tensor_data = match value {
            crate::core::value::Value::Float(f) => vec![f],
            crate::core::value::Value::Int(i) => vec![i as f32],
            crate::core::value::Value::Bool(b) => vec![if b { 1.0 } else { 0.0 }],
            _ => {
                return Err(EngineError::InvalidOp(format!(
                    "Cannot convert field '{}' to tensor",
                    field_name
                )))
            }
        };

        let lineage = Lineage {
            execution_id: ctx.execution_id(),
            operation: format!("FIELD_ACCESS({})", field_name),
            inputs: Vec::new(),
        };
        let metadata = crate::core::tensor::TensorMetadata::new(new_id, None).with_lineage(lineage);
        let tensor = crate::core::tensor::Tensor::new(new_id, shape, tensor_data, metadata)
            .map_err(EngineError::InvalidOp)?;

        let out_id = self.store.insert_existing_tensor(tensor)?;
        self.names.insert(
            output_name.into(),
            NameEntry {
                id: out_id,
                kind: TensorKind::Normal,
            },
        );
        Ok(())
    }

    /// Extract a column from a dataset: output = dataset.column
    /// Returns the column as a vector tensor
    pub fn eval_column_access(
        &mut self,
        ctx: &mut ExecutionContext,
        output_name: impl Into<String>,
        var_or_name: &str,
        column_name: &str,
    ) -> Result<(), EngineError> {
        // 1. Resolve dataset name
        let ds_name = self
            .dataset_vars
            .get(var_or_name)
            .map(|s| s.as_str())
            .unwrap_or(var_or_name);

        // 2. Try as tensor-first dataset (Zero-copy path)
        if let Some(ds) = self.tensor_datasets.get(ds_name) {
            if let Some(reference) = ds.get_reference(column_name) {
                let graph = crate::core::dataset::DatasetGraph::new(&self.tensor_datasets);
                let tensor_id = graph.resolve_to_tensor(reference).map_err(|e| {
                    EngineError::InvalidOp(format!("Dependency resolution error: {}", e))
                })?;

                // Determine kind (Normal/Strict) - for now default to Normal
                self.names.insert(
                    output_name.into(),
                    NameEntry {
                        id: tensor_id,
                        kind: TensorKind::Normal,
                    },
                );
                return Ok(());
            } else {
                return Err(EngineError::InvalidOp(format!(
                    "Column '{}' not found in tensor dataset '{}'",
                    column_name, ds_name
                )));
            }
        }

        // 3. Try legacy dataset (Materialization path)
        let dataset = self.get_dataset(ds_name)?.clone();
        let column_values = dataset
            .get_column(column_name)
            .map_err(EngineError::InvalidOp)?;

        // Convert column values to tensor
        let new_id = self.store.gen_id();
        let shape = crate::core::tensor::Shape::new(vec![column_values.len()]);

        let tensor_data: Result<Vec<f32>, String> = column_values
            .iter()
            .map(|v| match v {
                crate::core::value::Value::Float(f) => Ok(*f),
                crate::core::value::Value::Int(i) => Ok(*i as f32),
                crate::core::value::Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
                _ => Err(format!("Cannot convert value to tensor: {:?}", v)),
            })
            .collect();

        let tensor_data = tensor_data.map_err(EngineError::InvalidOp)?;
        let lineage = Lineage {
            execution_id: ctx.execution_id(),
            operation: format!("COLUMN_ACCESS({})", column_name),
            inputs: Vec::new(),
        };
        let metadata = crate::core::tensor::TensorMetadata::new(new_id, None).with_lineage(lineage);
        let tensor = crate::core::tensor::Tensor::new(new_id, shape, tensor_data, metadata)
            .map_err(EngineError::InvalidOp)?;

        let out_id = self.store.insert_existing_tensor(tensor)?;
        self.names.insert(
            output_name.into(),
            NameEntry {
                id: out_id,
                kind: TensorKind::Normal,
            },
        );
        Ok(())
    }

    /// Create a standard hash index on a dataset column
    pub fn create_index(
        &mut self,
        dataset_name: &str,
        column_name: &str,
    ) -> Result<(), EngineError> {
        let dataset = self.get_dataset_mut(dataset_name)?;
        let index = Box::new(crate::core::index::hash::HashIndex::new());
        dataset
            .create_index(column_name.to_string(), index)
            .map_err(EngineError::InvalidOp)
    }

    /// Create a vector index on a dataset column
    pub fn create_vector_index(
        &mut self,
        dataset_name: &str,
        column_name: &str,
    ) -> Result<(), EngineError> {
        let dataset = self.get_dataset_mut(dataset_name)?;
        let index = Box::new(crate::core::index::vector::VectorIndex::new());
        dataset
            .create_index(column_name.to_string(), index)
            .map_err(EngineError::InvalidOp)
    }

    /// Get all indices info
    pub fn list_indices(&self) -> Vec<(String, String, String)> {
        let mut result = Vec::new();
        for name in self.dataset_store.list_names() {
            if let Ok(ds) = self.get_dataset(&name) {
                for (col, idx) in &ds.indices {
                    let type_str = match idx.index_type() {
                        crate::core::index::IndexType::Hash => "HASH",
                        crate::core::index::IndexType::Vector => "VECTOR",
                    };
                    result.push((name.clone(), col.clone(), type_str.to_string()));
                }
            }
        }
        result
    }

    pub fn eval_lazy_binary(
        &mut self,
        _ctx: &mut ExecutionContext,
        output_name: &str,
        left_name: &str,
        right_name: &str,
        op: BinaryOp,
    ) -> Result<(), EngineError> {
        let left_expr = self.get_expression_or_literal(left_name)?;
        let right_expr = self.get_expression_or_literal(right_name)?;

        let expr = match op {
            BinaryOp::Add => {
                crate::core::tensor::Expression::Add(Box::new(left_expr), Box::new(right_expr))
            }
            BinaryOp::Subtract => {
                crate::core::tensor::Expression::Sub(Box::new(left_expr), Box::new(right_expr))
            }
            BinaryOp::Multiply => {
                crate::core::tensor::Expression::Multiply(Box::new(left_expr), Box::new(right_expr))
            }
            BinaryOp::Divide => {
                crate::core::tensor::Expression::Divide(Box::new(left_expr), Box::new(right_expr))
            }
            _ => return Err(EngineError::InvalidOp(format!("Lazy {} not supported", op))),
        };

        self.register_lazy(output_name, expr)
    }

    pub fn eval_lazy_unary(
        &mut self,
        _ctx: &mut ExecutionContext,
        output_name: &str,
        input_name: &str,
        op: UnaryOp,
    ) -> Result<(), EngineError> {
        let inner_expr = self.get_expression_or_literal(input_name)?;

        let expr = match op {
            UnaryOp::Scale(s) => {
                crate::core::tensor::Expression::ScalarMul(Box::new(inner_expr), s)
            }
            UnaryOp::Normalize => crate::core::tensor::Expression::Normalize(Box::new(inner_expr)),
            UnaryOp::Sum => crate::core::tensor::Expression::Sum(Box::new(inner_expr)),
            UnaryOp::Mean => crate::core::tensor::Expression::Mean(Box::new(inner_expr)),
            UnaryOp::Stdev => crate::core::tensor::Expression::Stdev(Box::new(inner_expr)),
            _ => return Err(EngineError::InvalidOp(format!("Lazy {} not supported", op))),
        };

        self.register_lazy(output_name, expr)
    }

    pub fn eval_lazy_matmul(
        &mut self,
        _ctx: &mut ExecutionContext,
        output_name: &str,
        left_name: &str,
        right_name: &str,
    ) -> Result<(), EngineError> {
        let left_expr = self.get_expression_or_literal(left_name)?;
        let right_expr = self.get_expression_or_literal(right_name)?;

        let expr =
            crate::core::tensor::Expression::MatMul(Box::new(left_expr), Box::new(right_expr));
        self.register_lazy(output_name, expr)
    }

    fn get_expression_or_literal(
        &self,
        name: &str,
    ) -> Result<crate::core::tensor::Expression, EngineError> {
        let entry = self
            .names
            .get(name)
            .ok_or_else(|| EngineError::NameNotFound(name.to_string()))?;

        if entry.kind == TensorKind::Lazy {
            let expr = self
                .lazy_store
                .get(&entry.id)
                .ok_or_else(|| EngineError::ExecutionError("Lazy expression missing".into()))?;
            Ok(expr.clone())
        } else {
            let tensor = self.store.get(entry.id).map_err(EngineError::Store)?;
            Ok(crate::core::tensor::Expression::Literal((*tensor).clone()))
        }
    }

    fn register_lazy(
        &mut self,
        name: &str,
        expr: crate::core::tensor::Expression,
    ) -> Result<(), EngineError> {
        let id = TensorId::new();
        self.names.insert(
            name.to_string(),
            NameEntry {
                id,
                kind: TensorKind::Lazy,
            },
        );
        self.lazy_store.insert(id, expr);
        Ok(())
    }

    pub fn evaluate(&mut self, name: &str) -> Result<(), EngineError> {
        let entry = self
            .names
            .get(name)
            .ok_or_else(|| EngineError::NameNotFound(name.to_string()))?;

        if entry.kind != TensorKind::Lazy {
            return Ok(());
        }

        let expr = self
            .lazy_store
            .get(&entry.id)
            .ok_or_else(|| EngineError::ExecutionError("Lazy expression missing".into()))?;

        let timestamp = chrono::Utc::now();
        let materialized = crate::engine::kernels::evaluate_expression(expr, timestamp)
            .map_err(EngineError::ExecutionError)?;

        self.store.insert_existing_tensor(materialized.clone())?;
        self.names.insert(
            name.to_string(),
            NameEntry {
                id: materialized.id,
                kind: TensorKind::Normal,
            },
        );

        Ok(())
    }
}
