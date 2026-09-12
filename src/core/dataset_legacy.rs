use super::tuple::{Schema, Tuple};
use super::value::{Value, ValueType};
use chrono::{DateTime, Utc};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

// Batch processing constants
const BATCH_SIZE: usize = 1024;
const BATCH_PARALLEL_THRESHOLD: usize = 10_000;

/// Unique identifier for datasets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct DatasetId(pub u64);

/// Statistics for a single column
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnStats {
    pub value_type: ValueType,
    pub null_count: usize,
    pub min: Option<Value>,
    pub max: Option<Value>,
}

impl ColumnStats {
    fn empty(value_type: ValueType) -> Self {
        Self {
            value_type,
            null_count: 0,
            min: None,
            max: None,
        }
    }

    /// Incorporate one value into these stats.
    fn merge_value(&mut self, value: &Value) {
        if value.is_null() {
            self.null_count += 1;
            return;
        }

        match &self.min {
            Some(current_min) => {
                if value.compare(current_min) == Some(std::cmp::Ordering::Less) {
                    self.min = Some(value.clone());
                }
            }
            None => self.min = Some(value.clone()),
        }

        match &self.max {
            Some(current_max) => {
                if value.compare(current_max) == Some(std::cmp::Ordering::Greater) {
                    self.max = Some(value.clone());
                }
            }
            None => self.max = Some(value.clone()),
        }
    }

    /// Compute stats for one field across a slice of rows -- used both for
    /// the whole-dataset summary and for a single partition's zone map.
    fn compute(field: &super::tuple::Field, rows: &[Tuple]) -> Self {
        let mut stats = Self::empty(field.value_type.clone());
        for row in rows {
            if let Some(value) = row.get(&field.name) {
                stats.merge_value(value);
            }
        }
        stats
    }
}

/// A zone map over a contiguous slice of `Dataset.rows` ([start, end)):
/// per-column min/max/null-count for just that slice. Lets the query
/// planner prove a whole partition can't satisfy a range predicate (e.g.
/// `WHERE age > 90` against a partition whose `age` column maxes out at 40)
/// and skip it without reading its rows -- see `try_prune_partitions` in
/// `query/planner.rs`. Partition boundaries are the same `BATCH_SIZE`-sized
/// chunks `filter_batched`/`map_batched`/`select_batched` already use; this
/// is read-only metadata over the existing row vector, never a change to
/// row storage or order.
#[derive(Debug, Clone)]
pub struct PartitionStats {
    pub start: usize,
    pub end: usize,
    pub column_stats: HashMap<String, ColumnStats>,
}

/// Metadata about a dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    pub name: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub version: u32,
    pub row_count: usize,
    pub column_stats: HashMap<String, ColumnStats>,
    pub schema: Schema,
    pub extra: HashMap<String, String>,
}

impl DatasetMetadata {
    pub fn new(name: Option<String>, schema: Schema) -> Self {
        let now = Utc::now();
        Self {
            name,
            created_at: now,
            updated_at: now,
            version: 1,
            row_count: 0,
            column_stats: HashMap::new(),
            schema,
            extra: HashMap::new(),
        }
    }

    /// Update statistics based on current rows. Also refreshes `self.schema`
    /// -- this is the metadata's own cached copy of the dataset's schema
    /// (separate from `Dataset.schema` itself), and every schema-changing
    /// mutation (e.g. `add_computed_column`) calls this to sync stats but
    /// was leaving `self.schema` stale, since only `column_stats`/
    /// `row_count` were being recomputed here. A stale cached schema here
    /// silently truncated data on the next `SAVE DATASET` + reload cycle:
    /// `save_legacy_metadata` persists this exact struct to `.meta.json`,
    /// and `ParquetStorage::load_dataset` rebuilds the reloaded dataset
    /// using `.meta.json`'s schema -- so a newly added column's *data*
    /// safely reached `data.parquet`, but became unreadable after reload
    /// because the metadata never admitted the column existed.
    pub fn update_stats(&mut self, schema: &Schema, rows: &[Tuple]) {
        self.row_count = rows.len();
        self.updated_at = Utc::now();
        self.schema = schema.clone();
        self.column_stats = schema
            .fields
            .iter()
            .map(|field| (field.name.clone(), ColumnStats::compute(field, rows)))
            .collect();
    }
}

use crate::core::index::Index;
use crate::query::logical::Expr;

/// Dataset represents a table-like collection of tuples
#[derive(Debug, Clone, Serialize)]
pub struct Dataset {
    pub id: DatasetId,
    pub schema: Arc<Schema>,
    pub rows: Vec<Tuple>,
    pub metadata: DatasetMetadata,
    #[serde(skip)]
    pub indices: HashMap<String, Box<dyn Index>>,
    #[serde(skip)]
    pub lazy_expressions: HashMap<String, Expr>, // column_name -> expression for lazy evaluation
    /// Per-partition zone maps for range-predicate pruning. Derived,
    /// read-only metadata over `rows` -- see `PartitionStats`. Not
    /// persisted (`#[serde(skip)]`), same rationale as `indices`: cheap to
    /// recompute, and only ever meaningful alongside the exact rows it was
    /// built from.
    #[serde(skip)]
    pub partitions: Vec<PartitionStats>,
}

impl Dataset {
    /// Create a new empty dataset
    pub fn new(id: DatasetId, schema: Arc<Schema>, name: Option<String>) -> Self {
        let mut metadata = DatasetMetadata::new(name, (*schema).clone());
        metadata.update_stats(&schema, &[]);

        Self {
            id,
            schema,
            rows: Vec::new(),
            metadata,
            indices: HashMap::new(),
            lazy_expressions: HashMap::new(),
            partitions: Vec::new(),
        }
    }

    /// Create a dataset with initial rows
    pub fn with_rows(
        id: DatasetId,
        schema: Arc<Schema>,
        rows: Vec<Tuple>,
        name: Option<String>,
    ) -> Result<Self, String> {
        // Validate all rows match schema (structural equality, not pointer equality,
        // so rows from different query results with the same shape are accepted)
        for (i, row) in rows.iter().enumerate() {
            if row.schema.as_ref() != schema.as_ref() {
                return Err(format!("Row {} has incompatible schema", i));
            }
        }

        let mut metadata = DatasetMetadata::new(name, (*schema).clone());
        metadata.update_stats(&schema, &rows);

        let mut dataset = Self {
            id,
            schema,
            rows,
            metadata,
            indices: HashMap::new(),
            lazy_expressions: HashMap::new(),
            partitions: Vec::new(),
        };
        dataset.rebuild_partitions();
        Ok(dataset)
    }

    /// Real content hash over schema + rows (SHA256, via
    /// `crate::core::provenance::compute_content_hash`) -- replaces the
    /// `format!("{name}:{row_count}")` placeholder that used to stand in for
    /// a hash in `core/storage.rs` and `dsl/persistence.rs`. Used as the
    /// stable, content-derived key `ProvenanceStore` resolves ancestry by
    /// (see `LINEAGE_AND_LINALG_PLAN.md`'s Phase 0 outcome, finding 4/5).
    ///
    /// Hashes `self.schema.fields` (`Vec<Field>`, order-stable) and each
    /// row's `values` -- deliberately never `Schema` or `Tuple` themselves
    /// (whole-struct serialization, via their `#[derive(Serialize)]`):
    /// `Schema` carries a derived `field_indices: HashMap<String, usize>`,
    /// and every `Tuple` embeds an `Arc<Schema>` with the very same field,
    /// so serializing either pulls that `HashMap` in. `HashMap`'s
    /// serialization order is randomized per instance (a fresh
    /// `RandomState` seed every `Schema::new` call), so hashing it made two
    /// structurally-identical datasets produced by different code paths --
    /// or even the same path called twice -- hash differently, silently
    /// breaking every ancestry match.
    pub fn content_hash(&self) -> String {
        let row_values: Vec<&Vec<Value>> = self.rows.iter().map(|t| &t.values).collect();
        let bytes = serde_json::to_vec(&(&self.schema.fields, &row_values))
            .unwrap_or_else(|_| format!("{:?}", self.schema.fields).into_bytes());
        crate::core::provenance::compute_content_hash(&bytes)
    }

    /// Recompute `metadata` (row count + column stats) and `partitions`
    /// (per-partition zone maps) from the current `rows`. Called by every
    /// dataset-rebuilding transformation (`filter`, `select`, `sort_by`,
    /// ...); `add_row` instead maintains both incrementally (see below) to
    /// avoid rescanning every row on every single insert.
    fn refresh_stats(&mut self) {
        let schema = self.schema.clone();
        self.metadata.update_stats(&schema, &self.rows);
        self.rebuild_partitions();
    }

    fn rebuild_partitions(&mut self) {
        self.partitions.clear();
        let mut start = 0;
        while start < self.rows.len() {
            let end = (start + BATCH_SIZE).min(self.rows.len());
            let chunk = &self.rows[start..end];
            let column_stats = self
                .schema
                .fields
                .iter()
                .map(|field| (field.name.clone(), ColumnStats::compute(field, chunk)))
                .collect();
            self.partitions.push(PartitionStats {
                start,
                end,
                column_stats,
            });
            start = end;
        }
    }

    /// Retrieve specific rows by their IDs (indices in the rows vector)
    /// Used for optimized query execution via indices
    pub fn get_rows_by_ids(&self, row_ids: &[usize]) -> Vec<Tuple> {
        let mut new_rows = Vec::with_capacity(row_ids.len());
        for &id in row_ids {
            if id < self.rows.len() {
                new_rows.push(self.rows[id].clone());
            }
        }
        new_rows
    }

    /// Add a row to the dataset
    pub fn add_row(&mut self, row: Tuple) -> Result<(), String> {
        if !Arc::ptr_eq(&row.schema, &self.schema) {
            return Err("Row schema does not match dataset schema".to_string());
        }

        let row_id = self.rows.len();

        // Update indices
        for (col_name, index) in &mut self.indices {
            if let Some(value) = row.get(col_name) {
                index.add(row_id, value)?;
            }
        }

        // Incremental stats maintenance: merge just this one row into the
        // dataset-wide summary and into the current (or a fresh) last
        // partition, instead of the O(n) `update_stats`/`rebuild_partitions`
        // full rescans -- those would make every single-row insert cost
        // O(n), turning a bulk load into O(n^2) overall.
        self.metadata.row_count += 1;
        self.metadata.updated_at = Utc::now();
        for field in &self.schema.fields {
            if let Some(value) = row.get(&field.name) {
                self.metadata
                    .column_stats
                    .entry(field.name.clone())
                    .or_insert_with(|| ColumnStats::empty(field.value_type.clone()))
                    .merge_value(value);
            }
        }

        let starts_new_partition = match self.partitions.last() {
            Some(p) => p.end - p.start >= BATCH_SIZE,
            None => true,
        };
        if starts_new_partition {
            let mut column_stats = HashMap::new();
            for field in &self.schema.fields {
                let mut stats = ColumnStats::empty(field.value_type.clone());
                if let Some(value) = row.get(&field.name) {
                    stats.merge_value(value);
                }
                column_stats.insert(field.name.clone(), stats);
            }
            self.partitions.push(PartitionStats {
                start: row_id,
                end: row_id + 1,
                column_stats,
            });
        } else {
            let partition = self.partitions.last_mut().expect("checked above");
            partition.end = row_id + 1;
            for field in &self.schema.fields {
                if let Some(value) = row.get(&field.name) {
                    partition
                        .column_stats
                        .entry(field.name.clone())
                        .or_insert_with(|| ColumnStats::empty(field.value_type.clone()))
                        .merge_value(value);
                }
            }
        }

        self.rows.push(row);
        Ok(())
    }

    /// Get number of rows
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Filter rows based on a predicate
    pub fn filter<F>(&self, predicate: F) -> Self
    where
        F: Fn(&Tuple) -> bool,
    {
        let filtered_rows: Vec<Tuple> =
            self.rows.iter().filter(|r| predicate(r)).cloned().collect();

        let mut new_dataset = Self {
            id: self.id,
            schema: self.schema.clone(),
            rows: filtered_rows,
            metadata: self.metadata.clone(),
            indices: HashMap::new(), // Indices are not preserved on filter for now
            lazy_expressions: self.lazy_expressions.clone(), // Preserve lazy expressions
            partitions: Vec::new(),
        };

        new_dataset.refresh_stats();
        new_dataset
    }

    /// Select specific columns (projection)
    pub fn select(&self, column_names: &[&str]) -> Result<Self, String> {
        // Build new schema with selected fields
        let mut new_fields = Vec::new();
        let mut field_indices = Vec::new();

        for &col_name in column_names {
            let idx = self
                .schema
                .get_field_index(col_name)
                .ok_or_else(|| format!("Column '{}' not found", col_name))?;
            new_fields.push(self.schema.fields[idx].clone());
            field_indices.push(idx);
        }

        let new_schema = Arc::new(Schema::new(new_fields));

        // Project rows
        let mut new_rows = Vec::new();
        for row in &self.rows {
            let new_values: Vec<Value> = field_indices
                .iter()
                .map(|&idx| row.values[idx].clone())
                .collect();

            new_rows.push(Tuple::new(new_schema.clone(), new_values)?);
        }

        // Preserve lazy expressions for selected columns
        let mut new_lazy_expressions = HashMap::new();
        for &col_name in column_names {
            if let Some(expr) = self.lazy_expressions.get(col_name) {
                new_lazy_expressions.insert(col_name.to_string(), expr.clone());
            }
        }

        let mut new_dataset = Self {
            id: self.id,
            schema: new_schema.clone(),
            rows: new_rows,
            metadata: self.metadata.clone(),
            indices: HashMap::new(),
            lazy_expressions: new_lazy_expressions,
            partitions: Vec::new(),
        };

        new_dataset.refresh_stats();
        Ok(new_dataset)
    }

    /// Take first N rows
    pub fn take(&self, n: usize) -> Self {
        let taken_rows: Vec<Tuple> = self.rows.iter().take(n).cloned().collect();

        let mut new_dataset = Self {
            id: self.id,
            schema: self.schema.clone(),
            rows: taken_rows,
            metadata: self.metadata.clone(),
            indices: HashMap::new(),
            lazy_expressions: self.lazy_expressions.clone(),
            partitions: Vec::new(),
        };

        new_dataset.refresh_stats();
        new_dataset
    }

    /// Skip first N rows
    pub fn skip(&self, n: usize) -> Self {
        let skipped_rows: Vec<Tuple> = self.rows.iter().skip(n).cloned().collect();

        let mut new_dataset = Self {
            id: self.id,
            schema: self.schema.clone(),
            rows: skipped_rows,
            metadata: self.metadata.clone(),
            indices: HashMap::new(),
            lazy_expressions: self.lazy_expressions.clone(),
            partitions: Vec::new(),
        };

        new_dataset.refresh_stats();
        new_dataset
    }

    /// Sort by a column
    pub fn sort_by(&self, column_name: &str, ascending: bool) -> Result<Self, String> {
        let col_idx = self
            .schema
            .get_field_index(column_name)
            .ok_or_else(|| format!("Column '{}' not found", column_name))?;

        let mut sorted_rows = self.rows.clone();
        sorted_rows.sort_by(|a, b| {
            let val_a = &a.values[col_idx];
            let val_b = &b.values[col_idx];

            let cmp = val_a.compare(val_b).unwrap_or(std::cmp::Ordering::Equal);

            if ascending {
                cmp
            } else {
                cmp.reverse()
            }
        });

        let mut new_dataset = Self {
            id: self.id,
            schema: self.schema.clone(),
            rows: sorted_rows,
            // Whole-dataset min/max/null-count are unaffected by row order,
            // so metadata can be reused as-is -- but per-partition zone maps
            // group rows by *position*, which sorting changes, so those
            // must still be rebuilt.
            metadata: self.metadata.clone(),
            indices: HashMap::new(),
            lazy_expressions: self.lazy_expressions.clone(),
            partitions: Vec::new(),
        };
        new_dataset.rebuild_partitions();
        Ok(new_dataset)
    }

    /// Map over rows to transform them
    pub fn map<F>(&self, f: F) -> Self
    where
        F: Fn(&Tuple) -> Tuple,
    {
        let mapped_rows: Vec<Tuple> = self.rows.iter().map(f).collect();

        let mut new_dataset = Self {
            id: self.id,
            schema: self.schema.clone(),
            rows: mapped_rows,
            metadata: self.metadata.clone(),
            indices: HashMap::new(),
            lazy_expressions: self.lazy_expressions.clone(),
            partitions: Vec::new(),
        };

        new_dataset.refresh_stats();
        new_dataset
    }

    pub fn get_column(&self, column_name: &str) -> Result<Vec<super::value::Value>, String> {
        let col_idx = self
            .schema
            .get_field_index(column_name)
            .ok_or_else(|| format!("Column '{}' not found", column_name))?;

        // Check if this is a lazy column
        let field = &self.schema.fields[col_idx];
        if field.is_lazy {
            // Evaluate lazy expression for each row
            use crate::query::physical::evaluate_expression;
            let expr = self
                .lazy_expressions
                .get(column_name)
                .ok_or_else(|| format!("Lazy expression not found for column '{}'", column_name))?;

            let mut column_values = Vec::with_capacity(self.rows.len());
            for row in &self.rows {
                let val = evaluate_expression(expr, row);
                column_values.push(val);
            }
            Ok(column_values)
        } else {
            // Regular column - just extract values
            let mut column_values = Vec::with_capacity(self.rows.len());
            for row in &self.rows {
                column_values.push(row.values[col_idx].clone());
            }
            Ok(column_values)
        }
    }

    /// Add an index to a column
    pub fn create_index(
        &mut self,
        column_name: String,
        mut index: Box<dyn Index>,
    ) -> Result<(), String> {
        if !self.schema_has_field(&column_name) {
            return Err(format!("Column '{}' not found in schema", column_name));
        }

        // Populate index with existing data
        for (i, row) in self.rows.iter().enumerate() {
            if let Some(val) = row.get(&column_name) {
                index.add(i, val)?;
            }
        }
        // Let the index compute any batch-derived structure (e.g. k-means
        // clusters for a vector index) now that the full backfill is in.
        index.build()?;

        self.indices.insert(column_name, index);
        Ok(())
    }

    /// Get index for a column
    pub fn get_index(&self, column_name: &str) -> Option<&dyn Index> {
        self.indices.get(column_name).map(|b| b.as_ref())
    }

    /// Snapshot which columns are indexed and with what index type, for
    /// persisting alongside the dataset (see `IndexDefinition`).
    pub fn index_definitions(&self) -> Vec<crate::core::index::IndexDefinition> {
        self.indices
            .iter()
            .map(|(column, index)| crate::core::index::IndexDefinition {
                column: column.clone(),
                index_type: index.index_type(),
            })
            .collect()
    }

    fn schema_has_field(&self, name: &str) -> bool {
        self.schema.fields.iter().any(|f| f.name == *name)
    }

    /// Add a new column to the dataset with a default value
    /// This creates a new schema and updates all existing rows
    pub fn add_column(
        &mut self,
        column_name: String,
        value_type: ValueType,
        default_value: Value,
        nullable: bool,
    ) -> Result<(), String> {
        // Validate that column doesn't already exist
        if self.schema.get_field(column_name.as_str()).is_some() {
            return Err(format!("Column '{}' already exists", column_name));
        }

        // Validate that default value matches the type
        if !default_value.is_null() && !default_value.matches_type(&value_type) {
            return Err(format!(
                "Default value type mismatch: expected {:?}, got {:?}",
                value_type,
                default_value.value_type()
            ));
        }

        // Create new schema with the additional field
        let mut new_fields = self.schema.fields.clone();
        new_fields.push(super::tuple::Field {
            name: column_name.clone(),
            value_type,
            nullable,
            is_lazy: false,
        });
        let new_schema = Arc::new(Schema::new(new_fields));

        // Update all existing rows to include the new column
        let mut new_rows = Vec::with_capacity(self.rows.len());
        for row in &self.rows {
            let mut new_values = row.values.clone();
            new_values.push(default_value.clone());
            new_rows.push(Tuple::new(new_schema.clone(), new_values)?);
        }

        // Update dataset
        self.schema = new_schema;
        self.rows = new_rows;
        self.metadata.update_stats(&self.schema, &self.rows);
        self.rebuild_partitions();

        Ok(())
    }

    /// Add a computed column to the dataset
    /// This evaluates an expression for each row and adds the result as a new column
    /// If lazy is true, stores NULL placeholders and evaluates on access
    pub fn add_computed_column(
        &mut self,
        column_name: String,
        value_type: ValueType,
        computed_values: Vec<Value>,
        expression: crate::query::logical::Expr,
        lazy: bool,
    ) -> Result<(), String> {
        // Validate that column doesn't already exist
        if self.schema.get_field(column_name.as_str()).is_some() {
            return Err(format!("Column '{}' already exists", column_name));
        }

        // Create new schema with the additional field
        let mut new_fields = self.schema.fields.clone();
        let new_field = super::tuple::Field {
            name: column_name.clone(),
            value_type: value_type.clone(),
            nullable: lazy, // Lazy columns can have NULL placeholders
            is_lazy: lazy,
        };
        new_fields.push(new_field.clone());
        let new_schema = Arc::new(Schema::new(new_fields));

        if lazy {
            // For lazy columns, store NULL placeholders and save expression
            let mut new_rows = Vec::with_capacity(self.rows.len());
            for row in &self.rows {
                let mut new_values = row.values.clone();
                new_values.push(Value::Null); // Placeholder for lazy column
                new_rows.push(Tuple::new(new_schema.clone(), new_values)?);
            }
            self.rows = new_rows;
            self.lazy_expressions
                .insert(column_name.clone(), expression);
        } else {
            // Materialized: validate and store computed values
            // Validate that computed values match the number of rows
            if computed_values.len() != self.rows.len() {
                return Err(format!(
                    "Computed values count ({}) doesn't match row count ({})",
                    computed_values.len(),
                    self.rows.len()
                ));
            }

            // Validate that all computed values match the type
            for (i, val) in computed_values.iter().enumerate() {
                if !val.matches_type(&new_field.value_type) {
                    return Err(format!(
                        "Computed value at row {} type mismatch: expected {:?}, got {:?}",
                        i,
                        new_field.value_type,
                        val.value_type()
                    ));
                }
            }

            // Update all existing rows to include the computed column
            let mut new_rows = Vec::with_capacity(self.rows.len());
            for (row, computed_val) in self.rows.iter().zip(computed_values.iter()) {
                let mut new_values = row.values.clone();
                new_values.push(computed_val.clone());
                new_rows.push(Tuple::new(new_schema.clone(), new_values)?);
            }
            self.rows = new_rows;
        }

        // Update dataset
        self.schema = new_schema;
        self.metadata.update_stats(&self.schema, &self.rows);
        self.rebuild_partitions();

        Ok(())
    }

    /// Evaluate a lazy column value for a specific row
    pub fn evaluate_lazy_column(&self, column_name: &str, row: &Tuple) -> Option<Value> {
        if let Some(expr) = self.lazy_expressions.get(column_name) {
            use crate::query::physical::evaluate_expression;
            Some(evaluate_expression(expr, row))
        } else {
            None
        }
    }

    /// Get a row with lazy columns evaluated
    pub fn get_row_evaluated(&self, index: usize) -> Option<Tuple> {
        if index >= self.rows.len() {
            return None;
        }

        let row = &self.rows[index];
        let mut evaluated_values = row.values.clone();

        // Evaluate any lazy columns
        for (i, field) in self.schema.fields.iter().enumerate() {
            if field.is_lazy && i < evaluated_values.len() {
                if let Some(evaluated_val) = self.evaluate_lazy_column(&field.name, row) {
                    evaluated_values[i] = evaluated_val;
                }
            }
        }

        Tuple::new(self.schema.clone(), evaluated_values).ok()
    }

    /// Materialize all lazy columns (convert to regular columns with computed values)
    pub fn materialize_lazy_columns(&mut self) -> Result<(), String> {
        let lazy_columns: Vec<String> = self
            .schema
            .fields
            .iter()
            .filter(|f| f.is_lazy)
            .map(|f| f.name.clone())
            .collect();

        if lazy_columns.is_empty() {
            return Ok(()); // Nothing to materialize
        }

        // Evaluate all lazy columns for all rows
        use crate::query::physical::evaluate_expression;
        let mut new_rows = Vec::with_capacity(self.rows.len());

        for row in &self.rows {
            let mut new_values = row.values.clone();

            // Evaluate lazy columns
            for (i, field) in self.schema.fields.iter().enumerate() {
                if field.is_lazy && i < new_values.len() {
                    if let Some(expr) = self.lazy_expressions.get(&field.name) {
                        let evaluated_val = evaluate_expression(expr, row);
                        new_values[i] = evaluated_val;
                    }
                }
            }

            new_rows.push(Tuple::new(self.schema.clone(), new_values)?);
        }

        // Update schema to mark columns as non-lazy
        let mut new_fields = self.schema.fields.clone();
        for field in &mut new_fields {
            if field.is_lazy {
                field.is_lazy = false;
            }
        }
        let new_schema = Arc::new(Schema::new(new_fields));

        // Update dataset
        self.rows = new_rows;
        self.schema = new_schema;

        // Clear lazy expressions (they're now materialized)
        for col_name in &lazy_columns {
            self.lazy_expressions.remove(col_name);
        }

        self.metadata.update_stats(&self.schema, &self.rows);
        self.rebuild_partitions();
        Ok(())
    }

    // ========== Batch Processing Methods ==========

    /// Get an iterator over batches of rows
    pub fn batches(&self, batch_size: usize) -> impl Iterator<Item = &[Tuple]> + '_ {
        self.rows.chunks(batch_size)
    }

    /// Filter rows in batches for better cache locality and optional parallelization
    pub fn filter_batched<F>(&self, predicate: F) -> Self
    where
        F: Fn(&Tuple) -> bool + Sync + Send,
    {
        let filtered_rows: Vec<Tuple> = if self.rows.len() >= BATCH_PARALLEL_THRESHOLD {
            // Parallel batch processing for large datasets
            self.rows
                .par_chunks(BATCH_SIZE)
                .flat_map(|batch| {
                    batch
                        .iter()
                        .filter(|r| predicate(r))
                        .cloned()
                        .collect::<Vec<_>>()
                })
                .collect()
        } else {
            // Serial batch processing for smaller datasets
            self.rows
                .chunks(BATCH_SIZE)
                .flat_map(|batch| batch.iter().filter(|r| predicate(r)).cloned())
                .collect()
        };

        let mut new_dataset = Self {
            id: self.id,
            schema: self.schema.clone(),
            rows: filtered_rows,
            metadata: self.metadata.clone(),
            indices: HashMap::new(),
            lazy_expressions: self.lazy_expressions.clone(),
            partitions: Vec::new(),
        };

        new_dataset.refresh_stats();
        new_dataset
    }

    /// Map over rows in batches for better cache locality and optional parallelization
    pub fn map_batched<F>(&self, f: F) -> Self
    where
        F: Fn(&Tuple) -> Tuple + Sync + Send,
    {
        let mapped_rows: Vec<Tuple> = if self.rows.len() >= BATCH_PARALLEL_THRESHOLD {
            // Parallel batch processing
            self.rows
                .par_chunks(BATCH_SIZE)
                .flat_map(|batch| batch.iter().map(&f).collect::<Vec<_>>())
                .collect()
        } else {
            // Serial batch processing
            self.rows
                .chunks(BATCH_SIZE)
                .flat_map(|batch| batch.iter().map(&f))
                .collect()
        };

        let mut new_dataset = Self {
            id: self.id,
            schema: self.schema.clone(),
            rows: mapped_rows,
            metadata: self.metadata.clone(),
            indices: HashMap::new(),
            lazy_expressions: self.lazy_expressions.clone(),
            partitions: Vec::new(),
        };

        new_dataset.refresh_stats();
        new_dataset
    }

    /// Select columns in batches for better memory access patterns
    pub fn select_batched(&self, column_names: &[&str]) -> Result<Self, String> {
        // Build new schema with selected fields
        let mut new_fields = Vec::new();
        let mut field_indices = Vec::new();

        for &col_name in column_names {
            let idx = self
                .schema
                .get_field_index(col_name)
                .ok_or_else(|| format!("Column '{}' not found", col_name))?;
            new_fields.push(self.schema.fields[idx].clone());
            field_indices.push(idx);
        }

        let new_schema = Arc::new(Schema::new(new_fields));

        // Project rows in batches
        let new_rows: Vec<Tuple> = if self.rows.len() >= BATCH_PARALLEL_THRESHOLD {
            // Parallel batch processing
            self.rows
                .par_chunks(BATCH_SIZE)
                .flat_map(|batch| {
                    batch
                        .iter()
                        .map(|row| {
                            let new_values: Vec<Value> = field_indices
                                .iter()
                                .map(|&idx| row.values[idx].clone())
                                .collect();
                            Tuple::new(new_schema.clone(), new_values)
                        })
                        .collect::<Result<Vec<_>, _>>()
                        .unwrap_or_default()
                })
                .collect()
        } else {
            // Serial batch processing
            let mut result = Vec::new();
            for batch in self.rows.chunks(BATCH_SIZE) {
                for row in batch {
                    let new_values: Vec<Value> = field_indices
                        .iter()
                        .map(|&idx| row.values[idx].clone())
                        .collect();
                    result.push(Tuple::new(new_schema.clone(), new_values)?);
                }
            }
            result
        };

        // Preserve lazy expressions for selected columns
        let mut new_lazy_expressions = HashMap::new();
        for &col_name in column_names {
            if let Some(expr) = self.lazy_expressions.get(col_name) {
                new_lazy_expressions.insert(col_name.to_string(), expr.clone());
            }
        }

        let mut new_dataset = Self {
            id: self.id,
            schema: new_schema.clone(),
            rows: new_rows,
            metadata: self.metadata.clone(),
            indices: HashMap::new(),
            lazy_expressions: new_lazy_expressions,
            partitions: Vec::new(),
        };

        new_dataset.refresh_stats();
        Ok(new_dataset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tuple::Field;

    fn create_test_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", ValueType::Int),
            Field::new("name", ValueType::String),
            Field::new("age", ValueType::Int),
            Field::new("score", ValueType::Float),
        ]))
    }

    fn create_test_rows(schema: Arc<Schema>) -> Vec<Tuple> {
        vec![
            Tuple::new(
                schema.clone(),
                vec![
                    Value::Int(1),
                    Value::String("Alice".to_string()),
                    Value::Int(30),
                    Value::Float(0.95),
                ],
            )
            .unwrap(),
            Tuple::new(
                schema.clone(),
                vec![
                    Value::Int(2),
                    Value::String("Bob".to_string()),
                    Value::Int(25),
                    Value::Float(0.85),
                ],
            )
            .unwrap(),
            Tuple::new(
                schema.clone(),
                vec![
                    Value::Int(3),
                    Value::String("Carol".to_string()),
                    Value::Int(35),
                    Value::Float(0.90),
                ],
            )
            .unwrap(),
        ]
    }

    #[test]
    fn test_dataset_creation() {
        let schema = create_test_schema();
        let dataset = Dataset::new(DatasetId(1), schema.clone(), Some("test".to_string()));

        assert_eq!(dataset.len(), 0);
        assert_eq!(dataset.metadata.name, Some("test".to_string()));
        assert_eq!(dataset.metadata.row_count, 0);
    }

    #[test]
    fn test_dataset_with_rows() {
        let schema = create_test_schema();
        let rows = create_test_rows(schema.clone());

        let dataset =
            Dataset::with_rows(DatasetId(1), schema, rows, Some("users".to_string())).unwrap();

        assert_eq!(dataset.len(), 3);
        assert_eq!(dataset.metadata.row_count, 3);
    }

    #[test]
    fn test_add_row() {
        let schema = create_test_schema();
        let mut dataset = Dataset::new(DatasetId(1), schema.clone(), None);

        let row = Tuple::new(
            schema.clone(),
            vec![
                Value::Int(1),
                Value::String("Alice".to_string()),
                Value::Int(30),
                Value::Float(0.95),
            ],
        )
        .unwrap();

        assert!(dataset.add_row(row).is_ok());
        assert_eq!(dataset.len(), 1);
    }

    #[test]
    fn test_filter() {
        let schema = create_test_schema();
        let rows = create_test_rows(schema.clone());
        let dataset = Dataset::with_rows(DatasetId(1), schema, rows, None).unwrap();

        // Filter age > 25
        let filtered = dataset.filter(|row| {
            if let Some(Value::Int(age)) = row.get("age") {
                *age > 25
            } else {
                false
            }
        });

        assert_eq!(filtered.len(), 2); // Alice (30) and Carol (35)
    }

    #[test]
    fn test_select() {
        let schema = create_test_schema();
        let rows = create_test_rows(schema.clone());
        let dataset = Dataset::with_rows(DatasetId(1), schema, rows, None).unwrap();

        let selected = dataset.select(&["name", "age"]).unwrap();

        assert_eq!(selected.schema.len(), 2);
        assert_eq!(selected.len(), 3);
        assert!(selected.schema.get_field("name").is_some());
        assert!(selected.schema.get_field("age").is_some());
        assert!(selected.schema.get_field("score").is_none());
    }

    #[test]
    fn test_take_and_skip() {
        let schema = create_test_schema();
        let rows = create_test_rows(schema.clone());
        let dataset = Dataset::with_rows(DatasetId(1), schema, rows, None).unwrap();

        let taken = dataset.take(2);
        assert_eq!(taken.len(), 2);

        let skipped = dataset.skip(1);
        assert_eq!(skipped.len(), 2);
    }

    #[test]
    fn test_sort_by() {
        let schema = create_test_schema();
        let rows = create_test_rows(schema.clone());
        let dataset = Dataset::with_rows(DatasetId(1), schema, rows, None).unwrap();

        // Sort by age ascending
        let sorted_asc = dataset.sort_by("age", true).unwrap();
        if let Some(Value::Int(age)) = sorted_asc.rows[0].get("age") {
            assert_eq!(*age, 25); // Bob is youngest
        }

        // Sort by age descending
        let sorted_desc = dataset.sort_by("age", false).unwrap();
        if let Some(Value::Int(age)) = sorted_desc.rows[0].get("age") {
            assert_eq!(*age, 35); // Carol is oldest
        }
    }

    #[test]
    fn test_metadata_stats() {
        let schema = create_test_schema();
        let rows = create_test_rows(schema.clone());
        let dataset = Dataset::with_rows(DatasetId(1), schema, rows, None).unwrap();

        // Check age stats
        let age_stats = dataset.metadata.column_stats.get("age").unwrap();
        assert_eq!(age_stats.min, Some(Value::Int(25)));
        assert_eq!(age_stats.max, Some(Value::Int(35)));
        assert_eq!(age_stats.null_count, 0);
    }
}
