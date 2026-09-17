//! External Parquet ingestion connector -- SCIENTIFIC_ENGINE_EXPANSION_PLAN.md Phase 1.
//!
//! Distinct from `core::storage::ParquetStorage` (this engine's own dataset-package format,
//! `data/{db}/datasets/{name}/data.parquet`, read via `LOAD DATASET`): this connector is the
//! `USE DATASET FROM "path"`/`IMPORT DATASET FROM "path"` ingestion path for a *generic*,
//! externally-produced `.parquet` file (e.g. a Pandas/Arrow/Spark export), reached through
//! `get_connector_registry()` exactly like `CsvConnector`. The two never collide: `LOAD DATASET`
//! never consults the connector registry at all.
//!
//! Mirrors `CsvConnector`'s structure closely (multi-row-group batches combined via
//! `concat_batches`, the same `FIELDS (...)` column-projection support) -- Parquet's schema is
//! already Arrow-native, so there's no type-inference step to add beyond what `parquet-arrow`
//! gives for free.

use crate::core::connectors::{Connector, ConnectorError};
use crate::core::dataset::{DatasetLineage, DatasetSchema, LineageNode};
use arrow::compute::concat_batches;
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use uuid::Uuid;

pub struct ParquetConnector;

impl ParquetConnector {
    pub fn new() -> Self {
        Self
    }
}

impl Connector for ParquetConnector {
    fn name(&self) -> &str {
        "parquet"
    }

    fn can_handle(&self, path: &str) -> bool {
        path.to_lowercase().ends_with(".parquet")
    }

    fn read_dataset(
        &self,
        path: &str,
        fields: Option<&[String]>,
    ) -> Result<(RecordBatch, DatasetLineage), ConnectorError> {
        let file = fs::File::open(path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| ConnectorError::Parse(format!("invalid Parquet file: {e}")))?;
        let arrow_schema = builder.schema().clone();
        let reader = builder
            .build()
            .map_err(|e| ConnectorError::Parse(format!("failed to build Parquet reader: {e}")))?;

        let mut batches = Vec::new();
        for batch in reader {
            batches.push(batch.map_err(ConnectorError::Arrow)?);
        }

        if batches.is_empty() {
            return Err(ConnectorError::Parse(
                "Parquet file has no row groups / is empty".to_string(),
            ));
        }

        let mut combined_batch = concat_batches(&arrow_schema, batches.iter())?;

        if let Some(names) = fields {
            let indices = names
                .iter()
                .map(|name| {
                    combined_batch.schema().index_of(name).map_err(|_| {
                        ConnectorError::Parse(format!(
                            "FIELDS: column '{name}' not found in Parquet file '{path}'"
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            combined_batch = combined_batch.project(&indices)?;
        }

        let mut lineage = DatasetLineage::new();
        lineage.add_node(LineageNode {
            id: Uuid::new_v4(),
            dataset_name: Path::new(path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string(),
            dataset_hash: crate::core::provenance::record_batch_content_hash(&combined_batch),
            operation: format!("Imported from Parquet: {}", path),
            parents: vec![],
            engine_version: env!("CARGO_PKG_VERSION").to_string(),
        });

        Ok((combined_batch, lineage))
    }

    fn inspect(&self, path: &str) -> Result<DatasetSchema, ConnectorError> {
        let file = fs::File::open(path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| ConnectorError::Parse(format!("invalid Parquet file: {e}")))?;
        let schema: DatasetSchema = Arc::new(builder.schema().as_ref().clone()).into();
        Ok(schema)
    }
}

impl Default for ParquetConnector {
    fn default() -> Self {
        Self::new()
    }
}
