use crate::core::connectors::{Connector, ConnectorError};
use crate::core::dataset::{DatasetLineage, DatasetSchema, LineageNode};
use arrow::compute::concat_batches;
use arrow::csv;
use arrow::record_batch::RecordBatch;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use uuid::Uuid;

pub struct CsvConnector;

impl CsvConnector {
    pub fn new() -> Self {
        Self
    }
}

impl Connector for CsvConnector {
    fn name(&self) -> &str {
        "csv"
    }

    fn can_handle(&self, path: &str) -> bool {
        path.to_lowercase().ends_with(".csv")
    }

    fn read_dataset(
        &self,
        path: &str,
        fields: Option<&[String]>,
    ) -> Result<(RecordBatch, DatasetLineage), ConnectorError> {
        let file = fs::File::open(path)?;
        let format = csv::reader::Format::default().with_header(true);
        let (arrow_schema, _) = format.infer_schema(file, Some(100))?;

        let arrow_schema_arc = Arc::new(arrow_schema);
        let builder = csv::ReaderBuilder::new(arrow_schema_arc.clone()).with_header(true);

        let file = fs::File::open(path)?;
        let csv_reader = builder.build(file)?;

        let mut batches = Vec::new();
        for batch in csv_reader {
            batches.push(batch?);
        }

        if batches.is_empty() {
            return Err(ConnectorError::Parse("CSV file is empty".to_string()));
        }

        // Combine batches into one for simplicity in this MVP
        // In a real scenario, we might want to keep them chunked
        let mut combined_batch = concat_batches(&arrow_schema_arc, batches.iter())?;

        if let Some(names) = fields {
            let indices = names
                .iter()
                .map(|name| {
                    combined_batch.schema().index_of(name).map_err(|_| {
                        ConnectorError::Parse(format!(
                            "FIELDS: column '{name}' not found in CSV file '{path}'"
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
            dataset_hash: "".to_string(), // TODO: Compute hash if needed
            operation: format!("Imported from CSV: {}", path),
            parents: vec![],
            engine_version: env!("CARGO_PKG_VERSION").to_string(),
        });

        Ok((combined_batch, lineage))
    }

    fn inspect(&self, path: &str) -> Result<DatasetSchema, ConnectorError> {
        let file = fs::File::open(path)?;
        let format = csv::reader::Format::default().with_header(true);
        let (arrow_schema, _) = format.infer_schema(file, Some(100))?;

        let schema: DatasetSchema = Arc::new(arrow_schema).into();
        Ok(schema)
    }
}

impl Default for CsvConnector {
    fn default() -> Self {
        Self::new()
    }
}
