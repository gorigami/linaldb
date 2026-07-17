use crate::core::connectors::{field_with_shape, Connector, ConnectorError};
use crate::core::dataset::{ColumnSchema, DatasetLineage, DatasetSchema};
use crate::core::tensor::Shape;
use crate::core::value::ValueType;
use arrow::array::{ArrayRef, Float32Array};
use arrow::datatypes::{DataType, Schema};
use arrow::record_batch::RecordBatch;
use ndarray::ArrayD;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

pub struct NumpyConnector;

impl Connector for NumpyConnector {
    fn name(&self) -> &str {
        "numpy"
    }

    fn can_handle(&self, path: &str) -> bool {
        let path = Path::new(path);
        matches!(
            path.extension().and_then(|s| s.to_str()),
            Some("npy") | Some("npz")
        )
    }

    fn read_dataset(&self, path: &str) -> Result<(RecordBatch, DatasetLineage), ConnectorError> {
        let path_obj = Path::new(path);
        let ext = path_obj
            .extension()
            .and_then(|s| s.to_str())
            .ok_or_else(|| ConnectorError::Parse("Missing file extension".to_string()))?;

        if ext == "npz" {
            self.read_npz(path)
        } else {
            self.read_npy(path)
        }
    }

    fn inspect(&self, path: &str) -> Result<DatasetSchema, ConnectorError> {
        let (batch, _) = self.read_dataset(path)?;

        let fields = batch
            .schema()
            .fields()
            .iter()
            .map(|f| {
                let shape = Shape::new(vec![batch.num_rows()]);
                ColumnSchema::new(f.name().clone(), ValueType::Float, shape)
            })
            .collect();

        Ok(DatasetSchema::new(fields))
    }
}

impl NumpyConnector {
    fn read_npy(&self, path: &str) -> Result<(RecordBatch, DatasetLineage), ConnectorError> {
        let arr: ArrayD<f32> = ndarray_npy::read_npy(path)
            .map_err(|e| ConnectorError::Parse(format!("Failed to read NPY: {}", e)))?;

        let (batch, lineage) = self.array_to_batch("array", arr)?;
        Ok((batch, lineage))
    }

    fn read_npz(&self, path: &str) -> Result<(RecordBatch, DatasetLineage), ConnectorError> {
        let file = File::open(path)?;
        let mut npz = ndarray_npy::NpzReader::new(file)
            .map_err(|e| ConnectorError::Parse(format!("Failed to open NPZ: {}", e)))?;

        let mut fields = Vec::new();
        let mut columns: Vec<ArrayRef> = Vec::new();
        let mut num_rows = 0;

        let names: Vec<String> = npz
            .names()
            .map_err(|e| ConnectorError::Parse(format!("Failed to list NPZ names: {}", e)))?
            .into_iter()
            .map(|s| s.to_string())
            .collect();

        for name in names {
            // Using a temporary result to help type inference
            let result: Result<ArrayD<f32>, _> = npz.by_name(&name);
            if let Ok(arr) = result {
                let len = arr.len();
                if num_rows == 0 {
                    num_rows = len;
                } else if len != num_rows {
                    continue;
                }

                let dims = arr.shape().to_vec();
                fields.push(field_with_shape(&name, DataType::Float32, false, &dims));
                let data: Vec<f32> = arr.iter().cloned().collect();
                columns.push(Arc::new(Float32Array::from(data)));
            }
        }

        if fields.is_empty() {
            return Err(ConnectorError::Parse(
                "No valid f32 arrays found in NPZ".to_string(),
            ));
        }

        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(schema, columns)?;

        let mut lineage = DatasetLineage::new();
        lineage.add_node(crate::core::dataset::lineage::LineageNode {
            id: uuid::Uuid::new_v4(),
            dataset_name: "numpy_import".to_string(),
            dataset_hash: "".to_string(),
            operation: "import".to_string(),
            parents: vec![],
            engine_version: env!("CARGO_PKG_VERSION").to_string(),
        });

        Ok((batch, lineage))
    }

    fn array_to_batch(
        &self,
        name: &str,
        arr: ArrayD<f32>,
    ) -> Result<(RecordBatch, DatasetLineage), ConnectorError> {
        let dims = arr.shape().to_vec();
        let data: Vec<f32> = arr.iter().cloned().collect();

        let schema = Arc::new(Schema::new(vec![field_with_shape(
            name,
            DataType::Float32,
            false,
            &dims,
        )]));

        let array = Arc::new(Float32Array::from(data));
        let batch = RecordBatch::try_new(schema, vec![array])?;

        let mut lineage = DatasetLineage::new();
        lineage.add_node(crate::core::dataset::lineage::LineageNode {
            id: uuid::Uuid::new_v4(),
            dataset_name: "numpy_import".to_string(),
            dataset_hash: "".to_string(),
            operation: "import".to_string(),
            parents: vec![],
            engine_version: env!("CARGO_PKG_VERSION").to_string(),
        });

        Ok((batch, lineage))
    }
}
