use crate::core::connectors::{field_with_shape, Connector, ConnectorError};
use crate::core::dataset::{ColumnSchema, DatasetLineage, DatasetSchema};
use crate::core::tensor::Shape;
use crate::core::value::ValueType;
use arrow::array::{ArrayRef, Float32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use hdf5::{Dataset, File, Group};
use std::path::Path;
use std::sync::Arc;

pub struct Hdf5Connector;

impl Connector for Hdf5Connector {
    fn name(&self) -> &str {
        "hdf5"
    }

    fn can_handle(&self, path: &str) -> bool {
        let path = Path::new(path);
        matches!(
            path.extension().and_then(|s| s.to_str()),
            Some("h5") | Some("hdf5") | Some("h5ad") | Some("nc")
        )
    }

    fn read_dataset(&self, path: &str) -> Result<(RecordBatch, DatasetLineage), ConnectorError> {
        let file = File::open(path)
            .map_err(|e| ConnectorError::Io(std::io::Error::other(e.to_string())))?;

        let mut columns = Vec::new();
        let mut fields = Vec::new();
        let mut num_rows = 0;

        self.visit_group(&file, "", &mut fields, &mut columns, &mut num_rows)?;

        if fields.is_empty() {
            return Err(ConnectorError::Parse(
                "No datasets found in HDF5 file".to_string(),
            ));
        }

        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(schema, columns)?;

        let mut lineage = DatasetLineage::new();
        lineage.add_node(crate::core::dataset::lineage::LineageNode {
            id: uuid::Uuid::new_v4(),
            dataset_name: "hdf5_import".to_string(),
            dataset_hash: "".to_string(),
            operation: "import".to_string(),
            parents: vec![],
            engine_version: env!("CARGO_PKG_VERSION").to_string(),
        });

        Ok((batch, lineage))
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

impl Hdf5Connector {
    fn visit_group(
        &self,
        group: &Group,
        prefix: &str,
        fields: &mut Vec<Field>,
        columns: &mut Vec<ArrayRef>,
        num_rows: &mut usize,
    ) -> Result<(), ConnectorError> {
        // Visit datasets in this group
        for member_name in group
            .member_names()
            .map_err(|e| ConnectorError::Other(e.to_string()))?
        {
            let name = if prefix.is_empty() {
                member_name.clone()
            } else {
                format!("{}_{}", prefix, member_name)
            };

            // Check if it's a dataset or a group
            if let Ok(ds) = group.dataset(&member_name) {
                self.process_dataset(&ds, &name, fields, columns, num_rows)?;
            } else if let Ok(subgroup) = group.group(&member_name) {
                self.visit_group(&subgroup, &name, fields, columns, num_rows)?;
            }
        }
        Ok(())
    }

    fn process_dataset(
        &self,
        ds: &Dataset,
        name: &str,
        fields: &mut Vec<Field>,
        columns: &mut Vec<ArrayRef>,
        num_row_count: &mut usize,
    ) -> Result<(), ConnectorError> {
        // We only support numeric datasets for now. The underlying Arrow
        // column is always a flat 1D array (LINAL's connector output
        // convention), but we stash the dataset's original shape as field
        // metadata so record_batch_to_tensors can rebuild the real
        // Matrix/Tensor shape instead of assuming a flat Vector.
        let shape = ds.shape();

        let data: Vec<f32> = match ds.read_raw::<f32>() {
            Ok(v) => v,
            Err(_) => {
                // Try reading as f64 and casting
                match ds.read_raw::<f64>() {
                    Ok(v) => v.into_iter().map(|x| x as f32).collect(),
                    Err(_) => {
                        // Skip non-numeric or incompatible datasets
                        return Ok(());
                    }
                }
            }
        };

        if *num_row_count == 0 {
            *num_row_count = data.len();
        } else if data.len() != *num_row_count {
            // Inconsistent length, skip or error?
            // In LINAL, consistency is key, but HDF5 datasets often have different shapes.
            // For now, we skip if inconsistent to allow specialized h5ad handling later.
            return Ok(());
        }

        fields.push(field_with_shape(name, DataType::Float32, false, &shape));
        columns.push(Arc::new(Float32Array::from(data)));

        Ok(())
    }
}
