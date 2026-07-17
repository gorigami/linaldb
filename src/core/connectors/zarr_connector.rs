use crate::core::connectors::{field_with_shape, Connector, ConnectorError};
use crate::core::dataset::{ColumnSchema, DatasetLineage, DatasetSchema};
use crate::core::tensor::Shape;
use crate::core::value::ValueType;
use arrow::array::{ArrayRef, Float32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use std::path::Path;
use std::sync::Arc;
use zarrs::array::Array;
use zarrs::filesystem::FilesystemStore;
use zarrs::group::Group;

pub struct ZarrConnector;

impl Connector for ZarrConnector {
    fn name(&self) -> &str {
        "zarr"
    }

    fn can_handle(&self, path: &str) -> bool {
        let path_obj = Path::new(path);
        path_obj.extension().and_then(|s| s.to_str()) == Some("zarr")
            || path_obj.join("zarr.json").exists()
            || path_obj.join(".zgroup").exists()
    }

    fn read_dataset(&self, path: &str) -> Result<(RecordBatch, DatasetLineage), ConnectorError> {
        let store = Arc::new(
            FilesystemStore::new(path)
                .map_err(|e| ConnectorError::Io(std::io::Error::other(e.to_string())))?,
        );

        let mut columns = Vec::new();
        let mut fields = Vec::new();
        let mut num_rows = 0;

        self.visit_group(
            store.clone(),
            "/",
            "",
            &mut fields,
            &mut columns,
            &mut num_rows,
        )?;

        if fields.is_empty() {
            return Err(ConnectorError::Parse(
                "No arrays found in Zarr store".to_string(),
            ));
        }

        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(schema, columns)?;

        let mut lineage = DatasetLineage::new();
        lineage.add_node(crate::core::dataset::lineage::LineageNode {
            id: uuid::Uuid::new_v4(),
            dataset_name: "zarr_import".to_string(),
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

impl ZarrConnector {
    fn visit_group(
        &self,
        store: Arc<FilesystemStore>,
        group_path: &str,
        prefix: &str,
        fields: &mut Vec<Field>,
        columns: &mut Vec<ArrayRef>,
        num_rows: &mut usize,
    ) -> Result<(), ConnectorError> {
        // In zarrs, arrays and groups are distinct. We can try to open as group.
        if let Ok(group) = Group::open(store.clone(), group_path) {
            let children = group
                .children(false)
                .map_err(|e| ConnectorError::Other(e.to_string()))?;

            for member in children.iter() {
                let member_name = member.name();
                let member_path = member.path().to_string();
                let member_prefix: String = if prefix.is_empty() {
                    member_name.to_string()
                } else {
                    format!("{}/{}", prefix, member_name)
                };

                // Try as array first
                if let Ok(array) = Array::open(store.clone(), &member_path) {
                    self.process_array(&array, &member_prefix, fields, columns, num_rows)?;
                } else {
                    // Try as group (recursive)
                    self.visit_group(
                        store.clone(),
                        &format!("{}/", member_path),
                        &member_prefix,
                        fields,
                        columns,
                        num_rows,
                    )?;
                }
            }
        } else if let Ok(array) = Array::open(store.clone(), group_path) {
            // Root might be an array
            self.process_array(&array, "data", fields, columns, num_rows)?;
        }

        Ok(())
    }

    fn process_array(
        &self,
        array: &Array<FilesystemStore>,
        name: &str,
        fields: &mut Vec<Field>,
        columns: &mut Vec<ArrayRef>,
        num_row_count: &mut usize,
    ) -> Result<(), ConnectorError> {
        let subset = array.subset_all();
        // zarrs 0.19 uses retrieve_array_subset_elements for sync reading.
        let data: Vec<f32> = array
            .retrieve_array_subset_elements::<f32>(&subset)
            .map_err(|e| ConnectorError::Other(e.to_string()))?;

        if *num_row_count == 0 {
            *num_row_count = data.len();
        } else if data.len() != *num_row_count {
            // Skip arrays with mismatching lengths or handle them?
            // For now, only include if length matches the first one.
            return Ok(());
        }

        let dims: Vec<usize> = array.shape().iter().map(|&d| d as usize).collect();
        fields.push(field_with_shape(name, DataType::Float32, false, &dims));
        columns.push(Arc::new(Float32Array::from(data)));

        Ok(())
    }
}
