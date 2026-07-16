use std::sync::Arc;

use crate::core::storage::ParquetStorage;
use crate::core::tensor::Shape;
use crate::core::tuple::{Field, Schema, Tuple};
use crate::core::value::{Value, ValueType};
use crate::dsl::ast::*;
use crate::dsl::{DslError, DslOutput};
use crate::engine::context::ExecutionContext;
use crate::engine::{TensorDb, TensorKind};
use crate::query::logical::LogicalPlan;
use crate::query::planner::Planner;

mod eval;
mod explain;
mod pipeline;
mod query;
mod show;

pub use eval::expr_to_string;
pub use explain::execute_explain;
pub(crate) use pipeline::execute_show_pipelines;

// ─── Entry point ──────────────────────────────────────────────────────────────

pub fn execute_statement(
    db: &mut TensorDb,
    stmt: Statement,
    line_no: usize,
    ctx: Option<&mut ExecutionContext>,
) -> Result<DslOutput, DslError> {
    use crate::dsl::persistence;

    let mut local_ctx;
    let ctx: &mut ExecutionContext = match ctx {
        Some(c) => c,
        None => {
            local_ctx = ExecutionContext::new();
            &mut local_ctx
        }
    };

    match stmt {
        // ── Tensor construction ─────────────────────────────────────────────
        Statement::DefineTensor(s) => {
            let shape = Shape::new(s.shape);
            let values: Vec<f32> = s.values.iter().map(|&v| v as f32).collect();
            let kind = to_engine_kind(s.kind);
            db.insert_named_with_kind(s.name.clone(), shape, values, kind)
                .map_err(|e| DslError::Engine {
                    line: line_no,
                    source: e,
                })?;
            Ok(DslOutput::Message(format!("Defined tensor: {}", s.name)))
        }

        Statement::Vector(s) => {
            let n = s.values.len();
            let shape = Shape::new(vec![n]);
            let values: Vec<f32> = s.values.iter().map(|&v| v as f32).collect();
            db.insert_named_with_kind(s.name.clone(), shape, values, TensorKind::Normal)
                .map_err(|e| DslError::Engine {
                    line: line_no,
                    source: e,
                })?;
            Ok(DslOutput::Message(format!("Defined vector: {}", s.name)))
        }

        Statement::Matrix(s) => {
            let shape = Shape::new(vec![s.rows, s.cols]);
            let values: Vec<f32> = s.values.iter().map(|&v| v as f32).collect();
            db.insert_named_with_kind(s.name.clone(), shape, values, TensorKind::Normal)
                .map_err(|e| DslError::Engine {
                    line: line_no,
                    source: e,
                })?;
            Ok(DslOutput::Message(format!("Defined matrix: {}", s.name)))
        }

        // ── Assignment / computation ────────────────────────────────────────
        Statement::Let(s) => eval::eval_let(db, ctx, &s.name, s.lazy, &s.expr, line_no),

        Statement::Derive(s) => eval::eval_let(db, ctx, &s.name, false, &s.source_expr, line_no),

        // ── Zero-copy semantics ─────────────────────────────────────────────
        Statement::Bind(s) => {
            db.active_instance_mut()
                .bind_resource(&s.alias, &s.target)
                .map_err(|e| DslError::Engine {
                    line: line_no,
                    source: e,
                })?;
            Ok(DslOutput::Message(format!(
                "Bound alias '{}' to resource '{}'",
                s.alias, s.target
            )))
        }

        Statement::Attach(s) => {
            db.active_instance_mut()
                .add_column_to_tensor_dataset(&s.dataset, &s.column, &s.tensor)
                .map_err(|e| DslError::Engine {
                    line: line_no,
                    source: e,
                })?;
            Ok(DslOutput::Message(format!(
                "Attached tensor '{}' as column '{}' in dataset '{}'",
                s.tensor, s.column, s.dataset
            )))
        }

        // ── Database management ─────────────────────────────────────────────
        Statement::CreateDatabase(s) => {
            if s.if_not_exists && db.list_databases().contains(&s.name) {
                return Ok(DslOutput::Message(format!(
                    "Database '{}' already exists (skipped)",
                    s.name
                )));
            }
            db.create_database(s.name.clone())
                .map_err(|e| DslError::Engine {
                    line: line_no,
                    source: e,
                })?;
            Ok(DslOutput::Message(format!("Created database: {}", s.name)))
        }

        Statement::DropDatabase(s) => {
            if s.if_exists && !db.list_databases().contains(&s.name) {
                return Ok(DslOutput::Message(format!(
                    "Database '{}' not found (skipped)",
                    s.name
                )));
            }
            db.drop_database(&s.name).map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;
            Ok(DslOutput::Message(format!("Dropped database: {}", s.name)))
        }

        Statement::UseDatabase(s) => {
            db.use_database(&s.name).map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;
            Ok(DslOutput::Message(format!(
                "Switched to database '{}'",
                s.name
            )))
        }

        // ── Introspection ───────────────────────────────────────────────────
        Statement::Show(s) => show::execute_show(db, s.target, line_no),

        Statement::Explain(s) => explain::execute_explain(db, s.target, line_no),

        Statement::Audit(s) => {
            let issues = db
                .verify_tensor_dataset(&s.target)
                .map_err(|e| DslError::Engine {
                    line: line_no,
                    source: e,
                })?;
            if issues.is_empty() {
                Ok(DslOutput::Message(format!(
                    "Audit PASSED for dataset '{}'. All column references are valid.",
                    s.target
                )))
            } else {
                Ok(DslOutput::Message(format!(
                    "Audit FAILED for dataset '{}'. The following columns point to missing or invalid tensors: {:?}",
                    s.target, issues
                )))
            }
        }

        // ── Dataset operations ──────────────────────────────────────────────
        Statement::CreateDataset(s) => {
            if let Some(clause) = s.from {
                query::execute_create_dataset_from(db, s.name, clause, line_no)
            } else {
                let fields: Vec<Field> = s
                    .columns
                    .iter()
                    .map(|c| {
                        let f = Field::new(&c.name, col_type_to_value_type(&c.col_type));
                        if c.nullable {
                            f.nullable()
                        } else {
                            f
                        }
                    })
                    .collect();
                let schema = Arc::new(Schema::new(fields));
                db.create_dataset(s.name.clone(), schema)
                    .map_err(|e| DslError::Engine {
                        line: line_no,
                        source: e,
                    })?;
                Ok(DslOutput::Message(format!("Created dataset: {}", s.name)))
            }
        }

        Statement::AlterDataset(s) => match s.operation {
            AlterOp::AddColumn(col_def) => {
                let vtype = col_type_to_value_type(&col_def.col_type);
                let default_value = match col_def.default_val {
                    Some(FilterValue::Int(n)) => Value::Int(n),
                    Some(FilterValue::Float(f)) => Value::Float(f as f32),
                    Some(FilterValue::Str(s)) => Value::String(s),
                    Some(FilterValue::Bool(b)) => Value::Bool(b),
                    None if col_def.nullable => Value::Null,
                    None => match &vtype {
                        ValueType::Int => Value::Int(0),
                        ValueType::Float => Value::Float(0.0),
                        ValueType::String => Value::String(String::new()),
                        ValueType::Bool => Value::Bool(false),
                        ValueType::Vector(dim) => Value::Vector(vec![0.0; *dim]),
                        ValueType::Matrix(r, c) => Value::Matrix(vec![vec![0.0; *c]; *r]),
                        ValueType::Null => Value::Null,
                    },
                };
                db.alter_dataset_add_column(
                    &s.dataset,
                    col_def.name.clone(),
                    vtype,
                    default_value,
                    col_def.nullable,
                )
                .map_err(|e| DslError::Engine {
                    line: line_no,
                    source: e,
                })?;
                Ok(DslOutput::Message(format!(
                    "Added column '{}' to dataset '{}'",
                    col_def.name, s.dataset
                )))
            }
            AlterOp::AddComputedColumn { name, expr, lazy } => {
                query::execute_add_computed_column(db, &s.dataset, &name, &expr, lazy, line_no)
            }
        },

        Statement::InsertInto(s) => {
            let dataset = s.dataset.clone();
            let schema = db
                .get_dataset(&dataset)
                .map_err(|e| DslError::Engine {
                    line: line_no,
                    source: e,
                })?
                .schema
                .clone();
            let values: Vec<Value> = match s.row {
                InsertRow::Named(named) => {
                    let col_map: std::collections::HashMap<&str, &InsertValue> =
                        named.iter().map(|(k, v)| (k.as_str(), v)).collect();
                    schema
                        .fields
                        .iter()
                        .map(|f| match col_map.get(f.name.as_str()) {
                            Some(InsertValue::Scalar(n)) => match f.value_type {
                                ValueType::Int => Value::Int(*n as i64),
                                _ => Value::Float(*n as f32),
                            },
                            Some(InsertValue::Text(t)) => Value::String(t.clone()),
                            Some(InsertValue::Vector(v)) => {
                                Value::Vector(v.iter().map(|&x| x as f32).collect())
                            }
                            Some(InsertValue::Matrix(m)) => Value::Matrix(
                                m.iter()
                                    .map(|row| row.iter().map(|&x| x as f32).collect())
                                    .collect(),
                            ),
                            Some(InsertValue::Bool(b)) => Value::Bool(*b),
                            Some(InsertValue::TensorRef(_)) | Some(InsertValue::Null) | None => {
                                Value::Null
                            }
                        })
                        .collect()
                }
                InsertRow::Positional(vals) => vals
                    .into_iter()
                    .zip(schema.fields.iter())
                    .map(|(v, f)| match v {
                        InsertValue::Scalar(n) => match f.value_type {
                            ValueType::Int => Value::Int(n as i64),
                            _ => Value::Float(n as f32),
                        },
                        InsertValue::Text(t) => Value::String(t),
                        InsertValue::Vector(v) => {
                            Value::Vector(v.into_iter().map(|x| x as f32).collect())
                        }
                        InsertValue::Matrix(m) => Value::Matrix(
                            m.into_iter()
                                .map(|row| row.into_iter().map(|x| x as f32).collect())
                                .collect(),
                        ),
                        InsertValue::Bool(b) => Value::Bool(b),
                        InsertValue::TensorRef(_) | InsertValue::Null => Value::Null,
                    })
                    .collect(),
            };
            let tuple = Tuple::new(schema.clone(), values).map_err(|e| DslError::Parse {
                line: line_no,
                msg: e,
            })?;
            db.insert_row(&dataset, tuple)
                .map_err(|e| DslError::Engine {
                    line: line_no,
                    source: e,
                })?;
            Ok(DslOutput::None)
        }

        Statement::Select(s) => query::execute_select(db, s, line_no),

        Statement::Materialize(s) => {
            let dataset_name = if let Some(dot) = s.target.find('.') {
                s.target[..dot].to_string()
            } else {
                s.target.clone()
            };
            db.materialize_lazy_columns(&dataset_name)
                .map_err(|e| DslError::Engine {
                    line: line_no,
                    source: e,
                })?;
            Ok(DslOutput::Message(format!(
                "Materialized lazy columns in dataset '{}'",
                dataset_name
            )))
        }

        Statement::Deliver(s) => {
            if db.get_dataset(&s.dataset).is_err() && db.get_tensor_dataset(&s.dataset).is_none() {
                return Err(DslError::Engine {
                    line: line_no,
                    source: crate::engine::EngineError::DatasetNotFound(s.dataset.clone()),
                });
            }

            let storage_base = if let Some(p) = &s.path {
                p.clone()
            } else {
                let mut path = db.config.storage.data_dir.clone();
                path.push(&db.active_instance().name);
                path.to_string_lossy().into_owned()
            };
            let storage = ParquetStorage::new(&storage_base);

            if storage.manifest_exists(&s.dataset) {
                Ok(DslOutput::Message(format!(
                    "Dataset '{}' is deliverable — manifest found under '{}/datasets/{}/' \
                     (served via /delivery/datasets/{}/manifest.json).",
                    s.dataset, storage_base, s.dataset, s.dataset
                )))
            } else {
                Ok(DslOutput::Message(format!(
                    "Dataset '{}' exists but has not been persisted yet. Run `SAVE DATASET {}` \
                     to generate the delivery package (manifest.json, schema.json, stats.json) \
                     served by the /delivery HTTP routes.",
                    s.dataset, s.dataset
                )))
            }
        }

        // ── Persistence ─────────────────────────────────────────────────────
        Statement::Save(s) => {
            persistence::save_typed(db, s.kind, &s.name, s.path.as_deref(), line_no)
        }

        Statement::Load(s) => {
            persistence::load_typed(db, s.kind, &s.name, s.path.as_deref(), line_no)
        }

        Statement::List(s) => persistence::list_typed(db, &s.target, line_no),

        Statement::Import(s) => {
            persistence::import_typed(db, s.ephemeral, &s.path, s.name.as_deref(), line_no)
        }

        Statement::ImportCsv(s) => {
            persistence::import_csv_typed(db, &s.path, s.name.as_deref(), line_no)
        }

        Statement::Export(s) => persistence::export_typed(db, &s.name, &s.path, line_no),

        // ── Index ───────────────────────────────────────────────────────────
        Statement::CreateIndex(s) => match s.kind {
            IndexKindAst::Default | IndexKindAst::Hash | IndexKindAst::BTree => {
                db.create_index(&s.dataset, &s.column)
                    .map_err(|e| DslError::Engine {
                        line: line_no,
                        source: e,
                    })?;
                Ok(DslOutput::Message(format!(
                    "Created index on {}({})",
                    s.dataset, s.column
                )))
            }
            IndexKindAst::Vector => {
                db.create_vector_index(&s.dataset, &s.column)
                    .map_err(|e| DslError::Engine {
                        line: line_no,
                        source: e,
                    })?;
                Ok(DslOutput::Message(format!(
                    "Created VECTOR index on {}({})",
                    s.dataset, s.column
                )))
            }
        },

        // ── Metadata ────────────────────────────────────────────────────────
        Statement::SetMetadata(s) => {
            let key = s.key.to_lowercase();
            let value = s.value.trim_matches('"').to_string();

            db.set_dataset_metadata(&s.dataset, key.clone(), value.clone())
                .map_err(|e| DslError::Engine {
                    line: line_no,
                    source: e,
                })?;

            let path = format!(
                "{}/{}",
                db.config.storage.data_dir.to_string_lossy(),
                db.active_instance().name
            );
            let storage = ParquetStorage::new(&path);

            if storage.metadata_exists(&s.dataset) {
                let mut metadata =
                    storage
                        .load_dataset_metadata(&s.dataset)
                        .map_err(|e| DslError::Parse {
                            line: line_no,
                            msg: format!("Failed to load metadata: {}", e),
                        })?;

                match key.as_str() {
                    "author" => metadata.author = Some(value.clone()),
                    "description" => metadata.description = Some(value.clone()),
                    "tag" => metadata.add_tag(value.clone()),
                    _ => {}
                }

                metadata.increment_version();
                storage
                    .save_dataset_metadata(&metadata)
                    .map_err(|e| DslError::Parse {
                        line: line_no,
                        msg: format!("Failed to save metadata: {}", e),
                    })?;
            }

            Ok(DslOutput::Message(format!(
                "Updated metadata for dataset '{}': {} = {}",
                s.dataset, key, value
            )))
        }

        // ── Search ──────────────────────────────────────────────────────────
        Statement::Search(s) => {
            let source_ds = db.get_dataset(&s.dataset).map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;
            let schema = source_ds.schema.clone();
            let query_tensor = match s.query {
                SearchQuery::TensorRef(ref name) => db
                    .get(name)
                    .map_err(|e| DslError::Engine {
                        line: line_no,
                        source: e,
                    })?
                    .clone(),
                SearchQuery::Inline(ref values) => {
                    use crate::core::tensor::{TensorId, TensorMetadata};
                    let vals_f32: Vec<f32> = values.iter().map(|&v| v as f32).collect();
                    let n = vals_f32.len();
                    let id = TensorId::new();
                    let meta = TensorMetadata::new(id, None);
                    crate::core::tensor::Tensor::new(id, Shape::new(vec![n]), vals_f32, meta)
                        .map_err(|e| DslError::Parse {
                            line: line_no,
                            msg: e,
                        })?
                }
            };
            let plan = LogicalPlan::VectorSearch {
                input: Box::new(LogicalPlan::Scan {
                    dataset_name: s.dataset.clone(),
                    schema,
                }),
                column: s.column.clone(),
                query: query_tensor,
                k: s.top_k,
            };
            let planner = Planner::new(db);
            let physical_plan =
                planner
                    .create_physical_plan(&plan)
                    .map_err(|e| DslError::Engine {
                        line: line_no,
                        source: e,
                    })?;
            let result_rows = physical_plan.execute(db).map_err(|e| DslError::Engine {
                line: line_no,
                source: e,
            })?;
            let result_schema = physical_plan.schema();
            let row_count = result_rows.len();
            let target = s.target.unwrap_or_else(|| "search_results".to_string());
            if let Ok(ds) = db.get_dataset_mut(&target) {
                ds.rows = result_rows;
                ds.metadata.update_stats(&ds.schema, &ds.rows);
            } else {
                db.create_dataset(target.clone(), result_schema.clone())
                    .map_err(|e| DslError::Engine {
                        line: line_no,
                        source: e,
                    })?;
                let ds = db.get_dataset_mut(&target).map_err(|e| DslError::Engine {
                    line: line_no,
                    source: e,
                })?;
                ds.rows = result_rows;
                ds.metadata.update_stats(&ds.schema, &ds.rows);
            }
            Ok(DslOutput::Message(format!(
                "Search completed. Found {} results in '{}'.",
                row_count, target
            )))
        }

        // ── Transform ────────────────────────────────────────────────────────
        Statement::Transform(s) => query::execute_transform(db, s, line_no),

        // ── Named pipelines ───────────────────────────────────────────────────
        Statement::DefinePipeline(s) => pipeline::execute_define_pipeline(db, s, line_no),
        Statement::ApplyPipeline(s) => pipeline::execute_apply_pipeline(db, s, line_no),
        Statement::DropPipeline(name) => pipeline::execute_drop_pipeline(db, name, line_no),
        Statement::DescribePipeline(name) => pipeline::execute_describe_pipeline(db, name, line_no),

        // ── Data mutation ────────────────────────────────────────────────────
        Statement::Update(s) => query::execute_update(db, s, line_no),
        Statement::Delete(s) => query::execute_delete(db, s, line_no),

        // ── Session ─────────────────────────────────────────────────────────
        Statement::Reset => {
            db.reset_session();
            Ok(DslOutput::Message(
                "Session reset complete. All in-memory data has been cleared from the active database.".to_string(),
            ))
        }
    }
}

// ─── Conversion helpers ───────────────────────────────────────────────────────

fn to_engine_kind(k: TensorKindAst) -> TensorKind {
    match k {
        TensorKindAst::Normal => TensorKind::Normal,
        TensorKindAst::Strict => TensorKind::Strict,
        TensorKindAst::Lazy => TensorKind::Lazy,
    }
}

pub(super) fn col_type_to_value_type(ct: &ColType) -> ValueType {
    match ct {
        ColType::Int => ValueType::Int,
        ColType::Float => ValueType::Float,
        ColType::String => ValueType::String,
        ColType::Bool => ValueType::Bool,
        ColType::Vector(n) => ValueType::Vector(*n),
        ColType::Matrix(r, c) => ValueType::Matrix(*r, *c),
        ColType::Tensor(dims) => match dims.as_slice() {
            [d] => ValueType::Vector(*d),
            [r, c] => ValueType::Matrix(*r, *c),
            _ => ValueType::Vector(0),
        },
    }
}
