use crate::core::value::Value;
use crate::dsl::DslError;
use crate::engine::TensorDb;
use crate::query::logical::LogicalPlan;

use super::dataset::parse_single_value;

pub fn build_search_query_plan(
    db: &TensorDb,
    line: &str,
    line_no: usize,
) -> Result<(Option<String>, LogicalPlan), DslError> {
    let rest = line.trim_start_matches("SEARCH").trim();

    if rest.contains(" FROM ") {
        // SEARCH target FROM source QUERY vector ON column K=k
        let parts: Vec<&str> = rest.splitn(2, " FROM ").collect();
        let target_name = parts[0].trim().to_string();
        let query_part = parts[1].trim();

        let parts2: Vec<&str> = query_part.splitn(2, " QUERY ").collect();
        if parts2.len() != 2 {
            return Err(DslError::Parse {
                line: line_no,
                msg: "Expected: ... FROM <source> QUERY <vector> ...".into(),
            });
        }
        let source_name = parts2[0].trim();
        let after_query = parts2[1].trim();

        let parts3: Vec<&str> = after_query.splitn(2, " ON ").collect();
        if parts3.len() != 2 {
            return Err(DslError::Parse {
                line: line_no,
                msg: "Expected: ... QUERY <vector> ON <column> ...".into(),
            });
        }
        let vector_str = parts3[0].trim();
        let after_on = parts3[1].trim();

        let parts4: Vec<&str> = if after_on.contains(" K=") {
            after_on.splitn(2, " K=").collect()
        } else if after_on.contains(" K =") {
            after_on.splitn(2, " K =").collect()
        } else {
            return Err(DslError::Parse {
                line: line_no,
                msg: "Expected: ... ON <column> K=<k>".into(),
            });
        };

        let column_name = parts4[0].trim();
        let k: usize = parts4[1].trim().parse().map_err(|_| DslError::Parse {
            line: line_no,
            msg: "Invalid K".into(),
        })?;

        let query_tensor = parse_query_vector(vector_str, line_no)?;
        let source_ds = db.get_dataset(source_name).map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;
        let plan = build_search_plan_internal(
            source_name,
            source_ds.schema.clone(),
            column_name,
            query_tensor,
            k,
        );
        Ok((Some(target_name), plan))
    } else {
        // SEARCH source WHERE col ~= vector LIMIT k
        let parts: Vec<&str> = rest.splitn(2, " WHERE ").collect();
        if parts.len() != 2 {
            return Err(DslError::Parse {
                line: line_no,
                msg: "Expected: SEARCH <source> WHERE <col> ~= <vector> LIMIT <k>".into(),
            });
        }
        let source_name = parts[0].trim();
        let condition_part = parts[1].trim();

        let limit_split: Vec<&str> = condition_part.splitn(2, " LIMIT ").collect();
        if limit_split.len() != 2 {
            return Err(DslError::Parse {
                line: line_no,
                msg: "Expected LIMIT <k>".into(),
            });
        }
        let where_clause = limit_split[0].trim();
        let k: usize = limit_split[1].trim().parse().map_err(|_| DslError::Parse {
            line: line_no,
            msg: "Invalid K".into(),
        })?;

        let cond_parts: Vec<&str> = where_clause.splitn(2, "~=").collect();
        if cond_parts.len() != 2 {
            return Err(DslError::Parse {
                line: line_no,
                msg: "Expected <col> ~= <vector>".into(),
            });
        }
        let column_name = cond_parts[0].trim();
        let vector_str = cond_parts[1].trim();

        let query_tensor = parse_query_vector(vector_str, line_no)?;
        let source_ds = db.get_dataset(source_name).map_err(|e| DslError::Engine {
            line: line_no,
            source: e,
        })?;
        let plan = build_search_plan_internal(
            source_name,
            source_ds.schema.clone(),
            column_name,
            query_tensor,
            k,
        );
        Ok((None, plan))
    }
}

fn parse_query_vector(
    vector_str: &str,
    line_no: usize,
) -> Result<crate::core::tensor::Tensor, DslError> {
    let query_val = parse_single_value(vector_str, line_no)?;
    match query_val {
        Value::Vector(data) => {
            use crate::core::tensor::{Shape, Tensor, TensorId, TensorMetadata};
            let id = TensorId::new();
            let metadata = TensorMetadata::new(id, None);
            Tensor::new(id, Shape::new(vec![data.len()]), data, metadata)
                .map_err(|e| DslError::Parse { line: line_no, msg: e })
        }
        _ => Err(DslError::Parse {
            line: line_no,
            msg: "Query must be a vector".into(),
        }),
    }
}

pub fn build_search_plan_internal(
    source_name: &str,
    source_schema: std::sync::Arc<crate::core::tuple::Schema>,
    column_name: &str,
    query_tensor: crate::core::tensor::Tensor,
    k: usize,
) -> LogicalPlan {
    let scan = LogicalPlan::Scan {
        dataset_name: source_name.to_string(),
        schema: source_schema,
    };
    LogicalPlan::VectorSearch {
        input: Box::new(scan),
        column: column_name.to_string(),
        query: query_tensor,
        k,
    }
}
