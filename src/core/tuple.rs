// src/tuple.rs

use super::value::{Value, ValueType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Field definition in a schema
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Field {
    pub name: String,
    pub value_type: ValueType,
    pub nullable: bool,
    #[serde(default)]
    pub is_lazy: bool, // True if this column is computed lazily (evaluated on access)
}

impl Field {
    pub fn new(name: impl Into<String>, value_type: ValueType) -> Self {
        Self {
            name: name.into(),
            value_type,
            nullable: false,
            is_lazy: false,
        }
    }

    pub fn nullable(mut self) -> Self {
        self.nullable = true;
        self
    }

    pub fn lazy(mut self) -> Self {
        self.is_lazy = true;
        self
    }

    /// Check if a value is compatible with this field
    pub fn is_compatible(&self, value: &Value) -> bool {
        if value.is_null() {
            return self.nullable;
        }

        match (&self.value_type, value.value_type()) {
            (ValueType::Float, ValueType::Float) => true,
            (ValueType::Int, ValueType::Int) => true,
            (ValueType::String, ValueType::String) => true,
            (ValueType::Bool, ValueType::Bool) => true,
            (ValueType::Vector(expected_dim), ValueType::Vector(actual_dim)) => {
                expected_dim == &0 || expected_dim == &actual_dim
            }
            (ValueType::Matrix(er, ec), ValueType::Matrix(ar, ac)) => er == &ar && ec == &ac,
            (ValueType::Null, ValueType::Null) => self.nullable,
            _ => false,
        }
    }
}

/// Schema defines the structure of a tuple
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Schema {
    pub fields: Vec<Field>,
    field_indices: HashMap<String, usize>,
}

impl Schema {
    pub fn new(fields: Vec<Field>) -> Self {
        let field_indices = fields
            .iter()
            .enumerate()
            .map(|(i, f)| (f.name.clone(), i))
            .collect();

        Self {
            fields,
            field_indices,
        }
    }

    /// Get field by name
    pub fn get_field(&self, name: &str) -> Option<&Field> {
        self.field_indices
            .get(name)
            .and_then(|&idx| self.fields.get(idx))
    }

    /// Get field index by name
    pub fn get_field_index(&self, name: &str) -> Option<usize> {
        self.field_indices.get(name).copied()
    }

    /// Number of fields
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Validate that values match this schema
    pub fn validate(&self, values: &[Value]) -> Result<(), String> {
        if values.len() != self.fields.len() {
            return Err(format!(
                "Value count mismatch: expected {}, got {}",
                self.fields.len(),
                values.len()
            ));
        }

        for (i, (field, value)) in self.fields.iter().zip(values.iter()).enumerate() {
            if !field.is_compatible(value) {
                return Err(format!(
                    "Type mismatch at field '{}' (index {}): expected {}, got {}",
                    field.name,
                    i,
                    field.value_type,
                    value.value_type()
                ));
            }
        }

        Ok(())
    }
}

/// Tuple represents a structured record with named fields
#[derive(Debug, Clone, Serialize)]
pub struct Tuple {
    pub schema: Arc<Schema>,
    pub values: Vec<Value>,
}

impl Tuple {
    /// Create a new tuple with validation
    pub fn new(schema: Arc<Schema>, values: Vec<Value>) -> Result<Self, String> {
        schema.validate(&values)?;
        Ok(Self { schema, values })
    }

    /// Get value by field name
    /// Note: For lazy columns, this returns the placeholder (NULL).
    /// Use Dataset::get_row_evaluated() or Dataset::evaluate_lazy_column() for lazy evaluation.
    pub fn get(&self, field_name: &str) -> Option<&Value> {
        self.schema
            .get_field_index(field_name)
            .and_then(|idx| self.values.get(idx))
    }

    /// Get value by index
    pub fn get_by_index(&self, index: usize) -> Option<&Value> {
        self.values.get(index)
    }

    /// Set value by field name
    pub fn set(&mut self, field_name: &str, value: Value) -> Result<(), String> {
        let idx = self
            .schema
            .get_field_index(field_name)
            .ok_or_else(|| format!("Field '{}' not found", field_name))?;

        let field = &self.schema.fields[idx];
        if !field.is_compatible(&value) {
            return Err(format!(
                "Type mismatch for field '{}': expected {}, got {}",
                field_name,
                field.value_type,
                value.value_type()
            ));
        }

        self.values[idx] = value;
        Ok(())
    }

    /// Number of fields
    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_schema() -> Schema {
        Schema::new(vec![
            Field::new("id", ValueType::Int),
            Field::new("name", ValueType::String),
            Field::new("score", ValueType::Float),
            Field::new("active", ValueType::Bool),
        ])
    }

    #[test]
    fn test_schema_creation() {
        let schema = create_test_schema();
        assert_eq!(schema.len(), 4);
        assert_eq!(schema.get_field("id").unwrap().value_type, ValueType::Int);
        assert_eq!(
            schema.get_field("name").unwrap().value_type,
            ValueType::String
        );
        assert_eq!(schema.get_field_index("score"), Some(2));
    }

    #[test]
    fn test_schema_validation() {
        let schema = create_test_schema();

        // Valid values
        let valid_values = vec![
            Value::Int(1),
            Value::String("Alice".to_string()),
            Value::Float(0.95),
            Value::Bool(true),
        ];
        assert!(schema.validate(&valid_values).is_ok());

        // Wrong number of values
        let wrong_count = vec![Value::Int(1), Value::String("Alice".to_string())];
        assert!(schema.validate(&wrong_count).is_err());

        // Wrong type
        let wrong_type = vec![
            Value::String("not an int".to_string()),
            Value::String("Alice".to_string()),
            Value::Float(0.95),
            Value::Bool(true),
        ];
        assert!(schema.validate(&wrong_type).is_err());
    }

    #[test]
    fn test_tuple_creation() {
        let schema = Arc::new(create_test_schema());
        let values = vec![
            Value::Int(1),
            Value::String("Alice".to_string()),
            Value::Float(0.95),
            Value::Bool(true),
        ];

        let tuple = Tuple::new(schema, values).unwrap();
        assert_eq!(tuple.len(), 4);
    }

    #[test]
    fn test_tuple_field_access() {
        let schema = Arc::new(create_test_schema());
        let values = vec![
            Value::Int(42),
            Value::String("Bob".to_string()),
            Value::Float(0.85),
            Value::Bool(false),
        ];

        let tuple = Tuple::new(schema, values).unwrap();

        assert_eq!(tuple.get("id"), Some(&Value::Int(42)));
        assert_eq!(tuple.get("name"), Some(&Value::String("Bob".to_string())));
        assert_eq!(tuple.get("score"), Some(&Value::Float(0.85)));
        assert_eq!(tuple.get("active"), Some(&Value::Bool(false)));
        assert_eq!(tuple.get("nonexistent"), None);
    }

    #[test]
    fn test_tuple_set_value() {
        let schema = Arc::new(create_test_schema());
        let values = vec![
            Value::Int(1),
            Value::String("Alice".to_string()),
            Value::Float(0.95),
            Value::Bool(true),
        ];

        let mut tuple = Tuple::new(schema, values).unwrap();

        // Valid update
        assert!(tuple.set("score", Value::Float(0.99)).is_ok());
        assert_eq!(tuple.get("score"), Some(&Value::Float(0.99)));

        // Invalid type
        assert!(tuple
            .set("score", Value::String("invalid".to_string()))
            .is_err());

        // Nonexistent field
        assert!(tuple.set("nonexistent", Value::Int(5)).is_err());
    }

    #[test]
    fn test_nullable_fields() {
        let schema = Schema::new(vec![
            Field::new("id", ValueType::Int),
            Field::new("optional_name", ValueType::String).nullable(),
        ]);

        let schema = Arc::new(schema);

        // Null value in nullable field
        let values = vec![Value::Int(1), Value::Null];
        assert!(Tuple::new(schema.clone(), values).is_ok());

        // Null value in non-nullable field
        let invalid_values = vec![Value::Null, Value::String("test".to_string())];
        assert!(Tuple::new(schema, invalid_values).is_err());
    }

    #[test]
    fn test_vector_field() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", ValueType::Int),
            Field::new("embedding", ValueType::Vector(3)),
        ]));

        // Valid vector
        let values = vec![Value::Int(1), Value::Vector(vec![0.1, 0.2, 0.3])];
        assert!(Tuple::new(schema.clone(), values).is_ok());

        // Wrong dimension
        let wrong_dim = vec![Value::Int(1), Value::Vector(vec![0.1, 0.2])];
        assert!(Tuple::new(schema, wrong_dim).is_err());
    }
}
