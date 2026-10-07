//! Sorted scalar index (`CREATE SORTED INDEX ON ds(col)`): the column's
//! values in order, for range lookups (`col BETWEEN a AND b`, `col >= a`,
//! ...) by binary search instead of a full scan. Used by the planner for
//! `WHERE` ranges and by `SEARCH ... PREFILTER` for a per-query window, such
//! as a precursor mass ± tolerance.

use super::{Index, IndexType};
use crate::core::tensor::Tensor;
use crate::core::value::Value;
use std::cmp::Ordering;

#[derive(Debug, Clone, Default)]
pub struct SortedIndex {
    /// `(value, row_id)` sorted by value (then row id), covering every row
    /// added before the last `build()`.
    sorted: Vec<(Value, usize)>,
    /// Rows added since the last `build()`, unsorted; always scanned.
    tail: Vec<(Value, usize)>,
}

fn order(a: &Value, b: &Value) -> Ordering {
    // `add` only accepts mutually comparable scalars, so `compare` is
    // always `Some` here.
    a.compare(b).unwrap_or(Ordering::Equal)
}

/// Does `v` satisfy `v <op> bound`?
fn satisfies(v: &Value, op: &str, bound: &Value) -> bool {
    match (op, v.compare(bound)) {
        (_, None) => false,
        ("<", Some(o)) => o == Ordering::Less,
        ("<=", Some(o)) => o != Ordering::Greater,
        (">", Some(o)) => o == Ordering::Greater,
        (">=", Some(o)) => o != Ordering::Less,
        ("=", Some(o)) => o == Ordering::Equal,
        _ => false,
    }
}

impl SortedIndex {
    pub fn new() -> Self {
        Self::default()
    }

    fn kind(v: &Value) -> Option<u8> {
        match v {
            Value::Int(_) | Value::Float(_) | Value::Float64(_) => Some(0),
            Value::String(_) => Some(1),
            _ => None,
        }
    }

    /// Row ids (ascending) whose value satisfies every `(op, bound)`
    /// constraint, `op` one of `<`, `<=`, `>`, `>=`, `=`. A bound that can't
    /// be compared with the column's values matches nothing.
    pub fn range(&self, constraints: &[(String, Value)]) -> Vec<usize> {
        // Narrow the sorted part to [lo, hi) by binary search, then check
        // each constraint exactly on what's left.
        let mut lo = 0;
        let mut hi = self.sorted.len();
        for (op, bound) in constraints {
            let below = |v: &Value| v.compare(bound).map(|o| o == Ordering::Less);
            let at_or_below = |v: &Value| v.compare(bound).map(|o| o != Ordering::Greater);
            match op.as_str() {
                ">" => {
                    lo = lo.max(
                        self.sorted
                            .partition_point(|(v, _)| at_or_below(v).unwrap_or(true)),
                    )
                }
                ">=" => {
                    lo = lo.max(
                        self.sorted
                            .partition_point(|(v, _)| below(v).unwrap_or(true)),
                    )
                }
                "<" => {
                    hi = hi.min(
                        self.sorted
                            .partition_point(|(v, _)| below(v).unwrap_or(false)),
                    )
                }
                "<=" => {
                    hi = hi.min(
                        self.sorted
                            .partition_point(|(v, _)| at_or_below(v).unwrap_or(false)),
                    )
                }
                "=" => {
                    lo = lo.max(
                        self.sorted
                            .partition_point(|(v, _)| below(v).unwrap_or(true)),
                    );
                    hi = hi.min(
                        self.sorted
                            .partition_point(|(v, _)| at_or_below(v).unwrap_or(false)),
                    );
                }
                _ => {}
            }
        }
        let keep = |v: &Value| constraints.iter().all(|(op, b)| satisfies(v, op, b));
        let mut ids: Vec<usize> = if lo < hi {
            self.sorted[lo..hi]
                .iter()
                .filter(|(v, _)| keep(v))
                .map(|(_, id)| *id)
                .collect()
        } else {
            Vec::new()
        };
        ids.extend(self.tail.iter().filter(|(v, _)| keep(v)).map(|(_, id)| *id));
        ids.sort_unstable();
        ids
    }
}

impl Index for SortedIndex {
    fn add(&mut self, row_id: usize, value: &Value) -> Result<(), String> {
        match value {
            Value::Null => return Ok(()),
            Value::Float(f) if f.is_nan() => {
                return Err("SORTED index: NaN has no ordering".to_string())
            }
            Value::Float64(f) if f.is_nan() => {
                return Err("SORTED index: NaN has no ordering".to_string())
            }
            _ => {}
        }
        let Some(kind) = Self::kind(value) else {
            return Err(format!(
                "SORTED index supports Int, Float, Float64 and String columns, got {:?}",
                value.value_type()
            ));
        };
        let existing = self.sorted.first().or(self.tail.first()).map(|(v, _)| v);
        if existing.is_some_and(|e| Self::kind(e) != Some(kind)) {
            return Err("SORTED index: a column can't mix numbers and strings".to_string());
        }
        self.tail.push((value.clone(), row_id));
        Ok(())
    }

    fn lookup(&self, value: &Value) -> Result<Vec<usize>, String> {
        Ok(self.range(&[("=".to_string(), value.clone())]))
    }

    fn search(&self, _query: &Tensor, _k: usize) -> Result<Vec<(usize, f32)>, String> {
        Err("SORTED index does not support vector search".to_string())
    }

    fn search_threshold(
        &self,
        _query: &Tensor,
        _threshold: f32,
        _strict: bool,
    ) -> Result<Vec<(usize, f32)>, String> {
        Err("SORTED index does not support vector search".to_string())
    }

    fn index_type(&self) -> IndexType {
        IndexType::Sorted
    }

    fn memory_bytes(&self) -> usize {
        let heap = |v: &Value| v.estimated_bytes() - std::mem::size_of::<Value>();
        (self.sorted.capacity() + self.tail.capacity()) * std::mem::size_of::<(Value, usize)>()
            + self
                .sorted
                .iter()
                .chain(self.tail.iter())
                .map(|(v, _)| heap(v))
                .sum::<usize>()
    }

    fn box_clone(&self) -> Box<dyn Index> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn build(&mut self) -> Result<(), String> {
        self.sorted.append(&mut self.tail);
        self.sorted
            .sort_by(|a, b| order(&a.0, &b.0).then(a.1.cmp(&b.1)));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn index(values: &[Value]) -> SortedIndex {
        let mut idx = SortedIndex::new();
        for (i, v) in values.iter().enumerate() {
            idx.add(i, v).unwrap();
        }
        idx.build().unwrap();
        idx
    }

    fn c(op: &str, v: Value) -> (String, Value) {
        (op.to_string(), v)
    }

    #[test]
    fn ranges_match_a_linear_scan() {
        let values: Vec<Value> = (0..200)
            .map(|i| Value::Float64(((i * 37) % 101) as f64 * 0.5))
            .collect();
        let idx = index(&values);
        let cases = vec![
            vec![c(">=", Value::Float64(10.0)), c("<=", Value::Float64(20.0))],
            vec![c(">", Value::Float64(10.0)), c("<", Value::Float64(20.0))],
            vec![c("=", Value::Float64(12.5))],
            vec![c("<", Value::Int(3))],
            vec![c(">", Value::Float64(1e9))],
            vec![c(">=", Value::Float(49.5))],
        ];
        for cs in cases {
            let expected: Vec<usize> = values
                .iter()
                .enumerate()
                .filter(|(_, v)| cs.iter().all(|(op, b)| satisfies(v, op, b)))
                .map(|(i, _)| i)
                .collect();
            assert_eq!(idx.range(&cs), expected, "{:?}", cs);
        }
    }

    #[test]
    fn tail_rows_are_found_and_nulls_skipped() {
        let mut idx = index(&[Value::Int(5), Value::Null, Value::Int(1)]);
        idx.add(3, &Value::Int(4)).unwrap();
        assert_eq!(idx.range(&[c(">=", Value::Int(4))]), vec![0, 3]);
        assert_eq!(idx.lookup(&Value::Int(1)).unwrap(), vec![2]);
    }

    #[test]
    fn rejects_unorderable_values() {
        let mut idx = SortedIndex::new();
        assert!(idx.add(0, &Value::Float(f32::NAN)).is_err());
        assert!(idx.add(0, &Value::Vector(vec![1.0])).is_err());
        idx.add(0, &Value::Int(1)).unwrap();
        assert!(idx.add(1, &Value::String("a".into())).is_err());
    }
}
