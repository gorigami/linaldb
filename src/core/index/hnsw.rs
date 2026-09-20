use super::{Index, IndexType};
use crate::core::tensor::Tensor;
use crate::core::value::Value;
use instant_distance::{Builder, HnswMap, Point as HnswPoint, Search};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Below this many vectors, HNSW graph-construction overhead isn't worth it
/// -- mirrors `vector::MIN_VECTORS_TO_CLUSTER`. `build()` leaves `graph`
/// unset and both search paths fall back to a full brute-force scan.
const MIN_VECTORS_TO_INDEX: usize = 16;

/// `ef_search` (the paper's recall/latency knob, see
/// `instant_distance::Builder::ef_search`) has to be fixed at build time --
/// it can't be adjusted per search call afterward. Set generously relative
/// to column size so a `LIMIT k` up to this bound gets full graph recall,
/// capped so build cost stays bounded on huge columns.
const EF_SEARCH_CAP: usize = 512;

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CosinePoint(Vec<f32>);

impl HnswPoint for CosinePoint {
    fn distance(&self, other: &Self) -> f32 {
        // instant-distance only requires a pseudo-metric (small = near,
        // large = far); 1 - cosine similarity is monotonic with angular
        // distance and matches this engine's own similarity ranking.
        // Clamped for float roundoff pushing an identical pair fractionally
        // above cos similarity 1.0.
        (1.0 - cosine_similarity(&self.0, &other.0)).max(0.0)
    }
}

/// An approximate-nearest-neighbor index for vector similarity search,
/// backed by `instant-distance`'s HNSW (Hierarchical Navigable Small World)
/// graph implementation.
///
/// Opted into explicitly via `CREATE VECTOR INDEX ... USING HNSW` (the
/// default, no `USING` clause, stays `vector::VectorIndex`'s IVF
/// clustering) -- additive, not a replacement. Only participates in top-k
/// similarity search (`VectorSearchExec`, i.e. `SEARCH ... LIMIT k`):
/// unlike IVF's clusters, an HNSW graph traversal has no cheap provable
/// bound on what it might have skipped, so it cannot safely accelerate an
/// *exact* predicate (`WHERE COSINE_SIM(...) > threshold`,
/// `CosineFilterExec`/`SimilarityJoinExec`) the way IVF's spherical-cap
/// upper bound does -- `search_threshold` below always scans every vector
/// directly instead of touching the graph, keeping that contract's
/// exactness real. See `PERFORMANCE_OPTIMIZATION_PLAN.md` Phase 1.
pub struct HnswIndex {
    /// All vectors ever added, in insertion order -- same role as
    /// `vector::VectorIndex::vectors`.
    vectors: Vec<(usize, Tensor)>,
    /// Built by the last `build()`/`restore_from_snapshot()`; `None` if
    /// never built or too few vectors to bother. `Arc`-wrapped since
    /// `HnswMap` isn't `Clone` and `Index: Send + Sync` needs cheap
    /// `box_clone()`.
    graph: Option<Arc<HnswMap<CosinePoint, usize>>>,
    /// `vectors[..indexed_count]` are represented in `graph`.
    /// `vectors[indexed_count..]` is the unindexed tail, always
    /// brute-force scanned by `search` alongside the graph -- mirrors
    /// `vector::VectorIndex::clustered_count`.
    indexed_count: usize,
}

// `HnswMap` doesn't implement `Debug` (see instant-distance's `Hnsw<P>`),
// so this can't be `#[derive(Debug)]`ed like `vector::VectorIndex` -- the
// `Index` trait requires `Debug`, satisfied here without exposing the
// graph's internals.
impl std::fmt::Debug for HnswIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HnswIndex")
            .field("vectors_len", &self.vectors.len())
            .field("indexed_count", &self.indexed_count)
            .field("has_graph", &self.graph.is_some())
            .finish()
    }
}

impl Default for HnswIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl HnswIndex {
    pub fn new() -> Self {
        Self {
            vectors: Vec::new(),
            graph: None,
            indexed_count: 0,
        }
    }

    fn brute_force(vectors: &[(usize, Tensor)], query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut scores: Vec<(usize, f32)> = vectors
            .iter()
            .map(|(row_id, v)| (*row_id, cosine_similarity(query, &v.data)))
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);
        scores
    }

    /// Exports the graph `build()` last computed, for `SAVE DATASET` to
    /// persist -- `None` if `build()` never ran or the vector count was
    /// below `MIN_VECTORS_TO_INDEX`. Unlike `VectorIndex`'s snapshot (which
    /// stores only cluster assignments and relies on rows being re-`add()`-ed
    /// in the same order to recover the actual vectors), the serialized
    /// `HnswMap` is fully self-contained -- its `values` are row ids
    /// directly, so restoring it doesn't depend on insertion order at all.
    pub fn snapshot(&self) -> Option<HnswIndexSnapshot> {
        let graph = self.graph.as_ref()?;
        Some(HnswIndexSnapshot {
            indexed_count: self.indexed_count,
            graph: serde_json::to_value(graph.as_ref()).ok()?,
        })
    }

    /// Restores a previously exported graph without recomputing it.
    pub fn restore_from_snapshot(&mut self, snapshot: HnswIndexSnapshot) -> Result<(), String> {
        if snapshot.indexed_count > self.vectors.len() {
            return Err(format!(
                "HNSW index snapshot expects at least {} vectors, only {} were added",
                snapshot.indexed_count,
                self.vectors.len()
            ));
        }
        let graph: HnswMap<CosinePoint, usize> =
            serde_json::from_value(snapshot.graph).map_err(|e| e.to_string())?;
        self.graph = Some(Arc::new(graph));
        self.indexed_count = snapshot.indexed_count;
        Ok(())
    }

    /// Content hash of a vector column's values, in row order. Shares
    /// `vector::VectorIndex`'s implementation -- both index types persist
    /// against the same invalidation contract (a stale snapshot is
    /// detected via mismatch, never silently trusted).
    pub fn content_hash(values: &[Value]) -> String {
        super::vector::VectorIndex::content_hash(values)
    }
}

/// `HnswIndex::snapshot`'s persistable output:
/// `datasets/<name>/hnsw_index_graphs.json`, keyed by column name, mirroring
/// `vector::PersistedVectorIndex`'s (content_hash, snapshot) shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswIndexSnapshot {
    indexed_count: usize,
    graph: serde_json::Value,
}

/// What `SAVE DATASET` writes to disk per HNSW-indexed column.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedHnswIndex {
    pub content_hash: String,
    pub snapshot: HnswIndexSnapshot,
}

impl Index for HnswIndex {
    fn add(&mut self, row_id: usize, value: &Value) -> Result<(), String> {
        match value {
            Value::Vector(data) => {
                use crate::core::tensor::{Shape, TensorId, TensorMetadata};
                let id = TensorId::new();
                let metadata = TensorMetadata::new(id, None);
                let tensor = Tensor::new(id, Shape::new(vec![data.len()]), data.clone(), metadata)
                    .map_err(|e| e.to_string())?;
                self.vectors.push((row_id, tensor));
                Ok(())
            }
            Value::Null => Ok(()),
            other => Err(format!("Cannot index {:?} as Vector", other)),
        }
    }

    fn lookup(&self, _value: &Value) -> Result<Vec<usize>, String> {
        Err("HnswIndex does not support exact value lookup".to_string())
    }

    fn search(&self, query: &Tensor, k: usize) -> Result<Vec<(usize, f32)>, String> {
        let tail = &self.vectors[self.indexed_count.min(self.vectors.len())..];
        let mut results = Self::brute_force(tail, &query.data, k);

        if let Some(graph) = &self.graph {
            let point = CosinePoint(query.data.to_vec());
            let mut search = Search::default();
            let graph_results = graph
                .search(&point, &mut search)
                .take(k)
                .map(|item| (*item.value, 1.0 - item.distance));
            results.extend(graph_results);
        } else if self.indexed_count > 0 {
            // Graph missing but indexed_count > 0 should never happen
            // (build()/restore_from_snapshot keep them in sync) -- fall back
            // to scanning the whole buffer rather than silently returning an
            // incomplete result.
            results = Self::brute_force(&self.vectors, &query.data, k);
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.dedup_by_key(|(row_id, _)| *row_id);
        results.truncate(k);
        Ok(results)
    }

    fn search_threshold(
        &self,
        query: &Tensor,
        threshold: f32,
        strict: bool,
    ) -> Result<Vec<(usize, f32)>, String> {
        // An HNSW traversal has no provable bound on what it skipped, so it
        // can never safely answer an *exact* predicate the way `search`'s
        // approximate top-k can be approximate -- always brute-force scan
        // every vector directly instead of touching the graph. See this
        // type's doc comment.
        let passes = |s: f32| {
            if strict {
                s > threshold
            } else {
                s >= threshold
            }
        };
        let mut results = Vec::with_capacity(self.vectors.len());
        for (row_id, v) in &self.vectors {
            let sim = cosine_similarity(&query.data, &v.data);
            if passes(sim) {
                results.push((*row_id, sim));
            }
        }
        Ok(results)
    }

    fn index_type(&self) -> IndexType {
        IndexType::Hnsw
    }

    fn box_clone(&self) -> Box<dyn Index> {
        Box::new(Self {
            vectors: self.vectors.clone(),
            graph: self.graph.clone(),
            indexed_count: self.indexed_count,
        })
    }

    fn build(&mut self) -> Result<(), String> {
        let n = self.vectors.len();
        self.graph = None;
        self.indexed_count = 0;

        if n < MIN_VECTORS_TO_INDEX {
            return Ok(());
        }

        let points: Vec<CosinePoint> = self
            .vectors
            .iter()
            .map(|(_, t)| CosinePoint(t.data.to_vec()))
            .collect();
        let values: Vec<usize> = self.vectors.iter().map(|(row_id, _)| *row_id).collect();

        let ef_search = n.clamp(1, EF_SEARCH_CAP);
        let graph = Builder::default()
            .ef_search(ef_search)
            .build(points, values);

        self.graph = Some(Arc::new(graph));
        self.indexed_count = n;
        Ok(())
    }

    fn export_snapshot(&self) -> Option<serde_json::Value> {
        serde_json::to_value(self.snapshot()?).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // One extra dimension beyond the "true" cluster axes, reserved for the
    // unindexed-tail test so a genuinely new direction never collides with
    // an already-indexed one -- mirrors `vector::tests`' fixture shape.
    const DIM: usize = 6;
    const PER_CLUSTER: usize = 20;
    const NUM_TRUE_CLUSTERS: usize = 5;

    fn point(axis: usize, jitter_step: usize) -> Value {
        let mut v = vec![0.0f32; DIM];
        v[axis] = 1.0;
        v[(axis + 1) % DIM] += 0.01 * (jitter_step as f32 / PER_CLUSTER as f32);
        Value::Vector(v)
    }

    fn one_hot_tensor(axis: usize) -> Tensor {
        let mut v = vec![0.0f32; DIM];
        v[axis] = 1.0;
        let id = crate::core::tensor::TensorId::new();
        let meta = crate::core::tensor::TensorMetadata::new(id, None);
        Tensor::new(id, crate::core::tensor::Shape::new(vec![DIM]), v, meta).unwrap()
    }

    fn well_separated_index() -> HnswIndex {
        let mut index = HnswIndex::new();
        let mut row_id = 0;
        for axis in 0..NUM_TRUE_CLUSTERS {
            for j in 0..PER_CLUSTER {
                index.add(row_id, &point(axis, j)).unwrap();
                row_id += 1;
            }
        }
        index.build().unwrap();
        index
    }

    #[test]
    fn build_actually_builds_a_graph_once_past_the_threshold() {
        let index = well_separated_index();
        assert!(
            index.graph.is_some(),
            "expected build() to construct a graph for {} well-separated vectors",
            NUM_TRUE_CLUSTERS * PER_CLUSTER
        );
        assert_eq!(index.indexed_count, NUM_TRUE_CLUSTERS * PER_CLUSTER);
    }

    #[test]
    fn small_index_has_no_snapshot_to_export() {
        let mut index = HnswIndex::new();
        for i in 0..(MIN_VECTORS_TO_INDEX - 1) {
            index.add(i, &point(0, i)).unwrap();
        }
        index.build().unwrap();
        assert!(index.snapshot().is_none());
    }

    #[test]
    fn small_dataset_stays_unindexed() {
        let mut index = HnswIndex::new();
        for i in 0..(MIN_VECTORS_TO_INDEX - 1) {
            index.add(i, &point(0, i)).unwrap();
        }
        index.build().unwrap();
        assert!(
            index.graph.is_none(),
            "expected no graph below MIN_VECTORS_TO_INDEX"
        );
    }

    #[test]
    fn snapshot_and_restore_round_trips_identical_search_results() {
        let original = well_separated_index();
        let snapshot = original.snapshot().expect("should have built a graph");

        let mut restored = HnswIndex::new();
        for axis in 0..NUM_TRUE_CLUSTERS {
            for j in 0..PER_CLUSTER {
                restored
                    .add(axis * PER_CLUSTER + j, &point(axis, j))
                    .unwrap();
            }
        }
        restored.restore_from_snapshot(snapshot).unwrap();

        let query = one_hot_tensor(2);
        let mut original_results = original.search(&query, 10).unwrap();
        let mut restored_results = restored.search(&query, 10).unwrap();
        original_results.sort_by_key(|(id, _)| *id);
        restored_results.sort_by_key(|(id, _)| *id);
        assert_eq!(
            original_results
                .iter()
                .map(|(id, _)| *id)
                .collect::<Vec<_>>(),
            restored_results
                .iter()
                .map(|(id, _)| *id)
                .collect::<Vec<_>>(),
            "a restored graph (deserialized, not recomputed) must answer identically"
        );
    }

    #[test]
    fn restore_from_snapshot_rejects_too_few_vectors() {
        let original = well_separated_index();
        let snapshot = original.snapshot().unwrap();

        let mut too_few = HnswIndex::new();
        too_few.add(0, &point(0, 0)).unwrap();
        assert!(too_few.restore_from_snapshot(snapshot).is_err());
    }

    #[test]
    fn content_hash_matches_vector_index_implementation() {
        let a = vec![Value::Vector(vec![1.0, 2.0, 3.0])];
        assert_eq!(
            HnswIndex::content_hash(&a),
            super::super::vector::VectorIndex::content_hash(&a)
        );
    }

    #[test]
    fn search_topk_finds_the_right_cluster_on_well_separated_data() {
        let index = well_separated_index();
        let query = one_hot_tensor(3);

        let results = index.search(&query, 10).unwrap();
        assert_eq!(results.len(), 10);
        for (row_id, score) in &results {
            assert!(
                (3 * PER_CLUSTER..4 * PER_CLUSTER).contains(row_id),
                "top-10 nearest neighbors of axis=3's centroid should all belong to its cluster, got row_id {}",
                row_id
            );
            assert!(
                *score > 0.99,
                "expected near-exact match, got score {}",
                score
            );
        }
    }

    #[test]
    fn search_threshold_is_exact_even_with_a_built_graph() {
        // The whole point of `search_threshold` bypassing the graph: an
        // exact predicate must return exactly the true cluster's members,
        // not whatever the approximate graph traversal happened to visit.
        let index = well_separated_index();
        let query = one_hot_tensor(2);

        let results = index.search_threshold(&query, 0.99, false).unwrap();
        let mut row_ids: Vec<usize> = results.into_iter().map(|(id, _)| id).collect();
        row_ids.sort_unstable();

        let expected: Vec<usize> = (2 * PER_CLUSTER..3 * PER_CLUSTER).collect();
        assert_eq!(
            row_ids, expected,
            "threshold search must return exactly the true cluster's members, no more, no less"
        );
    }

    #[test]
    fn unindexed_tail_added_after_build_is_still_found() {
        let mut index = well_separated_index();

        let tail_axis = DIM - 1;
        assert_eq!(
            tail_axis, NUM_TRUE_CLUSTERS,
            "tail axis must not overlap a true cluster axis"
        );
        let tail_start = NUM_TRUE_CLUSTERS * PER_CLUSTER;
        for j in 0..10 {
            index.add(tail_start + j, &point(tail_axis, j)).unwrap();
        }

        let query = one_hot_tensor(tail_axis);
        let results = index.search(&query, 10).unwrap();
        let mut row_ids: Vec<usize> = results.into_iter().map(|(id, _)| id).collect();
        row_ids.sort_unstable();

        assert_eq!(
            row_ids,
            (tail_start..tail_start + 10).collect::<Vec<_>>(),
            "vectors added after build() must still be found via the unindexed-tail scan"
        );
    }

    #[test]
    fn box_clone_produces_an_independently_searchable_index() {
        let original = well_separated_index();
        let cloned = Index::box_clone(&original);
        let query = one_hot_tensor(1);
        let results = cloned.search(&query, 5).unwrap();
        assert_eq!(results.len(), 5);
        for (row_id, _) in &results {
            assert!((PER_CLUSTER..2 * PER_CLUSTER).contains(row_id));
        }
    }
}
