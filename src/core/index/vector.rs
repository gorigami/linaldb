use super::{Index, IndexType};
use crate::core::tensor::Tensor;
use crate::core::value::Value;
use serde::{Deserialize, Serialize};

/// Below this many vectors, clustering overhead isn't worth it -- a brute
/// force scan is already fast, so `build()` leaves `clusters` empty and
/// search falls back to scanning everything (the original MVP behavior).
const MIN_VECTORS_TO_CLUSTER: usize = 64;
const MAX_CLUSTERS: usize = 256;
const KMEANS_ITERATIONS: usize = 10;

/// One IVF (inverted-file) bucket: a centroid plus the indices (into
/// `VectorIndex::vectors`) of the vectors assigned to it.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Cluster {
    centroid: Tensor,
    members: Vec<usize>,
    /// The minimum cosine similarity between the centroid and any of its
    /// members -- i.e. cos(max angular radius of the cluster). Combined with
    /// the query's similarity to the centroid, this lets `search_threshold`
    /// derive a provable upper bound on any member's similarity to the query
    /// (spherical triangle inequality) without touching the members
    /// themselves, so whole clusters can be safely skipped.
    min_member_similarity: f32,
}

/// An index for vector similarity search.
///
/// Implements IVF-style clustering: `build()` (called once after a batch of
/// `add()`s, e.g. `CREATE INDEX` backfill or `LOAD DATASET` rebuild) runs a
/// small k-means pass to group vectors into clusters. `search` (approximate
/// top-k) then only scans the nearest few clusters instead of every vector;
/// `search_threshold` (exact, used for `WHERE COSINE_SIM(...) > t`) scans a
/// cluster only if its provable similarity bound says it could contain a
/// passing entry.
///
/// Vectors added after the last `build()` (e.g. rows inserted into a
/// dataset that already has this index) aren't clustered -- they sit in an
/// "unclustered tail" that both search paths always scan in full, so
/// correctness never depends on `build()` having (re-)run recently, only
/// performance does.
#[derive(Debug)]
pub struct VectorIndex {
    /// All vectors ever added, in insertion order.
    vectors: Vec<(usize, Tensor)>,
    /// Built by the last `build()` call; empty if never built or too few
    /// vectors to bother.
    clusters: Vec<Cluster>,
    /// `vectors[..clustered_count]` are represented in `clusters`.
    /// `vectors[clustered_count..]` is the unclustered tail.
    clustered_count: usize,
}

impl Default for VectorIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorIndex {
    pub fn new() -> Self {
        Self {
            vectors: Vec::new(),
            clusters: Vec::new(),
            clustered_count: 0,
        }
    }

    /// Calculate cosine similarity between two tensors
    fn cosine_similarity(t1: &Tensor, t2: &Tensor) -> Result<f32, String> {
        if t1.shape != t2.shape {
            return Err(format!("Shape mismatch: {:?} vs {:?}", t1.shape, t2.shape));
        }

        if t1.data.len() != t2.data.len() {
            return Err("Data length mismatch".to_string());
        }

        let dot_product: f32 = t1.data.iter().zip(t2.data.iter()).map(|(a, b)| a * b).sum();
        let norm_t1: f32 = t1.data.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_t2: f32 = t2.data.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_t1 == 0.0 || norm_t2 == 0.0 {
            return Ok(0.0); // Handle zero vectors
        }

        Ok(dot_product / (norm_t1 * norm_t2))
    }

    /// Run k-means (spherical: assignment and centroid quality are both
    /// judged by cosine similarity, matching this index's search metric) over
    /// `self.vectors[0..n]` and return the resulting clusters.
    fn kmeans(vectors: &[(usize, Tensor)]) -> Result<Vec<Cluster>, String> {
        let n = vectors.len();
        let dim = vectors[0].1.data.len();
        let num_clusters = ((n as f64).sqrt().round() as usize).clamp(2, MAX_CLUSTERS.min(n / 2));

        // Deterministic seeding: evenly-spaced picks from the input, so
        // results are reproducible (matters for LOAD DATASET rebuilding an
        // equivalent clustering from the same rows) without needing an RNG.
        let mut centroids: Vec<Tensor> = (0..num_clusters)
            .map(|i| vectors[i * n / num_clusters].1.clone())
            .collect();

        let mut assignment: Vec<usize> = vec![0; n];

        for _ in 0..KMEANS_ITERATIONS {
            let mut changed = false;
            for (i, (_, v)) in vectors.iter().enumerate() {
                let mut best = 0usize;
                let mut best_sim = f32::MIN;
                for (c_idx, c) in centroids.iter().enumerate() {
                    let sim = Self::cosine_similarity(v, c)?;
                    if sim > best_sim {
                        best_sim = sim;
                        best = c_idx;
                    }
                }
                if assignment[i] != best {
                    changed = true;
                }
                assignment[i] = best;
            }

            let mut sums = vec![vec![0f32; dim]; num_clusters];
            let mut counts = vec![0usize; num_clusters];
            for (i, (_, v)) in vectors.iter().enumerate() {
                let c = assignment[i];
                counts[c] += 1;
                for (d, val) in v.data.iter().enumerate() {
                    sums[c][d] += val;
                }
            }
            for c_idx in 0..num_clusters {
                if counts[c_idx] == 0 {
                    continue; // keep previous centroid if the cluster went empty
                }
                let mean: Vec<f32> = sums[c_idx]
                    .iter()
                    .map(|s| s / counts[c_idx] as f32)
                    .collect();
                let id = crate::core::tensor::TensorId::new();
                let meta = crate::core::tensor::TensorMetadata::new(id, None);
                centroids[c_idx] =
                    Tensor::new(id, crate::core::tensor::Shape::new(vec![dim]), mean, meta)
                        .map_err(|e| e.to_string())?;
            }

            if !changed {
                break;
            }
        }

        let mut members_per_cluster: Vec<Vec<usize>> = vec![Vec::new(); num_clusters];
        for (i, &c) in assignment.iter().enumerate() {
            members_per_cluster[c].push(i);
        }

        centroids
            .into_iter()
            .zip(members_per_cluster)
            .filter(|(_, members)| !members.is_empty())
            .map(|(centroid, members)| {
                let mut min_sim = f32::MAX;
                for &idx in &members {
                    let sim = Self::cosine_similarity(&vectors[idx].1, &centroid)?;
                    if sim < min_sim {
                        min_sim = sim;
                    }
                }
                Ok(Cluster {
                    centroid,
                    members,
                    min_member_similarity: min_sim.clamp(-1.0, 1.0),
                })
            })
            .collect()
    }

    /// Upper bound on the similarity to `query` of any member of `cluster`,
    /// derived from the cluster's centroid similarity and angular radius via
    /// the spherical triangle inequality. If this bound doesn't pass the
    /// threshold, no member can either.
    fn cluster_upper_bound(cluster: &Cluster, query: &Tensor) -> Result<f32, String> {
        let centroid_sim = Self::cosine_similarity(query, &cluster.centroid)?.clamp(-1.0, 1.0);
        let angle_to_centroid = centroid_sim.acos();
        let radius_angle = cluster.min_member_similarity.acos();
        let best_possible_angle = (angle_to_centroid - radius_angle).max(0.0);
        Ok(best_possible_angle.cos())
    }

    /// Exports the clustering `build()` last computed, for `SAVE DATASET`
    /// to persist -- `None` if `build()` never ran or the vector count was
    /// below `MIN_VECTORS_TO_CLUSTER` (nothing expensive to have saved).
    pub fn snapshot(&self) -> Option<VectorIndexSnapshot> {
        if self.clusters.is_empty() {
            return None;
        }
        Some(VectorIndexSnapshot {
            clustered_count: self.clustered_count,
            clusters: self.clusters.clone(),
        })
    }

    /// Restores a previously exported clustering without recomputing
    /// k-means -- the entire point of persisting it (`build()`'s k-means
    /// pass is exactly the "full rebuild + blocking k-means on every ...
    /// LOAD DATASET" cost `SCIENTIFIC_ENGINE_EXPANSION_PLAN.md`'s audit
    /// flagged). Only valid when `self.vectors` was populated via `add()`
    /// in the exact same order as when the snapshot was taken -- `members`
    /// are indices *into* `self.vectors`, not row IDs, so a reordering
    /// would silently point clusters at the wrong vectors. The caller is
    /// responsible for confirming that via `content_hash` before calling
    /// this (not re-checked here); `LOAD DATASET` re-inserts rows in their
    /// saved order, so this always holds there.
    pub fn restore_from_snapshot(&mut self, snapshot: VectorIndexSnapshot) -> Result<(), String> {
        if snapshot.clustered_count > self.vectors.len() {
            return Err(format!(
                "vector index snapshot expects at least {} vectors, only {} were added",
                snapshot.clustered_count,
                self.vectors.len()
            ));
        }
        self.clusters = snapshot.clusters;
        self.clustered_count = snapshot.clustered_count;
        Ok(())
    }

    /// Content hash of a vector column's values, in row order -- lets
    /// `LOAD DATASET` detect a persisted clustering snapshot that no
    /// longer matches the data it was built from (e.g. `data.parquet`
    /// edited independently of `vector_index_clusters.json`) and fall
    /// back to a full rebuild instead of silently restoring a stale
    /// clustering. Non-`Vector` values are skipped rather than erroring --
    /// a genuine vector-indexed column should never contain any, but this
    /// is a hash, not a validator.
    pub fn content_hash(values: &[Value]) -> String {
        let mut bytes = Vec::new();
        for v in values {
            if let Value::Vector(data) = v {
                for f in data {
                    bytes.extend_from_slice(&f.to_le_bytes());
                }
            }
        }
        crate::core::provenance::compute_content_hash(&bytes)
    }
}

/// `VectorIndex::snapshot`'s persistable output: everything `build()`
/// computes, minus the vectors themselves (those come back for free by
/// re-`add()`-ing the freshly loaded rows, in the same order, which is
/// what makes `members`'s indices-into-`vectors` still valid on restore).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorIndexSnapshot {
    clustered_count: usize,
    clusters: Vec<Cluster>,
}

/// What `SAVE DATASET` actually writes to disk per vector-indexed column
/// (`datasets/<name>/vector_index_clusters.json`, keyed by column name):
/// the snapshot plus the content hash it was computed from, so `LOAD
/// DATASET` can tell a still-valid snapshot from a stale one before
/// trusting it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedVectorIndex {
    pub content_hash: String,
    pub snapshot: VectorIndexSnapshot,
}

impl Index for VectorIndex {
    fn add(&mut self, row_id: usize, value: &Value) -> Result<(), String> {
        match value {
            Value::Vector(data) => {
                // Convert Vec<f32> to Tensor (MVP: Shape is inferred as [len])
                use crate::core::tensor::{Shape, Tensor, TensorId, TensorMetadata};
                let id = TensorId::new();
                let metadata = TensorMetadata::new(id, None);
                let tensor = Tensor::new(id, Shape::new(vec![data.len()]), data.clone(), metadata)
                    .map_err(|e| e.to_string())?;

                self.vectors.push((row_id, tensor));
                Ok(())
            }
            Value::Bool(_) => Err("Cannot index Boolean as Vector".to_string()),
            Value::Int(_) => Err("Cannot index Int as Vector".to_string()),
            Value::String(_) => Err("Cannot index String as Vector".to_string()),
            Value::Null => Ok(()),
            Value::Float(_) => Err("Cannot index Float as Vector".to_string()),
            Value::Float64(_) => Err("Cannot index Double as Vector".to_string()),
            Value::Matrix(_) => Err("Cannot index Matrix as Vector".to_string()),
        }
    }

    fn lookup(&self, _value: &Value) -> Result<Vec<usize>, String> {
        Err("VectorIndex does not support exact value lookup".to_string())
    }

    fn search(&self, query: &Tensor, k: usize) -> Result<Vec<(usize, f32)>, String> {
        // Unclustered tail is always a candidate: it holds every vector
        // added since the last build(), which we have no cluster info for.
        let mut candidate_indices: Vec<usize> =
            (self.clustered_count..self.vectors.len()).collect();

        if self.clusters.is_empty() {
            candidate_indices = (0..self.vectors.len()).collect();
        } else {
            // Rank clusters by centroid similarity to the query, then probe
            // (fully scan) only the nearest few -- this is the approximate
            // part of IVF: a true nearest neighbor assigned to a
            // non-probed cluster can be missed. That's an accepted
            // trade-off for `search`'s top-k use (VectorSearchExec); exact
            // predicates go through `search_threshold` instead.
            let mut ranked: Vec<(usize, f32)> = self
                .clusters
                .iter()
                .enumerate()
                .map(|(ci, c)| Self::cosine_similarity(query, &c.centroid).map(|s| (ci, s)))
                .collect::<Result<_, String>>()?;
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let nprobe = (self.clusters.len() / 4).max(1);
            for &(ci, _) in ranked.iter().take(nprobe) {
                candidate_indices.extend_from_slice(&self.clusters[ci].members);
            }
        }

        let mut scores: Vec<(usize, f32)> = candidate_indices
            .into_iter()
            .map(|idx| {
                let (row_id, v) = &self.vectors[idx];
                Self::cosine_similarity(query, v).map(|s| (*row_id, s))
            })
            .collect::<Result<_, String>>()?;

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scores.into_iter().take(k).collect())
    }

    fn search_threshold(
        &self,
        query: &Tensor,
        threshold: f32,
        strict: bool,
    ) -> Result<Vec<(usize, f32)>, String> {
        let passes = |s: f32| {
            if strict {
                s > threshold
            } else {
                s >= threshold
            }
        };

        // Unclustered tail: no bound available, always scan.
        let mut candidate_indices: Vec<usize> =
            (self.clustered_count..self.vectors.len()).collect();

        if self.clusters.is_empty() {
            candidate_indices = (0..self.vectors.len()).collect();
        } else {
            for cluster in &self.clusters {
                let upper_bound = Self::cluster_upper_bound(cluster, query)?;
                if passes(upper_bound) {
                    candidate_indices.extend_from_slice(&cluster.members);
                }
                // else: proven no member of this cluster can pass -- skip it entirely.
            }
        }

        let mut results = Vec::with_capacity(candidate_indices.len());
        for idx in candidate_indices {
            let (row_id, v) = &self.vectors[idx];
            let sim = Self::cosine_similarity(query, v)?;
            if passes(sim) {
                results.push((*row_id, sim));
            }
        }
        Ok(results)
    }

    fn index_type(&self) -> IndexType {
        IndexType::Vector
    }

    fn box_clone(&self) -> Box<dyn Index> {
        Box::new(Self {
            vectors: self.vectors.clone(),
            clusters: self.clusters.clone(),
            clustered_count: self.clustered_count,
        })
    }

    fn build(&mut self) -> Result<(), String> {
        let n = self.vectors.len();
        self.clusters.clear();
        self.clustered_count = 0;

        if n < MIN_VECTORS_TO_CLUSTER {
            // Too few vectors for clustering to pay off; brute-force
            // fallback (both search paths treat an empty `clusters` as
            // "scan everything") stays exact and is already fast at this
            // size.
            return Ok(());
        }

        self.clusters = Self::kmeans(&self.vectors)?;
        self.clustered_count = n;
        Ok(())
    }

    fn export_snapshot(&self) -> Option<serde_json::Value> {
        serde_json::to_value(self.snapshot()?).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // One extra dimension beyond the 5 "true" cluster axes (0..5), reserved
    // for the unclustered-tail test below so that a genuinely new direction
    // (axis 5) never collides with an already-clustered one.
    const DIM: usize = 6;
    const PER_CLUSTER: usize = 60;
    const NUM_TRUE_CLUSTERS: usize = 5;

    /// A one-hot vector at `axis`, with a small deterministic perturbation
    /// added to a different axis so points within a "true" cluster aren't
    /// all bit-for-bit identical, while staying far closer (in cosine
    /// similarity) to their own axis than to any other.
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

    /// Populates a `VectorIndex` with `NUM_TRUE_CLUSTERS * PER_CLUSTER`
    /// vectors arranged in well-separated groups (enough to clear
    /// `MIN_VECTORS_TO_CLUSTER`), and calls `build()`.
    fn well_separated_index() -> VectorIndex {
        let mut index = VectorIndex::new();
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
    fn build_actually_clusters_once_past_the_threshold() {
        let index = well_separated_index();
        assert!(
            !index.clusters.is_empty(),
            "expected build() to produce clusters for {} well-separated vectors",
            NUM_TRUE_CLUSTERS * PER_CLUSTER
        );
        assert_eq!(index.clustered_count, NUM_TRUE_CLUSTERS * PER_CLUSTER);
    }

    #[test]
    fn small_index_has_no_snapshot_to_export() {
        let mut index = VectorIndex::new();
        for i in 0..(MIN_VECTORS_TO_CLUSTER - 1) {
            index.add(i, &point(0, i)).unwrap();
        }
        index.build().unwrap();
        assert!(index.snapshot().is_none());
    }

    #[test]
    fn snapshot_and_restore_round_trips_identical_search_results() {
        let original = well_separated_index();
        let snapshot = original.snapshot().expect("should have clustered");

        // Rebuild a fresh index from the same points, in the same order,
        // then restore from the snapshot instead of calling build() --
        // mirrors exactly what LOAD DATASET does (re-add() the reloaded
        // rows, then restore_from_snapshot instead of a full k-means pass).
        let mut restored = VectorIndex::new();
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
        );
    }

    #[test]
    fn restore_from_snapshot_rejects_too_few_vectors() {
        let original = well_separated_index();
        let snapshot = original.snapshot().unwrap();

        let mut too_few = VectorIndex::new();
        too_few.add(0, &point(0, 0)).unwrap();
        assert!(too_few.restore_from_snapshot(snapshot).is_err());
    }

    #[test]
    fn content_hash_is_stable_and_sensitive_to_data_changes() {
        let a = vec![Value::Vector(vec![1.0, 2.0, 3.0])];
        let b = vec![Value::Vector(vec![1.0, 2.0, 3.0])];
        let c = vec![Value::Vector(vec![1.0, 2.0, 3.1])];
        assert_eq!(VectorIndex::content_hash(&a), VectorIndex::content_hash(&b));
        assert_ne!(VectorIndex::content_hash(&a), VectorIndex::content_hash(&c));
    }

    #[test]
    fn content_hash_is_order_sensitive() {
        let a = vec![Value::Vector(vec![1.0, 0.0]), Value::Vector(vec![0.0, 1.0])];
        let b = vec![Value::Vector(vec![0.0, 1.0]), Value::Vector(vec![1.0, 0.0])];
        assert_ne!(VectorIndex::content_hash(&a), VectorIndex::content_hash(&b));
    }

    #[test]
    fn small_dataset_stays_unclustered() {
        let mut index = VectorIndex::new();
        for i in 0..(MIN_VECTORS_TO_CLUSTER - 1) {
            index.add(i, &point(0, i)).unwrap();
        }
        index.build().unwrap();
        assert!(
            index.clusters.is_empty(),
            "expected no clustering below MIN_VECTORS_TO_CLUSTER"
        );
    }

    #[test]
    fn search_threshold_is_exact_on_clustered_data() {
        let index = well_separated_index();
        let query = one_hot_tensor(2);

        let results = index.search_threshold(&query, 0.99, false).unwrap();
        let mut row_ids: Vec<usize> = results.into_iter().map(|(id, _)| id).collect();
        row_ids.sort_unstable();

        // Cluster axis=2 occupies row ids [2*PER_CLUSTER, 3*PER_CLUSTER).
        let expected: Vec<usize> = (2 * PER_CLUSTER..3 * PER_CLUSTER).collect();
        assert_eq!(
            row_ids, expected,
            "threshold search must return exactly the true cluster's members, no more, no less"
        );
    }

    #[test]
    fn search_topk_finds_the_right_cluster_on_clustered_data() {
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
    fn unclustered_tail_added_after_build_is_still_found() {
        let mut index = well_separated_index();

        // Simulate rows inserted after CREATE INDEX already ran (and
        // therefore after the last build()): a brand new 6th cluster, along
        // axis 5, that exists only in the "unclustered tail" and was never
        // seen by any of the 5 true clusters above (which only used axes
        // 0..5).
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
        let results = index.search_threshold(&query, 0.99, false).unwrap();
        let mut row_ids: Vec<usize> = results.into_iter().map(|(id, _)| id).collect();
        row_ids.sort_unstable();

        assert_eq!(
            row_ids,
            (tail_start..tail_start + 10).collect::<Vec<_>>(),
            "vectors added after build() must still be found via the unclustered-tail scan"
        );
    }
}
