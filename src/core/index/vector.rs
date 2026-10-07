use super::flat::{cosine_with_norms, l2_norm, FlatVectors};
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

/// One IVF (inverted-file) bucket: a centroid plus the positions (into
/// `VectorIndex::store`) of the vectors assigned to it.
#[derive(Debug, Clone)]
struct Cluster {
    centroid: Vec<f32>,
    centroid_norm: f32,
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
///
/// Vectors are held once, contiguously, in a `FlatVectors` store with their
/// norms precomputed; scores are bit-identical to `COSINE_SIM`'s.
#[derive(Debug, Clone, Default)]
pub struct VectorIndex {
    store: FlatVectors,
    /// Built by the last `build()` call; empty if never built or too few
    /// vectors to bother.
    clusters: Vec<Cluster>,
    /// `store[..clustered_count]` are represented in `clusters`.
    /// `store[clustered_count..]` is the unclustered tail.
    clustered_count: usize,
}

impl VectorIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Run k-means (spherical: assignment and centroid quality are both
    /// judged by cosine similarity, matching this index's search metric) over
    /// every vector in `store` and return the resulting clusters.
    fn kmeans(store: &FlatVectors) -> Vec<Cluster> {
        use rayon::prelude::*;

        let n = store.len();
        let dim = store.dim();
        let num_clusters = ((n as f64).sqrt().round() as usize).clamp(2, MAX_CLUSTERS.min(n / 2));

        // Deterministic seeding: evenly-spaced picks from the input, so
        // results are reproducible (matters for LOAD DATASET rebuilding an
        // equivalent clustering from the same rows) without needing an RNG.
        let mut centroids: Vec<Vec<f32>> = (0..num_clusters)
            .map(|i| store.values(i * n / num_clusters).into_owned())
            .collect();
        let mut centroid_norms: Vec<f32> = centroids.iter().map(|c| l2_norm(c)).collect();

        let mut assignment: Vec<usize> = vec![0; n];

        for iteration in 0..KMEANS_ITERATIONS {
            // Assignment is independent per vector: parallel, and
            // deterministic (first best centroid wins, as before).
            let new_assignment: Vec<usize> = (0..n)
                .into_par_iter()
                .map(|i| {
                    let mut best = 0usize;
                    let mut best_sim = f32::MIN;
                    for (c_idx, c) in centroids.iter().enumerate() {
                        let sim = store.cosine(i, c, centroid_norms[c_idx]);
                        if sim > best_sim {
                            best_sim = sim;
                            best = c_idx;
                        }
                    }
                    best
                })
                .collect();
            let changed = iteration == 0 || new_assignment != assignment;
            assignment = new_assignment;

            let mut sums = vec![vec![0f32; dim]; num_clusters];
            let mut counts = vec![0usize; num_clusters];
            for (i, &c) in assignment.iter().enumerate() {
                counts[c] += 1;
                store.add_to(i, &mut sums[c]);
            }
            for c_idx in 0..num_clusters {
                if counts[c_idx] == 0 {
                    continue; // keep previous centroid if the cluster went empty
                }
                centroids[c_idx] = sums[c_idx]
                    .iter()
                    .map(|s| s / counts[c_idx] as f32)
                    .collect();
                centroid_norms[c_idx] = l2_norm(&centroids[c_idx]);
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
            .zip(centroid_norms)
            .zip(members_per_cluster)
            .filter(|(_, members)| !members.is_empty())
            .map(|((centroid, centroid_norm), members)| {
                let min_sim = members
                    .iter()
                    .map(|&idx| store.cosine(idx, &centroid, centroid_norm))
                    .fold(f32::MAX, f32::min);
                Cluster {
                    centroid,
                    centroid_norm,
                    members,
                    min_member_similarity: min_sim.clamp(-1.0, 1.0),
                }
            })
            .collect()
    }

    /// Upper bound on the similarity to `query` of any member of `cluster`,
    /// derived from the cluster's centroid similarity and angular radius via
    /// the spherical triangle inequality. If this bound doesn't pass the
    /// threshold, no member can either.
    fn cluster_upper_bound(cluster: &Cluster, query: &[f32], query_norm: f32) -> f32 {
        let centroid_sim =
            cosine_with_norms(query, query_norm, &cluster.centroid, cluster.centroid_norm)
                .clamp(-1.0, 1.0);
        let angle_to_centroid = centroid_sim.acos();
        let radius_angle = cluster.min_member_similarity.acos();
        let best_possible_angle = (angle_to_centroid - radius_angle).max(0.0);
        best_possible_angle.cos()
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
            clusters: self
                .clusters
                .iter()
                .map(|c| SnapshotCluster {
                    centroid: c.centroid.clone(),
                    members: c.members.iter().map(|&m| m as u32).collect(),
                    min_member_similarity: c.min_member_similarity,
                })
                .collect(),
        })
    }

    /// Restores a previously exported clustering without recomputing
    /// k-means. Only valid when the store was populated via `add()` in the
    /// exact same order as when the snapshot was taken -- `members` are
    /// positions in the store, not row IDs, so a reordering would silently
    /// point clusters at the wrong vectors. The caller confirms that via
    /// `content_hash` before calling this; `LOAD DATASET` re-inserts rows
    /// in their saved order, so it always holds there. Member positions and
    /// centroid dimensions are still bounds-checked here.
    pub fn restore_from_snapshot(&mut self, snapshot: VectorIndexSnapshot) -> Result<(), String> {
        if snapshot.clustered_count > self.store.len() {
            return Err(format!(
                "vector index snapshot expects at least {} vectors, only {} were added",
                snapshot.clustered_count,
                self.store.len()
            ));
        }
        let mut clusters = Vec::with_capacity(snapshot.clusters.len());
        for c in snapshot.clusters {
            if c.centroid.len() != self.store.dim() {
                return Err("vector index snapshot has the wrong dimension".to_string());
            }
            if c.members
                .iter()
                .any(|&m| m as usize >= snapshot.clustered_count)
            {
                return Err("vector index snapshot has an out-of-range member".to_string());
            }
            clusters.push(Cluster {
                centroid_norm: l2_norm(&c.centroid),
                centroid: c.centroid,
                members: c.members.into_iter().map(|m| m as usize).collect(),
                min_member_similarity: c.min_member_similarity,
            });
        }
        self.clusters = clusters;
        self.clustered_count = snapshot.clustered_count;
        Ok(())
    }

    /// Content hash of a vector column's values, in row order -- lets
    /// `LOAD DATASET` detect a persisted clustering snapshot that no
    /// longer matches the data it was built from (e.g. `data.parquet`
    /// edited independently of the snapshot) and fall back to a full
    /// rebuild instead of silently restoring a stale clustering.
    /// Non-`Vector` values are skipped rather than erroring -- a genuine
    /// vector-indexed column should never contain any, but this is a hash,
    /// not a validator.
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

/// One cluster as persisted: centroid, member positions, radius.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SnapshotCluster {
    #[serde(deserialize_with = "legacy::centroid")]
    centroid: Vec<f32>,
    #[serde(deserialize_with = "legacy::members")]
    members: Vec<u32>,
    min_member_similarity: f32,
}

/// `VectorIndex::snapshot`'s persistable output: everything `build()`
/// computes, minus the vectors themselves (those come back for free by
/// re-`add()`-ing the freshly loaded rows, in the same order).
///
/// Written by `SAVE DATASET` in the binary `vector_index_clusters.bin`
/// (`encode_snapshots`). The serde derive only exists to keep reading the
/// `vector_index_clusters.json` older versions wrote.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorIndexSnapshot {
    clustered_count: usize,
    clusters: Vec<SnapshotCluster>,
}

impl VectorIndexSnapshot {
    /// How many vectors (from the start of the column) the clustering covers.
    pub fn clustered_count(&self) -> usize {
        self.clustered_count
    }
}

/// What `SAVE DATASET` writes per vector-indexed column: the snapshot plus
/// the content hash it was computed from, so `LOAD DATASET` can tell a
/// still-valid snapshot from a stale one before trusting it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedVectorIndex {
    pub content_hash: String,
    pub snapshot: VectorIndexSnapshot,
}

const IVF_MAGIC: &[u8; 8] = b"LNLIVF1\0";

/// Binary layout of `vector_index_clusters.bin` (little-endian): magic
/// `LNLIVF1\0`, u32 column count, then per column (sorted by name): column
/// name and content hash (u32 length + UTF-8 each), u64 clustered count,
/// u32 cluster count, and per cluster: f32 radius (min member similarity),
/// the centroid (u64 length + f32s), the member positions (u64 length +
/// u32s).
pub fn encode_snapshots(
    snapshots: &std::collections::HashMap<String, PersistedVectorIndex>,
) -> Vec<u8> {
    let mut w = super::binio::Writer::new(IVF_MAGIC);
    let mut columns: Vec<&String> = snapshots.keys().collect();
    columns.sort();
    w.u32(columns.len() as u32);
    for column in columns {
        let p = &snapshots[column];
        w.str(column);
        w.str(&p.content_hash);
        w.u64(p.snapshot.clustered_count as u64);
        w.u32(p.snapshot.clusters.len() as u32);
        for c in &p.snapshot.clusters {
            w.f32(c.min_member_similarity);
            w.f32s(&c.centroid);
            w.u32s(&c.members);
        }
    }
    w.into_bytes()
}

pub fn decode_snapshots(
    bytes: &[u8],
) -> Result<std::collections::HashMap<String, PersistedVectorIndex>, String> {
    let mut r = super::binio::Reader::new(bytes, IVF_MAGIC)?;
    let mut out = std::collections::HashMap::new();
    for _ in 0..r.u32()? {
        let column = r.str()?;
        let content_hash = r.str()?;
        let clustered_count = r.u64()? as usize;
        let n = r.u32()?;
        let mut clusters = Vec::new();
        for _ in 0..n {
            let min_member_similarity = r.f32()?;
            let centroid = r.f32s()?;
            let members = r.u32s()?;
            clusters.push(SnapshotCluster {
                centroid,
                members,
                min_member_similarity,
            });
        }
        out.insert(
            column,
            PersistedVectorIndex {
                content_hash,
                snapshot: VectorIndexSnapshot {
                    clustered_count,
                    clusters,
                },
            },
        );
    }
    r.finish()?;
    Ok(out)
}

/// Reading `vector_index_clusters.json` from before the binary format,
/// where each centroid was a whole serialized `Tensor`.
mod legacy {
    use serde::{Deserialize, Deserializer};

    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Centroid {
        Plain(Vec<f32>),
        Tensor(crate::core::tensor::Tensor),
    }

    pub fn centroid<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<f32>, D::Error> {
        Ok(match Centroid::deserialize(d)? {
            Centroid::Plain(v) => v,
            Centroid::Tensor(t) => t.to_logical_vec(),
        })
    }

    pub fn members<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u32>, D::Error> {
        let v = Vec::<usize>::deserialize(d)?;
        v.into_iter()
            .map(|m| u32::try_from(m).map_err(serde::de::Error::custom))
            .collect()
    }
}

impl Index for VectorIndex {
    fn add(&mut self, row_id: usize, value: &Value) -> Result<(), String> {
        self.store.add_value(row_id, value)
    }

    fn lookup(&self, _value: &Value) -> Result<Vec<usize>, String> {
        Err("VectorIndex does not support exact value lookup".to_string())
    }

    fn memory_bytes(&self) -> usize {
        let clusters = self
            .clusters
            .iter()
            .map(|c| {
                std::mem::size_of::<Cluster>()
                    + c.centroid.capacity() * std::mem::size_of::<f32>()
                    + c.members.capacity() * std::mem::size_of::<usize>()
            })
            .sum::<usize>();
        self.store.memory_bytes() + clusters
    }

    fn search(&self, query: &Tensor, k: usize) -> Result<Vec<(usize, f32)>, String> {
        let query = super::binio::query_values(query);
        self.store.check_query(&query)?;
        let query_norm = l2_norm(&query);

        // Unclustered tail is always a candidate: it holds every vector
        // added since the last build(), which we have no cluster info for.
        let mut candidates: Vec<usize> = (self.clustered_count..self.store.len()).collect();

        if self.clusters.is_empty() {
            candidates = (0..self.store.len()).collect();
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
                .map(|(ci, c)| {
                    (
                        ci,
                        cosine_with_norms(&query, query_norm, &c.centroid, c.centroid_norm),
                    )
                })
                .collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let nprobe = (self.clusters.len() / 4).max(1);
            for &(ci, _) in ranked.iter().take(nprobe) {
                candidates.extend_from_slice(&self.clusters[ci].members);
            }
        }

        Ok(self.store.top_k(candidates.into_iter(), &query, k))
    }

    fn search_threshold(
        &self,
        query: &Tensor,
        threshold: f32,
        strict: bool,
    ) -> Result<Vec<(usize, f32)>, String> {
        let query = super::binio::query_values(query);
        self.store.check_query(&query)?;
        let query_norm = l2_norm(&query);
        let passes = |s: f32| {
            if strict {
                s > threshold
            } else {
                s >= threshold
            }
        };

        // Unclustered tail: no bound available, always scan.
        let mut candidates: Vec<usize> = (self.clustered_count..self.store.len()).collect();

        if self.clusters.is_empty() {
            candidates = (0..self.store.len()).collect();
        } else {
            for cluster in &self.clusters {
                if passes(Self::cluster_upper_bound(cluster, &query, query_norm)) {
                    candidates.extend_from_slice(&cluster.members);
                }
                // else: proven no member of this cluster can pass -- skip it entirely.
            }
        }

        let mut results = Vec::with_capacity(candidates.len());
        for i in candidates {
            let sim = self.store.cosine(i, &query, query_norm);
            if passes(sim) {
                results.push((self.store.row_id(i), sim));
            }
        }
        Ok(results)
    }

    fn index_type(&self) -> IndexType {
        IndexType::Vector
    }

    fn box_clone(&self) -> Box<dyn Index> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn build(&mut self) -> Result<(), String> {
        let n = self.store.len();
        self.clusters.clear();
        self.clustered_count = 0;

        if n < MIN_VECTORS_TO_CLUSTER {
            // Too few vectors for clustering to pay off; brute-force
            // fallback (both search paths treat an empty `clusters` as
            // "scan everything") stays exact and is already fast at this
            // size.
            return Ok(());
        }

        self.clusters = Self::kmeans(&self.store);
        self.clustered_count = n;
        Ok(())
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
