//! HNSW (Hierarchical Navigable Small World) vector index.
//!
//! Implemented here rather than through a crate so that the graph can share
//! the index's one `FlatVectors` copy of the vectors (a library graph keeps
//! its own copy of every point), store neighbor lists as compact `u32`
//! arrays, persist as just those arrays, and score candidates with exactly
//! the cosine `COSINE_SIM` computes. Follows Malkov & Yashunin (2018):
//! greedy descent through the upper layers, beam search (`ef`) on each
//! layer, and the neighbor-selection heuristic that keeps a candidate only
//! if it is closer to the new node than to every neighbor already kept,
//! topped up with the pruned candidates.

use super::flat::{l2_norm, FlatVectors};
use super::{Index, IndexType};
use crate::core::tensor::Tensor;
use crate::core::value::Value;
use std::cell::RefCell;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::sync::Arc;

/// Below this many vectors, graph construction isn't worth it -- mirrors
/// `vector::MIN_VECTORS_TO_CLUSTER`. `build()` leaves `graph` unset and
/// search falls back to a full scan.
const MIN_VECTORS_TO_INDEX: usize = 16;
/// Neighbors kept per node on layers >= 1 (the paper's `M`).
const M: usize = 32;
/// Neighbors kept per node on layer 0 (`M_max0 = 2M`).
const M0: usize = 2 * M;
/// Beam width while building.
const EF_CONSTRUCTION: usize = 256;
/// Beam width while searching; widened to `k` for a larger `LIMIT`.
const EF_SEARCH: usize = 768;
/// Upper bound on how many nodes are inserted per parallel step.
const MAX_BATCH: usize = 1024;
const MAX_LEVEL: u8 = 15;
/// Empty neighbor slot.
const NONE: u32 = u32::MAX;

/// A node and its similarity to the current query. Ordered by similarity,
/// then by lower node id, so every heap operation -- and therefore the
/// whole build -- is deterministic.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Cand {
    sim: f32,
    id: u32,
}
impl Eq for Cand {}
impl PartialOrd for Cand {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Cand {
    fn cmp(&self, other: &Self) -> Ordering {
        self.sim
            .total_cmp(&other.sim)
            .then_with(|| other.id.cmp(&self.id))
    }
}

/// The graph over `store[..n]` (`n = levels.len()`).
#[derive(Debug, Clone)]
struct Graph {
    entry: u32,
    max_level: u8,
    levels: Vec<u8>,
    /// `M0` slots per node.
    layer0: Vec<u32>,
    /// Start of each node's upper-layer slots in `upper_links` (`NONE` for
    /// a level-0 node). A level-`L` node has `L * M` slots: layer `l` at
    /// `offset + (l - 1) * M`.
    upper_offset: Vec<u32>,
    upper_links: Vec<u32>,
}

impl Graph {
    fn with_levels(levels: Vec<u8>) -> Self {
        let n = levels.len();
        let mut upper_offset = vec![NONE; n];
        let mut total = 0usize;
        for (i, &l) in levels.iter().enumerate() {
            if l > 0 {
                upper_offset[i] = total as u32;
                total += l as usize * M;
            }
        }
        Graph {
            entry: NONE,
            max_level: 0,
            levels,
            layer0: vec![NONE; n * M0],
            upper_offset,
            upper_links: vec![NONE; total],
        }
    }

    fn slot_range(&self, node: u32, layer: u8) -> std::ops::Range<usize> {
        let node = node as usize;
        if layer == 0 {
            node * M0..(node + 1) * M0
        } else {
            let start = self.upper_offset[node] as usize + (layer as usize - 1) * M;
            start..start + M
        }
    }

    fn links(&self, node: u32, layer: u8) -> impl Iterator<Item = u32> + '_ {
        let range = self.slot_range(node, layer);
        let slots = if layer == 0 {
            &self.layer0[range]
        } else {
            &self.upper_links[range]
        };
        slots.iter().copied().take_while(|&x| x != NONE)
    }

    fn set_links(&mut self, node: u32, layer: u8, links: &[u32]) {
        let range = self.slot_range(node, layer);
        let slots = if layer == 0 {
            &mut self.layer0[range]
        } else {
            &mut self.upper_links[range]
        };
        slots.fill(NONE);
        slots[..links.len()].copy_from_slice(links);
    }

    fn memory_bytes(&self) -> usize {
        self.levels.capacity()
            + 4 * (self.layer0.capacity()
                + self.upper_offset.capacity()
                + self.upper_links.capacity())
    }
}

/// Deterministic level for node `i`: `floor(-ln(U) / ln(M))`, with `U`
/// drawn from a SplitMix64 hash of `i`, so the same rows always build the
/// same graph.
fn level_for(i: usize) -> u8 {
    let mut x = (i as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    let u = ((x >> 11) as f64 + 1.0) / (1u64 << 53) as f64; // (0, 1]
    let level = (-u.ln() / (M as f64).ln()).floor();
    (level as u8).min(MAX_LEVEL)
}

thread_local! {
    /// Per-thread visited marks: `stamps[i] == generation` means visited.
    static VISITED: RefCell<(Vec<u32>, u32)> = const { RefCell::new((Vec::new(), 0)) };
}

/// Runs `f` with a cleared visited set sized for `n` nodes.
fn with_visited<R>(n: usize, f: impl FnOnce(&mut dyn FnMut(u32) -> bool) -> R) -> R {
    VISITED.with(|cell| {
        let mut guard = cell.borrow_mut();
        let (stamps, generation) = &mut *guard;
        if stamps.len() < n {
            stamps.resize(n, 0);
        }
        *generation = generation.wrapping_add(1);
        if *generation == 0 {
            stamps.fill(0);
            *generation = 1;
        }
        let g = *generation;
        let mut first_visit = |id: u32| {
            let slot = &mut stamps[id as usize];
            if *slot == g {
                false
            } else {
                *slot = g;
                true
            }
        };
        f(&mut first_visit)
    })
}

/// Beam search on one layer from `entry_points`; returns up to `ef` nodes,
/// most similar first.
fn search_layer(
    store: &FlatVectors,
    graph: &Graph,
    query: &[f32],
    query_norm: f32,
    entry_points: &[u32],
    ef: usize,
    layer: u8,
) -> Vec<Cand> {
    with_visited(graph.levels.len(), |first_visit| {
        let mut candidates: BinaryHeap<Cand> = BinaryHeap::new();
        let mut results: BinaryHeap<Reverse<Cand>> = BinaryHeap::new();
        for &e in entry_points {
            if first_visit(e) {
                let c = Cand {
                    sim: store.cosine(e as usize, query, query_norm),
                    id: e,
                };
                candidates.push(c);
                results.push(Reverse(c));
                if results.len() > ef {
                    results.pop();
                }
            }
        }
        while let Some(c) = candidates.pop() {
            let worst = results.peek().map_or(f32::MIN, |r| r.0.sim);
            if results.len() >= ef && c.sim < worst {
                break;
            }
            for nb in graph.links(c.id, layer) {
                if !first_visit(nb) {
                    continue;
                }
                let sim = store.cosine(nb as usize, query, query_norm);
                let worst = results.peek().map_or(f32::MIN, |r| r.0.sim);
                if results.len() < ef || sim > worst {
                    let n = Cand { sim, id: nb };
                    candidates.push(n);
                    results.push(Reverse(n));
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }
        let mut out: Vec<Cand> = results.into_iter().map(|r| r.0).collect();
        out.sort_by(|a, b| b.cmp(a));
        out
    })
}

/// `search_layer` on layer 0 where only nodes passing `accept` can enter the
/// results; every node can still be traversed, so the search reaches
/// accepted nodes behind rejected ones. Stops once `ef` accepted nodes are
/// held and the best unexplored node can't beat the worst of them -- with
/// few accepted nodes that can mean walking most of the graph.
fn search_layer_filtered(
    store: &FlatVectors,
    graph: &Graph,
    query: &[f32],
    query_norm: f32,
    entry_points: &[u32],
    ef: usize,
    accept: &dyn Fn(u32) -> bool,
) -> Vec<Cand> {
    with_visited(graph.levels.len(), |first_visit| {
        let mut candidates: BinaryHeap<Cand> = BinaryHeap::new();
        let mut results: BinaryHeap<Reverse<Cand>> = BinaryHeap::new();
        let offer = |c: Cand, results: &mut BinaryHeap<Reverse<Cand>>| {
            if accept(c.id) {
                results.push(Reverse(c));
                if results.len() > ef {
                    results.pop();
                }
            }
        };
        for &e in entry_points {
            if first_visit(e) {
                let c = Cand {
                    sim: store.cosine(e as usize, query, query_norm),
                    id: e,
                };
                candidates.push(c);
                offer(c, &mut results);
            }
        }
        while let Some(c) = candidates.pop() {
            let worst = results.peek().map_or(f32::MIN, |r| r.0.sim);
            if results.len() >= ef && c.sim < worst {
                break;
            }
            for nb in graph.links(c.id, 0) {
                if !first_visit(nb) {
                    continue;
                }
                let sim = store.cosine(nb as usize, query, query_norm);
                let worst = results.peek().map_or(f32::MIN, |r| r.0.sim);
                if results.len() < ef || sim > worst {
                    let n = Cand { sim, id: nb };
                    candidates.push(n);
                    offer(n, &mut results);
                }
            }
        }
        let mut out: Vec<Cand> = results.into_iter().map(|r| r.0).collect();
        out.sort_by(|a, b| b.cmp(a));
        out
    })
}

/// The neighbor-selection heuristic: walk `candidates` (most similar to the
/// base node first) and keep one only if it is more similar to the base
/// node than to every neighbor kept so far; then top up with the skipped
/// ones, so a dense cluster still gets `m` links.
fn select_neighbors(store: &FlatVectors, candidates: &[Cand], m: usize) -> Vec<u32> {
    let mut kept: Vec<u32> = Vec::with_capacity(m);
    let mut skipped: Vec<u32> = Vec::new();
    for c in candidates {
        if kept.len() >= m {
            break;
        }
        let diverse = kept
            .iter()
            .all(|&k| store.cosine_between(c.id as usize, k as usize) < c.sim);
        if diverse {
            kept.push(c.id);
        } else {
            skipped.push(c.id);
        }
    }
    for id in skipped {
        if kept.len() >= m {
            break;
        }
        kept.push(id);
    }
    kept
}

/// Neighbors chosen for node `q` on each of its layers, computed against a
/// read-only graph (plus the batch's earlier nodes, which the graph doesn't
/// hold yet).
fn plan_insert(
    store: &FlatVectors,
    graph: &Graph,
    q: usize,
    batch_start: usize,
    ef_construction: usize,
) -> Vec<Vec<u32>> {
    let level = graph.levels[q];
    let query = store.values(q);
    let query = &query[..];
    let query_norm = store.norm(q);
    let mut per_layer: Vec<Vec<Cand>> = vec![Vec::new(); level as usize + 1];

    if graph.entry != NONE {
        let mut eps = vec![graph.entry];
        let mut layer = graph.max_level;
        while layer > level {
            let best = search_layer(store, graph, query, query_norm, &eps, 1, layer);
            eps = vec![best[0].id];
            layer -= 1;
        }
        let top = level.min(graph.max_level);
        for layer in (0..=top).rev() {
            let found = search_layer(
                store,
                graph,
                query,
                query_norm,
                &eps,
                ef_construction,
                layer,
            );
            eps = found.iter().map(|c| c.id).collect();
            per_layer[layer as usize] = found;
        }
    }

    for other in batch_start..q {
        let other_level = graph.levels[other];
        let sim = store.cosine_between(q, other);
        for layer in 0..=level.min(other_level) {
            per_layer[layer as usize].push(Cand {
                sim,
                id: other as u32,
            });
        }
    }

    per_layer
        .into_iter()
        .map(|mut cands| {
            cands.sort_by(|a, b| b.cmp(a));
            cands.dedup_by_key(|c| c.id);
            select_neighbors(store, &cands, M)
        })
        .collect()
}

/// Applies a batch's planned links: each new node gets its own neighbor
/// lists, then every node that gained reverse links is updated -- the new
/// links appended in node order, and a list that overflows its capacity
/// pruned back with the same heuristic. The pruning is independent per
/// node, so it runs in parallel; the outcome doesn't depend on scheduling.
fn apply_batch(store: &FlatVectors, graph: &mut Graph, start: usize, plans: Vec<Vec<Vec<u32>>>) {
    use rayon::prelude::*;
    use std::collections::BTreeMap;

    let count = plans.len();
    // (layer, neighbor) -> new nodes linking to it, in insertion order.
    let mut reverse: BTreeMap<(u8, u32), Vec<u32>> = BTreeMap::new();
    for (offset, plan) in plans.into_iter().enumerate() {
        let q = (start + offset) as u32;
        for (layer, neighbors) in plan.into_iter().enumerate() {
            let layer = layer as u8;
            graph.set_links(q, layer, &neighbors);
            for nb in neighbors {
                reverse.entry((layer, nb)).or_default().push(q);
            }
        }
    }

    let updates: Vec<((u8, u32), Vec<u32>)> = reverse
        .into_par_iter()
        .map(|((layer, nb), added)| {
            let cap = if layer == 0 { M0 } else { M };
            let mut links: Vec<u32> = graph.links(nb, layer).collect();
            for q in added {
                if !links.contains(&q) {
                    links.push(q);
                }
            }
            if links.len() > cap {
                let mut cands: Vec<Cand> = links
                    .iter()
                    .map(|&id| Cand {
                        sim: store.cosine_between(nb as usize, id as usize),
                        id,
                    })
                    .collect();
                cands.sort_by(|a, b| b.cmp(a));
                links = select_neighbors(store, &cands, cap);
            }
            ((layer, nb), links)
        })
        .collect();
    for ((layer, nb), links) in updates {
        graph.set_links(nb, layer, &links);
    }

    for q in start..start + count {
        let level = graph.levels[q];
        if graph.entry == NONE || level > graph.max_level {
            graph.entry = q as u32;
            graph.max_level = level;
        }
    }
}

/// Builds a graph over every vector in `store`. Nodes go in, in order, in
/// batches: each batch's neighbor searches run in parallel against the
/// graph as it stood before the batch (plus a direct comparison with the
/// batch's earlier nodes), then `apply_batch` links them. Same input, same
/// graph, regardless of thread count.
fn build_graph(store: &FlatVectors, ef_construction: usize, max_batch: usize) -> Graph {
    use rayon::prelude::*;

    let n = store.len();
    let mut graph = Graph::with_levels((0..n).map(level_for).collect());
    let mut inserted = 0;
    while inserted < n {
        let end = (inserted + (inserted / 8).clamp(1, max_batch)).min(n);
        let plans: Vec<Vec<Vec<u32>>> = (inserted..end)
            .into_par_iter()
            .map(|q| plan_insert(store, &graph, q, inserted, ef_construction))
            .collect();
        apply_batch(store, &mut graph, inserted, plans);
        inserted = end;
    }
    graph
}

/// An approximate-nearest-neighbor index for vector similarity search,
/// opted into via `CREATE VECTOR INDEX ... USING HNSW` (the default, no
/// `USING` clause, stays `vector::VectorIndex`'s IVF clustering).
///
/// Only participates in top-k similarity search (`SEARCH ... LIMIT k`):
/// unlike IVF's clusters, a graph traversal has no cheap provable bound on
/// what it might have skipped, so it cannot safely accelerate an *exact*
/// predicate (`WHERE COSINE_SIM(...) > threshold`) -- `search_threshold`
/// always scans every vector instead of touching the graph.
#[derive(Debug, Clone, Default)]
pub struct HnswIndex {
    store: FlatVectors,
    /// Built by the last `build()`/`restore_from_snapshot()` over
    /// `store[..indexed_count]`; `None` if never built or too few vectors.
    graph: Option<Arc<Graph>>,
    /// `store[indexed_count..]` is the unindexed tail (rows added after the
    /// last build), always brute-force scanned alongside the graph.
    indexed_count: usize,
}

impl HnswIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Exports the graph `build()` last computed, for `SAVE DATASET` to
    /// persist -- `None` if `build()` never ran or the vector count was
    /// below `MIN_VECTORS_TO_INDEX`. Holds only the graph structure; the
    /// vectors come back by re-`add()`-ing the reloaded rows in order.
    pub fn snapshot(&self) -> Option<HnswIndexSnapshot> {
        Some(HnswIndexSnapshot {
            indexed_count: self.indexed_count,
            graph: self.graph.clone()?,
        })
    }

    /// Restores a previously exported graph without rebuilding it. Like
    /// `VectorIndex::restore_from_snapshot`, valid only when rows were
    /// re-`add()`-ed in the order the snapshot was taken (the caller checks
    /// the content hash); the graph's shape is still validated here.
    pub fn restore_from_snapshot(&mut self, snapshot: HnswIndexSnapshot) -> Result<(), String> {
        if snapshot.indexed_count > self.store.len() {
            return Err(format!(
                "HNSW index snapshot expects at least {} vectors, only {} were added",
                snapshot.indexed_count,
                self.store.len()
            ));
        }
        validate(&snapshot.graph, snapshot.indexed_count)?;
        self.graph = Some(snapshot.graph);
        self.indexed_count = snapshot.indexed_count;
        Ok(())
    }

    /// Content hash of a vector column's values, in row order. Shares
    /// `vector::VectorIndex`'s implementation.
    pub fn content_hash(values: &[Value]) -> String {
        super::vector::VectorIndex::content_hash(values)
    }

    fn search_graph(&self, graph: &Graph, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        self.search_graph_ef(graph, query, k, EF_SEARCH.max(k))
    }

    /// Top-`k` among the rows `allowed` accepts (by row id), of which there
    /// are `allowed_count`: a filtered graph walk plus a scan of the
    /// unindexed tail. Exact instead -- a scan of every allowed row -- when
    /// that set is small (at most `4 * max(k, ef_search)` rows), when there
    /// is no graph, or when the walk found fewer than `k` while at least `k`
    /// rows are allowed; so it returns `min(k, allowed_count)` rows, like an
    /// exact pre-filtered search.
    pub fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        allowed: &dyn Fn(usize) -> bool,
        allowed_count: usize,
    ) -> Result<Vec<(usize, f32)>, String> {
        self.store.check_query(query)?;
        let wanted = k.min(allowed_count);
        let exact = |s: &Self| {
            s.store.top_k(
                (0..s.store.len()).filter(|&i| allowed(s.store.row_id(i))),
                query,
                k,
            )
        };
        let ef = EF_SEARCH.max(k);
        let graph = match &self.graph {
            Some(g) if allowed_count > 4 * ef && g.entry != NONE => g,
            _ => return Ok(exact(self)),
        };
        let query_norm = l2_norm(query);
        let mut eps = vec![graph.entry];
        let mut layer = graph.max_level;
        while layer > 0 {
            let best = search_layer(&self.store, graph, query, query_norm, &eps, 1, layer);
            eps = vec![best[0].id];
            layer -= 1;
        }
        let accept = |pos: u32| allowed(self.store.row_id(pos as usize));
        let mut results: Vec<(usize, f32)> =
            search_layer_filtered(&self.store, graph, query, query_norm, &eps, ef, &accept)
                .into_iter()
                .map(|c| (self.store.row_id(c.id as usize), c.sim))
                .collect();
        results.extend(self.store.top_k(
            (self.indexed_count..self.store.len()).filter(|&i| allowed(self.store.row_id(i))),
            query,
            k,
        ));
        if results.len() < wanted {
            return Ok(exact(self));
        }
        results.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        results.truncate(k);
        Ok(results)
    }

    fn search_graph_ef(
        &self,
        graph: &Graph,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Vec<(usize, f32)> {
        if graph.entry == NONE || k == 0 {
            return Vec::new();
        }
        let query_norm = l2_norm(query);
        let mut eps = vec![graph.entry];
        let mut layer = graph.max_level;
        while layer > 0 {
            let best = search_layer(&self.store, graph, query, query_norm, &eps, 1, layer);
            eps = vec![best[0].id];
            layer -= 1;
        }
        search_layer(&self.store, graph, query, query_norm, &eps, ef, 0)
            .into_iter()
            .take(k)
            .map(|c| (self.store.row_id(c.id as usize), c.sim))
            .collect()
    }
}

fn validate(graph: &Graph, n: usize) -> Result<(), String> {
    let bad = || Err("HNSW index snapshot is inconsistent".to_string());
    if graph.levels.len() != n || graph.layer0.len() != n * M0 || graph.upper_offset.len() != n {
        return bad();
    }
    if n > 0 && (graph.entry as usize >= n || graph.levels[graph.entry as usize] != graph.max_level)
    {
        return bad();
    }
    let mut total = 0usize;
    for (i, &l) in graph.levels.iter().enumerate() {
        if l > MAX_LEVEL {
            return bad();
        }
        let expected = if l > 0 { total as u32 } else { NONE };
        if graph.upper_offset[i] != expected {
            return bad();
        }
        total += l as usize * M;
    }
    if graph.upper_links.len() != total {
        return bad();
    }
    let in_range = |&x: &u32| x == NONE || (x as usize) < n;
    if !graph.layer0.iter().all(in_range) || !graph.upper_links.iter().all(in_range) {
        return bad();
    }
    Ok(())
}

/// `HnswIndex::snapshot`'s output: the graph structure only.
#[derive(Debug, Clone)]
pub struct HnswIndexSnapshot {
    indexed_count: usize,
    graph: Arc<Graph>,
}

/// What `SAVE DATASET` writes per HNSW-indexed column.
#[derive(Debug, Clone)]
pub struct PersistedHnswIndex {
    pub content_hash: String,
    pub snapshot: HnswIndexSnapshot,
}

const HNSW_MAGIC: &[u8; 8] = b"LNLHNS1\0";

/// Binary layout of `hnsw_index_graphs.bin` (little-endian): magic
/// `LNLHNS1\0`, u32 column count, then per column (sorted by name): column
/// name and content hash (u32 length + UTF-8 each), u64 indexed count, u32
/// entry node, u32 max level, then four u64-length-prefixed arrays: node
/// levels (u32 each), layer-0 links (`2M` = 32 u32 slots per node), upper
/// offsets (u32 per node), upper links (u32). Empty slots are `u32::MAX`.
/// Vectors are not stored: they are the dataset's own column.
pub fn encode_snapshots(
    snapshots: &std::collections::HashMap<String, PersistedHnswIndex>,
) -> Vec<u8> {
    let mut w = super::binio::Writer::new(HNSW_MAGIC);
    let mut columns: Vec<&String> = snapshots.keys().collect();
    columns.sort();
    w.u32(columns.len() as u32);
    for column in columns {
        let p = &snapshots[column];
        let g = &p.snapshot.graph;
        w.str(column);
        w.str(&p.content_hash);
        w.u64(p.snapshot.indexed_count as u64);
        w.u32(g.entry);
        w.u32(g.max_level as u32);
        let levels: Vec<u32> = g.levels.iter().map(|&l| l as u32).collect();
        w.u32s(&levels);
        w.u32s(&g.layer0);
        w.u32s(&g.upper_offset);
        w.u32s(&g.upper_links);
    }
    w.into_bytes()
}

pub fn decode_snapshots(
    bytes: &[u8],
) -> Result<std::collections::HashMap<String, PersistedHnswIndex>, String> {
    let mut r = super::binio::Reader::new(bytes, HNSW_MAGIC)?;
    let mut out = std::collections::HashMap::new();
    for _ in 0..r.u32()? {
        let column = r.str()?;
        let content_hash = r.str()?;
        let indexed_count = r.u64()? as usize;
        let entry = r.u32()?;
        let max_level = r.u32()?;
        let levels = r.u32s()?;
        let graph = Graph {
            entry,
            max_level: u8::try_from(max_level).map_err(|e| e.to_string())?,
            levels: levels
                .into_iter()
                .map(|l| u8::try_from(l).map_err(|e| e.to_string()))
                .collect::<Result<_, _>>()?,
            layer0: r.u32s()?,
            upper_offset: r.u32s()?,
            upper_links: r.u32s()?,
        };
        validate(&graph, indexed_count)?;
        out.insert(
            column,
            PersistedHnswIndex {
                content_hash,
                snapshot: HnswIndexSnapshot {
                    indexed_count,
                    graph: Arc::new(graph),
                },
            },
        );
    }
    r.finish()?;
    Ok(out)
}

impl Index for HnswIndex {
    fn add(&mut self, row_id: usize, value: &Value) -> Result<(), String> {
        self.store.add_value(row_id, value)
    }

    fn lookup(&self, _value: &Value) -> Result<Vec<usize>, String> {
        Err("HnswIndex does not support exact value lookup".to_string())
    }

    fn memory_bytes(&self) -> usize {
        self.store.memory_bytes() + self.graph.as_ref().map_or(0, |g| g.memory_bytes())
    }

    fn search(&self, query: &Tensor, k: usize) -> Result<Vec<(usize, f32)>, String> {
        let query = super::binio::query_values(query);
        self.store.check_query(&query)?;
        let mut results = self
            .store
            .top_k(self.indexed_count..self.store.len(), &query, k);
        if let Some(graph) = &self.graph {
            results.extend(self.search_graph(graph, &query, k));
        } else if self.indexed_count > 0 {
            // Graph missing but indexed_count > 0 should never happen
            // (build()/restore_from_snapshot keep them in sync) -- fall back
            // to scanning everything rather than returning an incomplete
            // result.
            results = self.store.top_k(0..self.store.len(), &query, k);
        }
        results.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        results.truncate(k);
        Ok(results)
    }

    fn search_threshold(
        &self,
        query: &Tensor,
        threshold: f32,
        strict: bool,
    ) -> Result<Vec<(usize, f32)>, String> {
        // A graph traversal has no provable bound on what it skipped, so an
        // exact predicate always scans every vector directly.
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
        Ok((0..self.store.len())
            .filter_map(|i| {
                let sim = self.store.cosine(i, &query, query_norm);
                passes(sim).then(|| (self.store.row_id(i), sim))
            })
            .collect())
    }

    fn index_type(&self) -> IndexType {
        IndexType::Hnsw
    }

    fn box_clone(&self) -> Box<dyn Index> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn build(&mut self) -> Result<(), String> {
        let n = self.store.len();
        self.graph = None;
        self.indexed_count = 0;
        if n < MIN_VECTORS_TO_INDEX {
            return Ok(());
        }
        if n >= NONE as usize {
            return Err(format!("HNSW index supports at most {} vectors", NONE - 1));
        }
        self.graph = Some(Arc::new(build_graph(
            &self.store,
            EF_CONSTRUCTION,
            MAX_BATCH,
        )));
        self.indexed_count = n;
        Ok(())
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

    fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut x = seed;
        (0..n)
            .map(|_| {
                (0..dim)
                    .map(|_| {
                        x = x
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        ((x >> 33) as f32 / (1u64 << 31) as f32) - 0.5
                    })
                    .collect()
            })
            .collect()
    }

    fn tensor(v: &[f32]) -> Tensor {
        let id = crate::core::tensor::TensorId::new();
        let meta = crate::core::tensor::TensorMetadata::new(id, None);
        Tensor::new(
            id,
            crate::core::tensor::Shape::new(vec![v.len()]),
            v.to_vec(),
            meta,
        )
        .unwrap()
    }

    fn random_index(n: usize, dim: usize) -> (HnswIndex, Vec<Vec<f32>>) {
        let data = random_vectors(n, dim, 42);
        let mut index = HnswIndex::new();
        for (i, v) in data.iter().enumerate() {
            index.add(i, &Value::Vector(v.clone())).unwrap();
        }
        index.build().unwrap();
        (index, data)
    }

    #[test]
    fn recall_at_10_against_brute_force() {
        let (index, data) = random_index(3000, 32);
        let queries = random_vectors(50, 32, 7);
        let mut hits = 0;
        for q in &queries {
            let qn = l2_norm(q);
            let mut exact: Vec<(usize, f32)> = data
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    (
                        i,
                        super::super::flat::cosine_with_norms(q, qn, v, l2_norm(v)),
                    )
                })
                .collect();
            exact.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let truth: std::collections::HashSet<usize> =
                exact.iter().take(10).map(|x| x.0).collect();
            let got = index.search(&tensor(q), 10).unwrap();
            assert_eq!(got.len(), 10);
            hits += got.iter().filter(|(id, _)| truth.contains(id)).count();
        }
        let recall = hits as f64 / (queries.len() * 10) as f64;
        assert!(recall >= 0.95, "recall@10 = {}", recall);
    }

    #[test]
    fn scores_are_exact_cosine() {
        let (index, data) = random_index(500, 16);
        let q = random_vectors(1, 16, 3).remove(0);
        for (row_id, score) in index.search(&tensor(&q), 5).unwrap() {
            let v = &data[row_id];
            let exact = super::super::flat::cosine_with_norms(&q, l2_norm(&q), v, l2_norm(v));
            assert_eq!(score.to_bits(), exact.to_bits());
        }
    }

    #[test]
    fn build_is_deterministic() {
        let (a, _) = random_index(2500, 16);
        let (b, _) = random_index(2500, 16);
        let (ga, gb) = (a.graph.unwrap(), b.graph.unwrap());
        assert_eq!(ga.entry, gb.entry);
        assert_eq!(ga.levels, gb.levels);
        assert_eq!(ga.layer0, gb.layer0);
        assert_eq!(ga.upper_links, gb.upper_links);
    }

    #[test]
    fn binary_snapshot_round_trips_and_rejects_corruption() {
        let (index, _) = random_index(400, 8);
        let mut map = std::collections::HashMap::new();
        map.insert(
            "e".to_string(),
            PersistedHnswIndex {
                content_hash: "h".into(),
                snapshot: index.snapshot().unwrap(),
            },
        );
        let bytes = encode_snapshots(&map);
        let back = decode_snapshots(&bytes).unwrap();
        let (g0, g1) = (&map["e"].snapshot.graph, &back["e"].snapshot.graph);
        assert_eq!(g0.layer0, g1.layer0);
        assert_eq!(g0.upper_links, g1.upper_links);
        assert_eq!(back["e"].content_hash, "h");

        assert!(decode_snapshots(&bytes[..bytes.len() - 3]).is_err());
        let mut bad_magic = bytes.clone();
        bad_magic[0] = b'X';
        assert!(decode_snapshots(&bad_magic).is_err());
        // Point a layer-0 link past the end: rejected, not trusted.
        let mut bad_link = bytes.clone();
        let tail = bad_link.len();
        let g = &map["e"].snapshot.graph;
        let upper_bytes = 8 + 4 * g.upper_links.len() + 8 + 4 * g.upper_offset.len();
        let first_link = tail - upper_bytes - 4 * g.layer0.len();
        bad_link[first_link..first_link + 4].copy_from_slice(&999_999u32.to_le_bytes());
        assert!(decode_snapshots(&bad_link).is_err());
    }

    #[test]
    fn dimension_mismatch_is_an_error() {
        let mut index = HnswIndex::new();
        index.add(0, &Value::Vector(vec![1.0, 0.0])).unwrap();
        assert!(index.add(1, &Value::Vector(vec![1.0, 0.0, 0.0])).is_err());
        assert!(index.search(&tensor(&[1.0, 0.0, 0.0]), 1).is_err());
    }

    /// Tuning experiment, not part of the suite: `HNSW_EXP_N=20000 cargo
    /// test --release --lib hnsw_experiment -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn hnsw_experiment() {
        let n: usize = std::env::var("HNSW_EXP_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(20000);
        let d = 128;
        let data = random_vectors(n, d, 42);
        let queries = random_vectors(100, d, 7);
        let truth: Vec<Vec<usize>> = queries
            .iter()
            .map(|q| {
                let qn = l2_norm(q);
                let mut ex: Vec<(usize, f32)> = data
                    .iter()
                    .enumerate()
                    .map(|(i, v)| {
                        (
                            i,
                            super::super::flat::cosine_with_norms(q, qn, v, l2_norm(v)),
                        )
                    })
                    .collect();
                ex.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                ex.iter().take(10).map(|x| x.0).collect()
            })
            .collect();
        let mut store = FlatVectors::default();
        for (i, v) in data.iter().enumerate() {
            store.push(i, v).unwrap();
        }
        for (efc, batch) in [(128, 1), (128, MAX_BATCH), (256, MAX_BATCH)] {
            let t = std::time::Instant::now();
            let graph = build_graph(&store, efc, batch);
            let secs = t.elapsed().as_secs_f64();
            let index = HnswIndex {
                store: store.clone(),
                graph: Some(Arc::new(graph)),
                indexed_count: n,
            };
            for ef in [64, 128, 256, 512] {
                let mut hits = 0;
                for (q, t) in queries.iter().zip(&truth) {
                    let got = index.search_graph_ef(index.graph.as_ref().unwrap(), q, 10, ef);
                    hits += got.iter().filter(|(id, _)| t.contains(id)).count();
                }
                println!(
                    "efc={} batch={} build={:.1}s ef={} recall={:.3}",
                    efc,
                    batch,
                    secs,
                    ef,
                    hits as f64 / 1000.0
                );
            }
        }
    }
}
