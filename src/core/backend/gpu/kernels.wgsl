// Compute kernels for the `gpu-wgpu` backend (see mod.rs).
//
// Both entry points share one binding layout:
//   0: first input  (A for matmul, the row matrix X for cosine)
//   1: second input (B for matmul, the query vector q for cosine)
//   2: output
//   3: dims uniform

struct Dims {
    // matmul: C[m, n] = A[m, k] * B[k, n]
    // cosine: rows = m, dim = k (n unused)
    m: u32,
    k: u32,
    n: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> in_a: array<f32>;
@group(0) @binding(1) var<storage, read> in_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

const TILE: u32 = 16u;
var<workgroup> tile_a: array<array<f32, 16>, 16>;
var<workgroup> tile_b: array<array<f32, 16>, 16>;

// Tiled GEMM: each 16x16 workgroup computes a 16x16 block of C, staging
// matching 16-wide slabs of A and B through workgroup memory.
@compute @workgroup_size(16, 16)
fn matmul(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = gid.y;
    let col = gid.x;
    var acc = 0.0;
    let tiles = (dims.k + TILE - 1u) / TILE;
    for (var t = 0u; t < tiles; t = t + 1u) {
        let a_col = t * TILE + lid.x;
        if (row < dims.m && a_col < dims.k) {
            tile_a[lid.y][lid.x] = in_a[row * dims.k + a_col];
        } else {
            tile_a[lid.y][lid.x] = 0.0;
        }
        let b_row = t * TILE + lid.y;
        if (b_row < dims.k && col < dims.n) {
            tile_b[lid.y][lid.x] = in_b[b_row * dims.n + col];
        } else {
            tile_b[lid.y][lid.x] = 0.0;
        }
        workgroupBarrier();
        for (var i = 0u; i < TILE; i = i + 1u) {
            acc = acc + tile_a[lid.y][i] * tile_b[i][lid.x];
        }
        workgroupBarrier();
    }
    if (row < dims.m && col < dims.n) {
        out[row * dims.n + col] = acc;
    }
}

// Cosine similarity of every row of X (m rows of dim k) against q, one
// row per invocation. A zero-norm row or query yields NaN -- the CPU kernel
// (`cosine_similarity_1d`) rejects that case with an error, and NaN lets the
// caller detect it per row instead of getting a plausible-looking 0.
@compute @workgroup_size(256)
fn cosine(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= dims.m) {
        return;
    }
    let base = row * dims.k;
    var dot = 0.0;
    var norm_x = 0.0;
    var norm_q = 0.0;
    for (var i = 0u; i < dims.k; i = i + 1u) {
        let x = in_a[base + i];
        let q = in_b[i];
        dot = dot + x * q;
        norm_x = norm_x + x * x;
        norm_q = norm_q + q * q;
    }
    let denom = sqrt(norm_x) * sqrt(norm_q);
    if (denom == 0.0) {
        out[row] = bitcast<f32>(0x7fc00000u);
    } else {
        out[row] = dot / denom;
    }
}
