//! Memory-mapped column files (CASMI_WORKLOADS_PLAN_2.md, P14).
//!
//! A `BitVector(N)`, `Vector(d, F16)` or `Vector(d, I8)` column can be saved
//! as one contiguous file per column (`SAVE DATASET <name> MMAP`, or every
//! `SAVE` with `[storage] mmap_columns = true`) and memory-mapped back on
//! `LOAD`: each cell's payload then stays in the mapped file instead of a heap
//! allocation per row, so a large fingerprint or quantized-vector library
//! costs page cache (which the OS can evict) rather than resident heap.
//!
//! The values stay ordinary `Value::BitVector` / `Value::QVector`: inside, the
//! words (`BitVec`) or elements (`QuantVec`) are a `Buf`, owned or a view into
//! the map. Every reader goes through a slice, so no executor changes; a
//! write copies the cell first (`Buf::to_mut`). A column file is never edited
//! in place -- `SAVE` writes a temporary file and renames it -- so a live map
//! never sees its bytes change.
//!
//! File layout (`<dataset package>/columns/<column>.lcol`), little-endian:
//!
//! | bytes | content |
//! |---|---|
//! | 0..8 | magic `LNLCOL01` |
//! | 8..12 | kind: 1 = BitVector, 2 = F16, 3 = I8 |
//! | 12..16 | reserved (0) |
//! | 16..24 | rows |
//! | 24..32 | dim (bits, or elements) |
//! | 32..40 | stride: bytes per row of payload, a multiple of 8 |
//! | 40..72 | SHA-256 of every byte after the header |
//! | 72..80 | reserved (0) |
//! | 80.. | validity: one byte per row (1 = value, 0 = NULL), padded to 8 |
//! | .. | I8 only: one f32 scale per row, padded to 8 |
//! | .. | payload: `rows * stride` bytes |
//!
//! Every section starts at a multiple of 8 from the page-aligned map, so a
//! row's words (`u64`) or elements (`u16`, `i8`) can be borrowed in place.

use std::sync::Arc;

/// Element types a column file stores.
pub trait Pod: Copy + Default + std::fmt::Debug + PartialEq + Send + Sync + 'static {}
impl Pod for u64 {}
impl Pod for u16 {}
impl Pod for i8 {}

/// A slice of `T` that is either owned or a view into a mapped column file.
#[derive(Clone)]
pub enum Buf<T: Pod> {
    Owned(Vec<T>),
    Mapped {
        map: Arc<memmap2::Mmap>,
        /// Byte offset into the map, a multiple of `align_of::<T>()`.
        offset: usize,
        len: usize,
    },
}

impl<T: Pod> std::ops::Deref for Buf<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        match self {
            Buf::Owned(v) => v,
            // SAFETY: `ColumnFile::open` only builds this for a
            // little-endian host, an `offset` aligned for `T` into a
            // page-aligned map, and `offset + len * size_of::<T>()` within
            // the map (all checked against the header and the file size);
            // the map stays alive (`Arc`) while the slice can be borrowed,
            // and LINAL never modifies a column file in place (`SAVE`
            // replaces it by rename), so the bytes don't change under us.
            // Every bit pattern is a valid `u64`/`u16`/`i8`.
            Buf::Mapped { map, offset, len } => unsafe {
                std::slice::from_raw_parts(map.as_ptr().add(*offset) as *const T, *len)
            },
        }
    }
}

impl<T: Pod> Buf<T> {
    /// The owned vector, copying a mapped slice first.
    pub fn to_mut(&mut self) -> &mut Vec<T> {
        if let Buf::Mapped { .. } = self {
            *self = Buf::Owned(self.to_vec());
        }
        match self {
            Buf::Owned(v) => v,
            Buf::Mapped { .. } => unreachable!(),
        }
    }

    /// Heap bytes held (0 for a mapped slice).
    pub fn heap_bytes(&self) -> usize {
        match self {
            Buf::Owned(v) => v.capacity() * std::mem::size_of::<T>(),
            Buf::Mapped { .. } => 0,
        }
    }

    /// Bytes of the mapped file this slice covers (0 when owned).
    pub fn mapped_bytes(&self) -> usize {
        match self {
            Buf::Owned(_) => 0,
            Buf::Mapped { len, .. } => len * std::mem::size_of::<T>(),
        }
    }
}

impl<T: Pod> From<Vec<T>> for Buf<T> {
    fn from(v: Vec<T>) -> Self {
        Buf::Owned(v)
    }
}

impl<T: Pod> FromIterator<T> for Buf<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Buf::Owned(iter.into_iter().collect())
    }
}

impl<T: Pod> std::fmt::Debug for Buf<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T: Pod> PartialEq for Buf<T> {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl<T: Pod + Eq> Eq for Buf<T> {}

impl<T: Pod + std::hash::Hash> std::hash::Hash for Buf<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (**self).hash(state)
    }
}

const MAGIC: &[u8; 8] = b"LNLCOL01";
const HEADER: usize = 80;

/// What a column file holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnKind {
    BitVector,
    F16,
    I8,
}

impl ColumnKind {
    fn code(self) -> u32 {
        match self {
            ColumnKind::BitVector => 1,
            ColumnKind::F16 => 2,
            ColumnKind::I8 => 3,
        }
    }
    fn from_code(c: u32) -> Option<Self> {
        Some(match c {
            1 => ColumnKind::BitVector,
            2 => ColumnKind::F16,
            3 => ColumnKind::I8,
            _ => return None,
        })
    }
    /// Payload bytes per row for `dim` bits / elements, padded to 8.
    fn stride(self, dim: usize) -> usize {
        let raw = match self {
            ColumnKind::BitVector => dim.div_ceil(64) * 8,
            ColumnKind::F16 => dim * 2,
            ColumnKind::I8 => dim,
        };
        raw.div_ceil(8) * 8
    }
    /// The column kind for a value type, if it can be mapped.
    pub fn of(t: &crate::core::value::ValueType) -> Option<(Self, usize)> {
        use crate::core::quant::Quantization;
        use crate::core::value::ValueType;
        match t {
            ValueType::BitVector(n) => Some((ColumnKind::BitVector, *n)),
            ValueType::QVector(d, Quantization::F16) => Some((ColumnKind::F16, *d)),
            ValueType::QVector(d, Quantization::I8) => Some((ColumnKind::I8, *d)),
            _ => None,
        }
    }
}

fn pad8(n: usize) -> usize {
    n.div_ceil(8) * 8
}

/// Writes `values` (one column, all `kind` of length `dim`, or NULL) as a
/// column file at `path`, atomically (temporary file + rename).
pub fn write_column(
    path: &std::path::Path,
    kind: ColumnKind,
    dim: usize,
    values: &[&crate::core::value::Value],
) -> Result<(), String> {
    use crate::core::quant::QuantVec;
    use crate::core::value::Value;
    use sha2::{Digest, Sha256};

    let rows = values.len();
    let stride = kind.stride(dim);
    let mut body = Vec::with_capacity(pad8(rows) + rows * (stride + 4));
    body.extend(values.iter().map(|v| u8::from(!v.is_null())));
    body.resize(pad8(rows), 0);
    if kind == ColumnKind::I8 {
        for v in values {
            let scale = match v {
                Value::QVector(QuantVec::I8 { scale, .. }) => *scale,
                _ => 0.0,
            };
            body.extend_from_slice(&scale.to_le_bytes());
        }
        body.resize(pad8(body.len()), 0);
    }
    let bad = |r: usize, v: &Value| {
        format!(
            "row {} is {:?}, not the column's {:?} of {}",
            r,
            v.value_type(),
            kind,
            dim
        )
    };
    for (r, v) in values.iter().enumerate() {
        let start = body.len();
        match (kind, v) {
            (_, Value::Null) => {}
            (ColumnKind::BitVector, Value::BitVector(b)) if b.len() == dim => {
                for w in b.words() {
                    body.extend_from_slice(&w.to_le_bytes());
                }
            }
            (ColumnKind::F16, Value::QVector(QuantVec::F16(d))) if d.len() == dim => {
                for h in d.iter() {
                    body.extend_from_slice(&h.to_le_bytes());
                }
            }
            (ColumnKind::I8, Value::QVector(QuantVec::I8 { data, .. })) if data.len() == dim => {
                body.extend(data.iter().map(|q| *q as u8));
            }
            _ => return Err(bad(r, v)),
        }
        body.resize(start + stride, 0);
    }

    let mut header = Vec::with_capacity(HEADER);
    header.extend_from_slice(MAGIC);
    header.extend_from_slice(&kind.code().to_le_bytes());
    header.extend_from_slice(&0u32.to_le_bytes());
    header.extend_from_slice(&(rows as u64).to_le_bytes());
    header.extend_from_slice(&(dim as u64).to_le_bytes());
    header.extend_from_slice(&(stride as u64).to_le_bytes());
    header.extend_from_slice(&Sha256::digest(&body));
    header.extend_from_slice(&0u64.to_le_bytes());
    debug_assert_eq!(header.len(), HEADER);

    let dir = path.parent().ok_or("column file has no parent directory")?;
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let tmp = path.with_extension("lcol.tmp");
    {
        use std::io::Write;
        let mut f = std::fs::File::create(&tmp).map_err(|e| e.to_string())?;
        f.write_all(&header).map_err(|e| e.to_string())?;
        f.write_all(&body).map_err(|e| e.to_string())?;
        f.sync_all().map_err(|e| e.to_string())?;
    }
    std::fs::rename(&tmp, path).map_err(|e| e.to_string())
}

/// A mapped column file, checked against its header and hash.
pub struct ColumnFile {
    map: Arc<memmap2::Mmap>,
    pub kind: ColumnKind,
    pub rows: usize,
    pub dim: usize,
    stride: usize,
    validity: usize,
    scales: usize,
    payload: usize,
}

impl ColumnFile {
    /// Maps `path` and checks its magic, kind, sizes and SHA-256 (which
    /// reads the whole file once). Any mismatch is an error naming the file.
    pub fn open(path: &std::path::Path) -> Result<Self, String> {
        use sha2::{Digest, Sha256};
        let name = path.display();
        if cfg!(target_endian = "big") {
            return Err(format!(
                "{}: memory-mapped columns need a little-endian host",
                name
            ));
        }
        let file = std::fs::File::open(path).map_err(|e| format!("{}: {}", name, e))?;
        // SAFETY: the file is only ever replaced by rename, never written
        // in place, so the mapped bytes don't change while mapped.
        let map = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| format!("{}: {}", name, e))?;
        let corrupt = |why: &str| {
            format!(
                "{}: not a valid column file ({}) -- SAVE the dataset again",
                name, why
            )
        };
        if map.len() < HEADER || &map[..8] != MAGIC {
            return Err(corrupt("bad header"));
        }
        let u32_at = |o: usize| u32::from_le_bytes(map[o..o + 4].try_into().unwrap());
        let u64_at = |o: usize| u64::from_le_bytes(map[o..o + 8].try_into().unwrap()) as usize;
        let kind = ColumnKind::from_code(u32_at(8)).ok_or_else(|| corrupt("unknown kind"))?;
        let (rows, dim, stride) = (u64_at(16), u64_at(24), u64_at(32));
        if stride != kind.stride(dim) {
            return Err(corrupt("stride does not match the dimension"));
        }
        let validity = HEADER;
        let scales = validity + pad8(rows);
        let payload = if kind == ColumnKind::I8 {
            scales + pad8(rows * 4)
        } else {
            scales
        };
        let expected = rows
            .checked_mul(stride)
            .and_then(|p| p.checked_add(payload))
            .ok_or_else(|| corrupt("sizes overflow"))?;
        if map.len() != expected {
            return Err(corrupt(&format!(
                "{} bytes, the header describes {} -- truncated or extended",
                map.len(),
                expected
            )));
        }
        if Sha256::digest(&map[HEADER..]).as_slice() != &map[40..72] {
            return Err(corrupt("content hash mismatch"));
        }
        Ok(ColumnFile {
            map: Arc::new(map),
            kind,
            rows,
            dim,
            stride,
            validity,
            scales,
            payload,
        })
    }

    /// Bytes mapped.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Row `r` as a `Value` borrowing the map (or `Null`).
    pub fn value(&self, r: usize) -> crate::core::value::Value {
        use crate::core::quant::QuantVec;
        use crate::core::value::Value;
        if self.map[self.validity + r] == 0 {
            return Value::Null;
        }
        let offset = self.payload + r * self.stride;
        match self.kind {
            ColumnKind::BitVector => Value::BitVector(crate::core::bitvec::BitVec::from_words(
                self.dim,
                Buf::Mapped {
                    map: self.map.clone(),
                    offset,
                    len: self.dim.div_ceil(64),
                },
            )),
            ColumnKind::F16 => Value::QVector(QuantVec::F16(Buf::Mapped {
                map: self.map.clone(),
                offset,
                len: self.dim,
            })),
            ColumnKind::I8 => {
                let o = self.scales + r * 4;
                let scale = f32::from_le_bytes(self.map[o..o + 4].try_into().unwrap());
                Value::QVector(QuantVec::I8 {
                    scale,
                    data: Buf::Mapped {
                        map: self.map.clone(),
                        offset,
                        len: self.dim,
                    },
                })
            }
        }
    }
}
