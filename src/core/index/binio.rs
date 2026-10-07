//! Minimal little-endian binary encoding for persisted index snapshots
//! (`vector_index_clusters.bin`, `hnsw_index_graphs.bin`). Hand-rolled rather
//! than a serde format: the layouts are a few flat arrays, and a fixed,
//! documented byte layout is what a future memory-mapped load needs.
//!
//! Every file starts with an 8-byte magic that names its kind and version;
//! a reader rejects anything else instead of guessing.

use std::borrow::Cow;

pub(crate) struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    pub fn new(magic: &[u8; 8]) -> Self {
        Self {
            buf: magic.to_vec(),
        }
    }
    pub fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    pub fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    pub fn f32(&mut self, v: f32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    pub fn str(&mut self, s: &str) {
        self.u32(s.len() as u32);
        self.buf.extend_from_slice(s.as_bytes());
    }
    pub fn f32s(&mut self, v: &[f32]) {
        self.u64(v.len() as u64);
        for x in v {
            self.f32(*x);
        }
    }
    pub fn u32s(&mut self, v: &[u32]) {
        self.u64(v.len() as u64);
        for x in v {
            self.u32(*x);
        }
    }
    /// Like `u32s`, but the values start at a 4-byte-aligned file offset
    /// (zero padding after the length), so a memory-mapped file can be read
    /// in place as `&[u32]`.
    pub fn u32s_aligned(&mut self, v: &[u32]) {
        self.u64(v.len() as u64);
        while !self.buf.len().is_multiple_of(4) {
            self.buf.push(0);
        }
        for x in v {
            self.u32(*x);
        }
    }
    pub fn into_bytes(self) -> Vec<u8> {
        self.buf
    }
}

pub(crate) struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    /// Checks the magic and positions the reader just past it.
    pub fn new(buf: &'a [u8], magic: &[u8; 8]) -> Result<Self, String> {
        if buf.len() < 8 || &buf[..8] != magic {
            return Err(format!(
                "not a {} file (bad header)",
                String::from_utf8_lossy(magic).trim_end_matches('\0')
            ));
        }
        Ok(Self { buf, pos: 8 })
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.buf.len())
            .ok_or_else(|| "truncated index snapshot".to_string())?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    pub fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    pub fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub fn str(&mut self) -> Result<String, String> {
        let n = self.u32()? as usize;
        String::from_utf8(self.take(n)?.to_vec()).map_err(|e| e.to_string())
    }
    fn len(&mut self, elem: usize) -> Result<usize, String> {
        let n = self.u64()? as usize;
        // Guard against a corrupt length asking for more than the file has.
        if n.saturating_mul(elem) > self.buf.len() - self.pos {
            return Err("truncated index snapshot".to_string());
        }
        Ok(n)
    }
    pub fn f32s(&mut self) -> Result<Vec<f32>, String> {
        let n = self.len(4)?;
        (0..n).map(|_| self.f32()).collect()
    }
    pub fn u32s(&mut self) -> Result<Vec<u32>, String> {
        let n = self.len(4)?;
        (0..n).map(|_| self.u32()).collect()
    }
    /// The `(byte offset, count)` of a `Writer::u32s_aligned` array, skipping
    /// over it.
    pub fn u32s_aligned_range(&mut self) -> Result<(usize, usize), String> {
        let n = self.u64()? as usize;
        while !self.pos.is_multiple_of(4) {
            self.take(1)?;
        }
        let start = self.pos;
        self.take(n.checked_mul(4).ok_or("truncated index snapshot")?)?;
        Ok((start, n))
    }
    /// A `Writer::u32s_aligned` array, copied out.
    pub fn u32s_aligned(&mut self) -> Result<Vec<u32>, String> {
        let (start, n) = self.u32s_aligned_range()?;
        Ok(self.buf[start..start + 4 * n]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect())
    }
    pub fn finish(&self) -> Result<(), String> {
        if self.pos != self.buf.len() {
            return Err("trailing bytes after index snapshot".to_string());
        }
        Ok(())
    }
}

/// A query tensor's logical values, borrowed when it's contiguous.
pub(crate) fn query_values(query: &crate::core::tensor::Tensor) -> Cow<'_, [f32]> {
    match query.as_contiguous_slice() {
        Some(s) => Cow::Borrowed(s),
        None => Cow::Owned(query.to_logical_vec()),
    }
}
