//! Fixed-length bit vectors (`BitVector(N)` columns), e.g. molecular
//! fingerprints compared with Tanimoto similarity. Stored as 64-bit words,
//! 32x smaller than the same bits as `Vector(N)` floats.
//!
//! Bit `i` is word `i / 64`, bit `i % 64` internally. The external byte
//! layout (`to_bytes`/`from_bytes`, the Arrow `FixedSizeBinary` encoding)
//! is MSB-first per byte: bit `i` is bit `7 - i % 8` of byte `i / 8`, the
//! layout `numpy.packbits` produces by default, and bit `i` is character
//! `i` of a bit string (`"0101..."`, RDKit's `ToBitString`).

use serde::{Deserialize, Serialize};

/// Serializes as its bit string (`"0101..."`), e.g. `{"BitVector": "0101"}`
/// in the HTTP API's JSON: readable, and no 64-bit integers for JSON
/// clients to lose precision on.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(into = "String", try_from = "String")]
pub struct BitVec {
    len: usize,
    /// `ceil(len / 64)` words; bits past `len` in the last word are always 0.
    /// Owned, or a view into a memory-mapped column file (`core::colbuf`).
    words: crate::core::colbuf::Buf<u64>,
}

impl BitVec {
    pub fn zeros(len: usize) -> Self {
        Self {
            len,
            words: vec![0; len.div_ceil(64)].into(),
        }
    }

    /// From its words (bit `i` = bit `i % 64` of word `i / 64`), e.g. a
    /// mapped column file's slice. `words` must hold `ceil(len / 64)` words
    /// with the bits past `len` cleared.
    pub fn from_words(len: usize, words: crate::core::colbuf::Buf<u64>) -> Self {
        debug_assert_eq!(words.len(), len.div_ceil(64));
        Self { len, words }
    }

    /// The words, bit `i` = bit `i % 64` of word `i / 64`.
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    /// Heap bytes held (0 when the words are mapped).
    pub fn heap_bytes(&self) -> usize {
        self.words.heap_bytes()
    }

    /// Bytes of a mapped column file these words cover (0 when owned).
    pub fn mapped_bytes(&self) -> usize {
        self.words.mapped_bytes()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn get(&self, i: usize) -> bool {
        i < self.len && (self.words[i / 64] >> (i % 64)) & 1 == 1
    }

    pub fn set(&mut self, i: usize, on: bool) {
        assert!(
            i < self.len,
            "bit {} out of range for BitVector({})",
            i,
            self.len
        );
        let mask = 1u64 << (i % 64);
        let words = self.words.to_mut();
        if on {
            words[i / 64] |= mask;
        } else {
            words[i / 64] &= !mask;
        }
    }

    /// From a string of `'0'`/`'1'` characters (bit `i` = character `i`).
    pub fn from_bit_string(s: &str) -> Result<Self, String> {
        let mut bv = Self::zeros(s.len());
        for (i, c) in s.bytes().enumerate() {
            match c {
                b'0' => {}
                b'1' => bv.set(i, true),
                other => {
                    return Err(format!(
                        "a bit string may only contain '0' and '1', found {:?} at position {}",
                        other as char, i
                    ))
                }
            }
        }
        Ok(bv)
    }

    pub fn to_bit_string(&self) -> String {
        (0..self.len)
            .map(|i| if self.get(i) { '1' } else { '0' })
            .collect()
    }

    /// From the MSB-first packed bytes of `len` bits (`numpy.packbits`'s
    /// default). Padding bits past `len` must be 0.
    pub fn from_bytes(bytes: &[u8], len: usize) -> Result<Self, String> {
        if bytes.len() != len.div_ceil(8) {
            return Err(format!(
                "{} bits need {} bytes, got {}",
                len,
                len.div_ceil(8),
                bytes.len()
            ));
        }
        let mut bv = Self::zeros(len);
        for (i, byte) in bytes.iter().enumerate() {
            for b in 0..8 {
                if (byte >> (7 - b)) & 1 == 1 {
                    let bit = i * 8 + b;
                    if bit >= len {
                        return Err(format!("padding bit {} past length {} is set", bit, len));
                    }
                    bv.set(bit, true);
                }
            }
        }
        Ok(bv)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = vec![0u8; self.len.div_ceil(8)];
        for i in 0..self.len {
            if self.get(i) {
                out[i / 8] |= 1 << (7 - i % 8);
            }
        }
        out
    }

    /// Nonzero entries become 1 bits.
    pub fn from_floats(v: &[f32]) -> Self {
        let mut bv = Self::zeros(v.len());
        for (i, x) in v.iter().enumerate() {
            if *x != 0.0 {
                bv.set(i, true);
            }
        }
        bv
    }

    pub fn to_floats(&self) -> Vec<f32> {
        (0..self.len)
            .map(|i| if self.get(i) { 1.0 } else { 0.0 })
            .collect()
    }

    pub fn count_ones(&self) -> u64 {
        self.words.iter().map(|w| w.count_ones() as u64).sum()
    }

    fn check_len(&self, other: &Self, what: &str) -> Result<(), String> {
        if self.len != other.len {
            return Err(format!(
                "{}: BitVector lengths differ ({} vs {})",
                what, self.len, other.len
            ));
        }
        Ok(())
    }

    /// `|a ∧ b|` and `|a ∨ b|`.
    fn and_or(&self, other: &Self) -> (u64, u64) {
        self.words
            .iter()
            .zip(other.words.iter())
            .fold((0, 0), |(and, or), (a, b)| {
                (
                    and + (a & b).count_ones() as u64,
                    or + (a | b).count_ones() as u64,
                )
            })
    }

    /// Tanimoto (Jaccard) similarity `|a ∧ b| / |a ∨ b|`. Two all-zero
    /// vectors give 1.0, as RDKit's `TanimotoSimilarity` does.
    pub fn tanimoto(&self, other: &Self) -> Result<f64, String> {
        self.check_len(other, "TANIMOTO")?;
        let (and, or) = self.and_or(other);
        Ok(if or == 0 { 1.0 } else { and as f64 / or as f64 })
    }

    /// `Σ v[i]` over the set bits `i`: the dot product of the bits (as 0/1)
    /// with a dense vector of the same length, summed in f64.
    pub fn dot_dense(&self, v: &[f32]) -> Result<f64, String> {
        if v.len() != self.len {
            return Err(format!(
                "DOT: BitVector has {} bits, the vector {} elements",
                self.len,
                v.len()
            ));
        }
        let mut sum = 0.0f64;
        for (w, &word) in self.words.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                sum += v[w * 64 + bits.trailing_zeros() as usize] as f64;
                bits &= bits - 1;
            }
        }
        Ok(sum)
    }

    /// Number of differing bits.
    pub fn hamming(&self, other: &Self) -> Result<u64, String> {
        self.check_len(other, "HAMMING")?;
        Ok(self
            .words
            .iter()
            .zip(other.words.iter())
            .map(|(a, b)| (a ^ b).count_ones() as u64)
            .sum())
    }
}

impl From<BitVec> for String {
    fn from(b: BitVec) -> String {
        b.to_bit_string()
    }
}

impl TryFrom<String> for BitVec {
    type Error = String;
    fn try_from(s: String) -> Result<Self, String> {
        BitVec::from_bit_string(&s)
    }
}

impl std::fmt::Display for BitVec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_bit_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_round_trip_and_match_packbits() {
        let bv = BitVec::from_bit_string("1000000001").unwrap();
        // numpy.packbits([1,0,0,0,0,0,0,0,0,1]) == [128, 64]
        assert_eq!(bv.to_bytes(), vec![128, 64]);
        assert_eq!(BitVec::from_bytes(&[128, 64], 10).unwrap(), bv);
        assert_eq!(bv.to_bit_string(), "1000000001");
        assert!(BitVec::from_bytes(&[128, 65], 10).is_err()); // padding bit set
        assert!(BitVec::from_bit_string("10x").is_err());
    }

    #[test]
    fn similarity_across_word_boundaries() {
        // 130 bits: crosses two word boundaries, last word partial.
        let mut a = BitVec::zeros(130);
        let mut b = BitVec::zeros(130);
        for i in [0, 63, 64, 127, 128, 129] {
            a.set(i, true);
        }
        for i in [63, 64, 129, 5] {
            b.set(i, true);
        }
        // a∧b = {63, 64, 129} = 3; a∨b = {0,5,63,64,127,128,129} = 7
        assert_eq!(a.tanimoto(&b).unwrap(), 3.0 / 7.0);
        assert_eq!(a.hamming(&b).unwrap(), 4);
        assert_eq!(a.count_ones(), 6);
        assert_eq!(BitVec::zeros(10).tanimoto(&BitVec::zeros(10)).unwrap(), 1.0);
        assert!(a.tanimoto(&BitVec::zeros(129)).is_err());
    }

    #[test]
    fn serializes_as_a_bit_string() {
        let v = crate::core::value::Value::BitVector(BitVec::from_bit_string("0110").unwrap());
        let json = serde_json::to_string(&v).unwrap();
        assert_eq!(json, r#"{"BitVector":"0110"}"#);
        let back: crate::core::value::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(back, v);
    }
}
