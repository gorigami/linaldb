//! Quantized vectors (`Vector(d, F16)` / `Vector(d, I8)` columns): opt-in,
//! lower-precision storage for large vector columns.
//!
//! - `F16`: IEEE half precision, 2 bytes per element. Values beyond ±65504
//!   are an error, not infinity.
//! - `I8`: symmetric per-vector scaling, 1 byte per element plus one `f32`
//!   scale: `scale = max|x| / 127`, `q = round(x / scale)` in `[-127, 127]`,
//!   value `= q * scale`.
//!
//! Quantization happens once, when a value enters the column (INSERT,
//! UPDATE, load); from then on the stored value *is* the value. Every
//! expression reads it through `dequantize()`, and the vector indexes use
//! the same formula element by element, so an index-accelerated score and
//! `COSINE_SIM` on the column agree bit for bit. Columns without an
//! encoding are untouched.

use half::f16;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Quantization {
    F16,
    I8,
}

impl std::fmt::Display for Quantization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Quantization::F16 => write!(f, "F16"),
            Quantization::I8 => write!(f, "I8"),
        }
    }
}

impl std::str::FromStr for Quantization {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.to_uppercase().as_str() {
            "F16" | "FLOAT16" => Ok(Quantization::F16),
            "I8" | "INT8" => Ok(Quantization::I8),
            other => Err(format!(
                "unknown vector encoding '{}' (use F16 or I8)",
                other
            )),
        }
    }
}

/// A quantized vector. Serializes as its encoding, scale and dequantized
/// values (`{"encoding": "I8", "scale": 0.01, "values": [...]}`), which
/// deserialize back to the same quantized vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(into = "QuantRepr", try_from = "QuantRepr")]
pub enum QuantVec {
    /// Owned, or a view into a memory-mapped column file (`core::colbuf`).
    F16(crate::core::colbuf::Buf<u16>),
    I8 {
        scale: f32,
        data: crate::core::colbuf::Buf<i8>,
    },
}

#[derive(Serialize, Deserialize)]
struct QuantRepr {
    encoding: Quantization,
    scale: f32,
    values: Vec<f32>,
}

impl From<QuantVec> for QuantRepr {
    fn from(q: QuantVec) -> Self {
        QuantRepr {
            encoding: q.encoding(),
            scale: match &q {
                QuantVec::I8 { scale, .. } => *scale,
                QuantVec::F16(_) => 1.0,
            },
            values: q.dequantize(),
        }
    }
}

impl TryFrom<QuantRepr> for QuantVec {
    type Error = String;
    fn try_from(r: QuantRepr) -> Result<Self, String> {
        match r.encoding {
            Quantization::F16 => QuantVec::quantize(&r.values, Quantization::F16),
            Quantization::I8 => Ok(QuantVec::I8 {
                scale: r.scale,
                data: r
                    .values
                    .iter()
                    .map(|v| {
                        if r.scale == 0.0 {
                            0
                        } else {
                            (v / r.scale).round().clamp(-127.0, 127.0) as i8
                        }
                    })
                    .collect(),
            }),
        }
    }
}

#[inline]
pub(crate) fn f16_to_f32(bits: u16) -> f32 {
    f16::from_bits(bits).to_f32()
}

#[inline]
pub(crate) fn i8_to_f32(q: i8, scale: f32) -> f32 {
    q as f32 * scale
}

impl QuantVec {
    pub fn quantize(v: &[f32], encoding: Quantization) -> Result<Self, String> {
        if let Some(i) = v.iter().position(|x| !x.is_finite()) {
            return Err(format!("element {} is not finite", i));
        }
        Ok(match encoding {
            Quantization::F16 => {
                let mut out = Vec::with_capacity(v.len());
                for (i, &x) in v.iter().enumerate() {
                    let h = f16::from_f32(x);
                    if h.is_infinite() {
                        return Err(format!(
                            "element {} ({}) is outside the F16 range (±65504)",
                            i, x
                        ));
                    }
                    out.push(h.to_bits());
                }
                QuantVec::F16(out.into())
            }
            Quantization::I8 => {
                let max = v.iter().fold(0.0f32, |m, x| m.max(x.abs()));
                let scale = max / 127.0;
                let data = v
                    .iter()
                    .map(|x| {
                        if scale == 0.0 {
                            0
                        } else {
                            (x / scale).round().clamp(-127.0, 127.0) as i8
                        }
                    })
                    .collect();
                QuantVec::I8 { scale, data }
            }
        })
    }

    pub fn dequantize(&self) -> Vec<f32> {
        match self {
            QuantVec::F16(d) => d.iter().map(|h| f16_to_f32(*h)).collect(),
            QuantVec::I8 { scale, data } => data.iter().map(|q| i8_to_f32(*q, *scale)).collect(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            QuantVec::F16(d) => d.len(),
            QuantVec::I8 { data, .. } => data.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn encoding(&self) -> Quantization {
        match self {
            QuantVec::F16(_) => Quantization::F16,
            QuantVec::I8 { .. } => Quantization::I8,
        }
    }

    /// Bytes of a mapped column file these elements cover (0 when owned).
    pub fn mapped_bytes(&self) -> usize {
        match self {
            QuantVec::F16(d) => d.mapped_bytes(),
            QuantVec::I8 { data, .. } => data.mapped_bytes(),
        }
    }

    /// Heap bytes held.
    pub fn heap_bytes(&self) -> usize {
        match self {
            QuantVec::F16(d) => d.heap_bytes(),
            QuantVec::I8 { data, .. } => data.heap_bytes(),
        }
    }
}

impl PartialEq for QuantVec {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (QuantVec::F16(a), QuantVec::F16(b)) => a == b,
            (QuantVec::I8 { scale: sa, data: a }, QuantVec::I8 { scale: sb, data: b }) => {
                sa.to_bits() == sb.to_bits() && a == b
            }
            _ => false,
        }
    }
}

impl Eq for QuantVec {}

impl std::hash::Hash for QuantVec {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            QuantVec::F16(d) => {
                0u8.hash(state);
                d.hash(state);
            }
            QuantVec::I8 { scale, data } => {
                1u8.hash(state);
                scale.to_bits().hash(state);
                data.hash(state);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_dequantize() {
        let v = [0.1f32, -2.5, 3.0, 0.0, 1e-3];
        let f = QuantVec::quantize(&v, Quantization::F16).unwrap();
        let back = f.dequantize();
        for (a, b) in v.iter().zip(&back) {
            assert!((a - b).abs() <= a.abs() * 1e-3 + 1e-6, "{} vs {}", a, b);
        }
        // Re-quantizing a dequantized F16 vector is exact.
        assert_eq!(QuantVec::quantize(&back, Quantization::F16).unwrap(), f);

        let q = QuantVec::quantize(&v, Quantization::I8).unwrap();
        let QuantVec::I8 { scale, data } = &q else {
            panic!()
        };
        assert_eq!(*scale, 3.0 / 127.0);
        assert_eq!(data[2], 127);
        assert_eq!(data[1], -106); // round(-2.5 / (3/127)) = round(-105.83)
        assert_eq!(
            QuantVec::quantize(&[0.0; 3], Quantization::I8)
                .unwrap()
                .dequantize(),
            vec![0.0; 3]
        );

        assert!(QuantVec::quantize(&[1e6], Quantization::F16).is_err());
        assert!(QuantVec::quantize(&[f32::NAN], Quantization::I8).is_err());
    }

    #[test]
    fn serde_round_trip_is_exact() {
        for enc in [Quantization::F16, Quantization::I8] {
            let q = QuantVec::quantize(&[0.3, -1.7, 2.2, 0.01], enc).unwrap();
            let json = serde_json::to_string(&q).unwrap();
            let back: QuantVec = serde_json::from_str(&json).unwrap();
            assert_eq!(back, q, "{}", json);
        }
    }
}
