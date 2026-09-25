use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    pub storage: StorageConfig,
    /// Write-ahead log (`[wal]` in `linal.toml`). Optional: a config file
    /// without this section keeps the WAL off, the pre-WAL behavior.
    #[serde(default)]
    pub wal: WalConfig,
    /// Compute backend (`[compute]` in `linal.toml`). Optional; defaults to
    /// the CPU backend.
    #[serde(default)]
    pub compute: ComputeConfig,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ComputeConfig {
    #[serde(default)]
    pub backend: ComputeBackendKind,
}

/// Which `ComputeBackend` each database uses (`core::backend::from_config`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ComputeBackendKind {
    /// SIMD/Rayon CPU kernels.
    #[default]
    Cpu,
    /// Large dense matmuls on the GPU via wgpu, everything else on the CPU.
    /// Needs a build with the `gpu-wgpu` feature; otherwise, or without a
    /// usable GPU, falls back to `Cpu` with a warning.
    Gpu,
}

/// Write-ahead log settings -- see `engine::wal`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalConfig {
    /// Log every successful mutating statement to `{data_dir}/{db}/wal.log`
    /// and replay it on startup. Off by default.
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub sync: WalSync,
    /// Write a checkpoint (and truncate the log) once `wal.log` grows past
    /// this many bytes. Checkpoints also happen after every successful
    /// `SAVE` and on an explicit `CHECKPOINT`.
    #[serde(default = "default_checkpoint_bytes")]
    pub checkpoint_bytes: u64,
}

fn default_checkpoint_bytes() -> u64 {
    64 * 1024 * 1024
}

impl Default for WalConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            sync: WalSync::default(),
            checkpoint_bytes: default_checkpoint_bytes(),
        }
    }
}

/// When a WAL append is flushed to stable storage.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WalSync {
    /// `fsync` after every record: survives a power loss or OS crash.
    #[default]
    Always,
    /// Written to the OS page cache only: survives the `linal` process
    /// crashing or being killed, not a power loss or OS crash.
    Never,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub data_dir: PathBuf,
    pub default_db: String,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            storage: StorageConfig {
                data_dir: PathBuf::from("./data"),
                default_db: "default".to_string(),
            },
            wal: WalConfig::default(),
            compute: ComputeConfig::default(),
        }
    }
}

impl EngineConfig {
    pub fn load() -> Self {
        let config_path = "linal.toml";
        if let Ok(content) = fs::read_to_string(config_path) {
            match toml::from_str(&content) {
                Ok(config) => return config,
                Err(e) => eprintln!(
                    "Warning: Failed to parse linal.toml: {}. Using defaults.",
                    e
                ),
            }
        }
        Self::default()
    }
}
