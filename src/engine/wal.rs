//! Write-ahead log: per-database durability for in-memory state.
//!
//! Without it, anything not explicitly `SAVE`d is lost when the process
//! stops. With `[wal] enabled = true`, every successful mutating statement
//! (`Statement::is_mutating`) is appended to `{data_dir}/{db}/wal.log` as one
//! JSON line, and on startup each database restores its last checkpoint
//! (`checkpoint/`, see `db/snapshot.rs`) and replays the log records after
//! it.
//!
//! It's a *logical* log -- statements, not bytes -- so replay re-executes
//! them. That's exact for everything computed inside the engine. For
//! statements that read an external file (`LOAD`, `IMPORT`), the record also
//! carries a fingerprint of what was loaded, and replay fails loudly if the
//! file now yields something different, rather than silently diverging. A
//! checkpoint after every `SAVE` keeps the common "LOAD, modify, SAVE"
//! workflow from ever replaying a `LOAD` against its own later `SAVE`.
//!
//! A record is appended only after its statement succeeded, so a log never
//! contains a statement that failed originally. If the process dies between
//! the statement and the append, that one statement is lost -- the same
//! window any commit-after-apply log has.

use crate::core::config::WalSync;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

pub const WAL_FILE: &str = "wal.log";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalRecord {
    pub seq: u64,
    pub ts: DateTime<Utc>,
    pub statement: String,
    /// Object name -> content fingerprint, for statements that read an
    /// external file (see `Statement::external_input_target`).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub fingerprints: BTreeMap<String, String>,
}

/// An open, append-only log for one database.
#[derive(Debug)]
pub struct Wal {
    path: PathBuf,
    file: File,
    next_seq: u64,
    sync: WalSync,
}

impl Wal {
    /// Opens (creating if needed) `{db_dir}/wal.log` for appending; the next
    /// record gets sequence number `next_seq`.
    pub fn open(db_dir: &Path, next_seq: u64, sync: WalSync) -> std::io::Result<Self> {
        std::fs::create_dir_all(db_dir)?;
        let path = db_dir.join(WAL_FILE);
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        Ok(Self {
            path,
            file,
            next_seq,
            sync,
        })
    }

    pub fn append(
        &mut self,
        statement: &str,
        fingerprints: BTreeMap<String, String>,
    ) -> std::io::Result<u64> {
        let record = WalRecord {
            seq: self.next_seq,
            ts: Utc::now(),
            statement: statement.to_string(),
            fingerprints,
        };
        let mut line = serde_json::to_string(&record).map_err(std::io::Error::other)?;
        line.push('\n');
        self.file.write_all(line.as_bytes())?;
        if self.sync == WalSync::Always {
            self.file.sync_data()?;
        }
        self.next_seq += 1;
        Ok(record.seq)
    }

    /// Sequence number of the last record appended (0 if none yet).
    pub fn last_seq(&self) -> u64 {
        self.next_seq - 1
    }

    pub fn size_bytes(&self) -> u64 {
        self.file.metadata().map(|m| m.len()).unwrap_or(0)
    }

    /// Atomically replaces the log with an empty one (write + rename), after
    /// a checkpoint has captured everything up to `last_seq()`.
    pub fn truncate(&mut self) -> std::io::Result<()> {
        let tmp = self.path.with_extension("log.new");
        File::create(&tmp)?.sync_all()?;
        std::fs::rename(&tmp, &self.path)?;
        self.file = OpenOptions::new().append(true).open(&self.path)?;
        Ok(())
    }
}

/// Rewrites `{db_dir}/wal.log` to exactly `records` (write + rename). Used
/// after `read_records` dropped a truncated tail, so the next append starts
/// on a clean line instead of being glued onto the partial record (which
/// would make it unreadable too).
pub fn rewrite(db_dir: &Path, records: &[WalRecord]) -> std::io::Result<()> {
    let path = db_dir.join(WAL_FILE);
    let tmp = path.with_extension("log.new");
    let mut out = String::new();
    for record in records {
        out.push_str(&serde_json::to_string(record).map_err(std::io::Error::other)?);
        out.push('\n');
    }
    let mut file = File::create(&tmp)?;
    file.write_all(out.as_bytes())?;
    file.sync_all()?;
    std::fs::rename(&tmp, &path)
}

/// The records in `{db_dir}/wal.log`, in order. A missing file is an empty
/// log.
///
/// A final line that doesn't parse is what a crash mid-append leaves
/// behind: it's dropped and reported as a warning (`Ok((records,
/// Some(warning)))`). An unparseable line *before* the last one, or a gap
/// or repeat in sequence numbers, is corruption and an error.
pub fn read_records(db_dir: &Path) -> Result<(Vec<WalRecord>, Option<String>), String> {
    let path = db_dir.join(WAL_FILE);
    let file = match File::open(&path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok((Vec::new(), None)),
        Err(e) => return Err(format!("cannot read {}: {}", path.display(), e)),
    };
    let lines: Vec<String> = BufReader::new(file)
        .lines()
        .collect::<Result<_, _>>()
        .map_err(|e| format!("cannot read {}: {}", path.display(), e))?;

    let mut records: Vec<WalRecord> = Vec::with_capacity(lines.len());
    let mut warning = None;
    for (i, line) in lines.iter().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<WalRecord>(line) {
            Ok(record) => {
                if let Some(prev) = records.last() {
                    if record.seq != prev.seq + 1 {
                        return Err(format!(
                            "{}: record at line {} has seq {} after seq {}",
                            path.display(),
                            i + 1,
                            record.seq,
                            prev.seq
                        ));
                    }
                }
                records.push(record);
            }
            Err(e) if i == lines.len() - 1 => {
                warning = Some(format!(
                    "{}: dropped a truncated final record (line {}): {}",
                    path.display(),
                    i + 1,
                    e
                ));
            }
            Err(e) => {
                return Err(format!(
                    "{}: corrupt record at line {}: {}",
                    path.display(),
                    i + 1,
                    e
                ));
            }
        }
    }
    Ok((records, warning))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_then_read_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let mut wal = Wal::open(dir.path(), 1, WalSync::Always).unwrap();
        wal.append("VECTOR v = [1]", BTreeMap::new()).unwrap();
        let mut fp = BTreeMap::new();
        fp.insert("t".to_string(), "abc".to_string());
        wal.append("LOAD TENSOR t", fp).unwrap();
        assert_eq!(wal.last_seq(), 2);

        let (records, warning) = read_records(dir.path()).unwrap();
        assert!(warning.is_none());
        assert_eq!(records.len(), 2);
        assert_eq!(records[1].statement, "LOAD TENSOR t");
        assert_eq!(records[1].fingerprints["t"], "abc");
    }

    #[test]
    fn truncated_tail_is_dropped_with_a_warning() {
        let dir = tempfile::tempdir().unwrap();
        let mut wal = Wal::open(dir.path(), 1, WalSync::Never).unwrap();
        wal.append("VECTOR v = [1]", BTreeMap::new()).unwrap();
        let mut f = OpenOptions::new()
            .append(true)
            .open(dir.path().join(WAL_FILE))
            .unwrap();
        f.write_all(b"{\"seq\":2,\"ts\":\"20").unwrap();

        let (records, warning) = read_records(dir.path()).unwrap();
        assert_eq!(records.len(), 1);
        assert!(warning.unwrap().contains("truncated"));
    }

    #[test]
    fn corruption_before_the_tail_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join(WAL_FILE),
            "not json\n{\"seq\":1,\"ts\":\"2026-01-01T00:00:00Z\",\"statement\":\"VECTOR v = [1]\"}\n",
        )
        .unwrap();
        assert!(read_records(dir.path()).unwrap_err().contains("corrupt"));
    }

    #[test]
    fn truncate_empties_the_log_and_keeps_appending() {
        let dir = tempfile::tempdir().unwrap();
        let mut wal = Wal::open(dir.path(), 1, WalSync::Always).unwrap();
        wal.append("VECTOR a = [1]", BTreeMap::new()).unwrap();
        wal.truncate().unwrap();
        wal.append("VECTOR b = [1]", BTreeMap::new()).unwrap();
        let (records, _) = read_records(dir.path()).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].seq, 2);
    }
}
