//! Per-database locking for `linal serve`.
//!
//! The server used to share one `Arc<RwLock<TensorDb>>` across every request,
//! so any statement that wasn't strictly read-only -- including every
//! `SELECT` -- took a write lock over *all* databases, and targeting a
//! database via `X-Linal-Database` meant mutating the one global active-db
//! pointer and restoring it afterwards.
//!
//! `SharedEngine` gives each database its own single-database `TensorDb`
//! (`TensorDb::from_instance`) behind its own lock. Requests against
//! different databases never wait on each other. The only cross-database
//! state is:
//!
//! - the catalog (which databases exist): `CREATE`/`DROP`/`USE DATABASE` and
//!   `SHOW DATABASES` are answered here, never by a per-database `TensorDb`;
//! - the server's active database, which a headerless `USE` persists;
//! - the pipeline registry, shared by every per-database `TensorDb`.
//!
//! Lock order: the catalog lock is never held while taking a database lock
//! (the `Arc` is cloned out first), so there's no lock-ordering cycle.

use crate::core::config::EngineConfig;
use crate::dsl::ast::{ShowTarget, Statement};
use crate::dsl::{can_execute_shared, execute_line, execute_line_shared, DslError, DslOutput};
use crate::engine::db::DatabaseInstance;
use crate::engine::{EngineError, PipelineRegistry, TensorDb};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Which database a request (or batch, job, scheduled task) executes
/// against, and what a `USE` inside it does.
#[derive(Debug, Clone)]
pub enum Session {
    /// Follows the server's active database; `USE` changes it for every
    /// later headerless request (the behavior of `/execute` without
    /// `X-Linal-Database`).
    Server,
    /// Pinned to one database (an `X-Linal-Database` header, a scheduled
    /// task's `target_db`); `USE` only moves this session, never the server.
    Pinned(String),
    /// A batch without the header: starts on the server's active database
    /// (snapshot at batch start), and a `USE` moves both this batch *and*
    /// the server -- the same persistence a headerless `USE` has.
    Following(String),
}

pub struct SharedEngine {
    config: EngineConfig,
    dbs: RwLock<HashMap<String, Arc<RwLock<TensorDb>>>>,
    server_active: RwLock<String>,
    pipelines: PipelineRegistry,
}

impl SharedEngine {
    /// Takes ownership of every database in `db` (see
    /// `TensorDb::take_single_database_engines`).
    pub fn from_tensor_db(db: &mut TensorDb) -> Self {
        let config = db.config.clone();
        let pipelines = db.pipelines.clone();
        let (engines, active) = db.take_single_database_engines();
        let dbs = engines
            .into_iter()
            .map(|e| (e.active_db().to_string(), Arc::new(RwLock::new(e))))
            .collect();
        Self {
            config,
            dbs: RwLock::new(dbs),
            server_active: RwLock::new(active),
            pipelines,
        }
    }

    pub fn server_active(&self) -> String {
        self.server_active.read().unwrap().clone()
    }

    /// Sorted database names.
    pub fn list_databases(&self) -> Vec<String> {
        let mut names: Vec<String> = self.dbs.read().unwrap().keys().cloned().collect();
        names.sort();
        names
    }

    /// The database's own engine and lock.
    pub fn database(&self, name: &str) -> Result<Arc<RwLock<TensorDb>>, EngineError> {
        self.dbs
            .read()
            .unwrap()
            .get(name)
            .cloned()
            .ok_or_else(|| EngineError::InvalidOp(format!("Database '{}' not found", name)))
    }

    /// Same checks and messages as `TensorDb::create_database`.
    pub fn create_database(&self, name: String) -> Result<(), EngineError> {
        let mut dbs = self.dbs.write().unwrap();
        if dbs.contains_key(&name) {
            return Err(EngineError::InvalidOp(format!(
                "Database '{}' already exists",
                name
            )));
        }
        let db_path = self.config.storage.data_dir.join(&name);
        if !db_path.exists() {
            std::fs::create_dir_all(&db_path).map_err(|e| {
                EngineError::InvalidOp(format!("Failed to create DB directory: {}", e))
            })?;
        }
        let mut instance = DatabaseInstance::new(name.clone(), db_path);
        instance.backend = crate::core::backend::from_config(&self.config.compute);
        let engine = TensorDb::from_instance(self.config.clone(), instance, self.pipelines.clone());
        dbs.insert(name, Arc::new(RwLock::new(engine)));
        Ok(())
    }

    /// Same checks and messages as `TensorDb::drop_database`. Waits for any
    /// in-flight statement on the database to finish before deleting its
    /// directory.
    pub fn drop_database(&self, name: &str) -> Result<(), EngineError> {
        if name == "default" {
            return Err(EngineError::InvalidOp(
                "Cannot drop the 'default' database".to_string(),
            ));
        }
        let removed = self
            .dbs
            .write()
            .unwrap()
            .remove(name)
            .ok_or_else(|| EngineError::InvalidOp(format!("Database '{}' not found", name)))?;

        {
            let mut active = self.server_active.write().unwrap();
            if active.as_str() == name {
                *active = "default".to_string();
            }
        }

        // Taking the write lock waits out anything still executing against it.
        drop(removed.write().unwrap());

        let db_path = self.config.storage.data_dir.join(name);
        if db_path.exists() {
            std::fs::remove_dir_all(&db_path).map_err(|e| {
                EngineError::InvalidOp(format!("Failed to remove DB directory: {}", e))
            })?;
        }
        Ok(())
    }

    /// Makes `name` the server's active database (what a headerless `USE`
    /// does), after checking it exists.
    pub fn use_database_for_server(&self, name: &str) -> Result<(), EngineError> {
        self.database(name)?;
        *self.server_active.write().unwrap() = name.to_string();
        Ok(())
    }

    /// The database `session` currently targets.
    pub fn resolve(&self, session: &Session) -> String {
        match session {
            Session::Server => self.server_active(),
            Session::Pinned(name) | Session::Following(name) => name.clone(),
        }
    }

    /// Executes one statement for `session`. Catalog statements are answered
    /// here; everything else runs on the target database's own `TensorDb`,
    /// under a read lock when `can_execute_shared` allows it and a write
    /// lock otherwise.
    pub fn execute(
        &self,
        session: &mut Session,
        line: &str,
        line_no: usize,
    ) -> Result<DslOutput, DslError> {
        if let Some(result) = self.execute_catalog(session, line, line_no) {
            return result;
        }
        let target = self.resolve(session);
        let db = self
            .database(&target)
            .map_err(|e| DslError::Engine { line: 0, source: e })?;
        run_on(&db, line, line_no)
    }

    /// Executes a sequence of statements for `session`, stopping at the
    /// first error. Consecutive statements on the same database run under
    /// one write-lock hold, so -- as before per-database locking -- another
    /// request can't interleave with them on that database.
    pub fn execute_batch<'a>(
        &self,
        session: &mut Session,
        statements: impl IntoIterator<Item = (&'a str, usize)>,
    ) -> Vec<(String, Result<DslOutput, DslError>)> {
        let mut results = Vec::new();
        let mut iter = statements.into_iter().peekable();

        while let Some(&(line, line_no)) = iter.peek() {
            if let Some(result) = self.execute_catalog(session, line, line_no) {
                iter.next();
                let failed = result.is_err();
                results.push((line.to_string(), result));
                if failed {
                    return results;
                }
                continue;
            }

            let target = self.resolve(session);
            let db = match self.database(&target) {
                Ok(db) => db,
                Err(e) => {
                    iter.next();
                    results.push((
                        line.to_string(),
                        Err(DslError::Engine { line: 0, source: e }),
                    ));
                    return results;
                }
            };
            let mut guard = db.write().unwrap();
            while let Some(&(line, line_no)) = iter.peek() {
                if is_catalog_statement(line) {
                    break;
                }
                iter.next();
                let result = execute_line(&mut guard, line, line_no);
                let failed = result.is_err();
                results.push((line.to_string(), result));
                if failed {
                    return results;
                }
            }
        }
        results
    }

    /// `Some` when `line` is a catalog statement (`CREATE`/`DROP`/`USE
    /// DATABASE`, `SHOW DATABASES`), answered with the same messages the
    /// per-database executor (`dsl::executor`) uses.
    fn execute_catalog(
        &self,
        session: &mut Session,
        line: &str,
        line_no: usize,
    ) -> Option<Result<DslOutput, DslError>> {
        let engine_err = |e| DslError::Engine {
            line: line_no,
            source: e,
        };
        let stmt = crate::dsl::parser::parse(line).ok()?;
        Some(match stmt {
            Statement::CreateDatabase(s) => {
                if s.if_not_exists && self.list_databases().contains(&s.name) {
                    return Some(Ok(DslOutput::Message(format!(
                        "Database '{}' already exists (skipped)",
                        s.name
                    ))));
                }
                self.create_database(s.name.clone())
                    .map(|_| DslOutput::Message(format!("Created database: {}", s.name)))
                    .map_err(engine_err)
            }
            Statement::DropDatabase(s) => {
                if s.if_exists && !self.list_databases().contains(&s.name) {
                    return Some(Ok(DslOutput::Message(format!(
                        "Database '{}' not found (skipped)",
                        s.name
                    ))));
                }
                match self.drop_database(&s.name) {
                    Ok(()) => {
                        // A session sitting on the dropped database falls
                        // back to `default`, as `TensorDb::drop_database`
                        // does for its own active database.
                        if let Session::Pinned(n) | Session::Following(n) = session {
                            if n == &s.name {
                                *n = "default".to_string();
                            }
                        }
                        Ok(DslOutput::Message(format!("Dropped database: {}", s.name)))
                    }
                    Err(e) => Err(engine_err(e)),
                }
            }
            Statement::UseDatabase(s) => {
                if let Err(e) = self.database(&s.name) {
                    return Some(Err(engine_err(e)));
                }
                match session {
                    Session::Server => *self.server_active.write().unwrap() = s.name.clone(),
                    Session::Pinned(n) => *n = s.name.clone(),
                    Session::Following(n) => {
                        *n = s.name.clone();
                        *self.server_active.write().unwrap() = s.name.clone();
                    }
                }
                Ok(DslOutput::Message(format!(
                    "Switched to database '{}'",
                    s.name
                )))
            }
            Statement::Show(s) if matches!(s.target, ShowTarget::AllDatabases) => {
                let mut output = String::from("--- ALL DATABASES ---\n");
                for name in self.list_databases() {
                    output.push_str(&format!("  - {}\n", name));
                }
                output.push_str("---------------------");
                Ok(DslOutput::Message(output))
            }
            _ => return None,
        })
    }
}

fn is_catalog_statement(line: &str) -> bool {
    match crate::dsl::parser::parse(line) {
        Ok(Statement::CreateDatabase(_))
        | Ok(Statement::DropDatabase(_))
        | Ok(Statement::UseDatabase(_)) => true,
        Ok(Statement::Show(s)) => matches!(s.target, ShowTarget::AllDatabases),
        _ => false,
    }
}

/// Runs `line` on one database: read lock when the statement can't mutate
/// it (checked and executed under the same guard), write lock otherwise.
fn run_on(db: &RwLock<TensorDb>, line: &str, line_no: usize) -> Result<DslOutput, DslError> {
    {
        let guard = db.read().unwrap();
        if can_execute_shared(&guard, line) {
            return execute_line_shared(&guard, line, line_no);
        }
    }
    let mut guard = db.write().unwrap();
    execute_line(&mut guard, line, line_no)
}
