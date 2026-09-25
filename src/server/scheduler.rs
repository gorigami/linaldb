use super::engine::{Session, SharedEngine};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::time::sleep;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledTask {
    pub id: Uuid,
    pub name: String,
    pub command: String,
    pub interval_secs: u64,
    pub target_db: Option<String>,
    pub last_run: Option<chrono::DateTime<chrono::Utc>>,
}

pub struct Scheduler {
    engine: Arc<SharedEngine>,
    tasks: Arc<Mutex<Vec<ScheduledTask>>>,
}

impl Scheduler {
    pub fn new(engine: Arc<SharedEngine>) -> Self {
        Self {
            engine,
            tasks: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn add_task(
        &self,
        name: String,
        command: String,
        interval_secs: u64,
        target_db: Option<String>,
    ) -> Uuid {
        let task = ScheduledTask {
            id: Uuid::new_v4(),
            name,
            command,
            interval_secs,
            target_db,
            last_run: None,
        };
        self.tasks.lock().unwrap().push(task.clone());
        task.id
    }

    pub fn list_tasks(&self) -> Vec<ScheduledTask> {
        self.tasks.lock().unwrap().clone()
    }

    pub fn remove_task(&self, id: Uuid) -> bool {
        let mut tasks = self.tasks.lock().unwrap();
        let len_before = tasks.len();
        tasks.retain(|t| t.id != id);
        tasks.len() < len_before
    }

    pub async fn start(self: Arc<Self>) {
        println!("Scheduler started");
        loop {
            let now = chrono::Utc::now();
            let mut tasks_to_run = Vec::new();

            {
                let mut tasks = self.tasks.lock().unwrap();
                for task in tasks.iter_mut() {
                    let should_run = match task.last_run {
                        None => true,
                        Some(last) => {
                            let next = last + chrono::Duration::seconds(task.interval_secs as i64);
                            now >= next
                        }
                    };

                    if should_run {
                        tasks_to_run.push(task.clone());
                        task.last_run = Some(now);
                    }
                }
            }

            for task in tasks_to_run {
                let engine = self.engine.clone();
                tokio::task::spawn_blocking(move || {
                    // A task's `target_db` switch is permanent by design: it
                    // becomes the server's active database for later headerless
                    // requests too (see `docs/ARCHITECTURE.md`, "`/schedule`'s
                    // `target_db` switch is permanent"). An unknown `target_db`
                    // is ignored and the task runs on the current active
                    // database, as before.
                    let mut session = match &task.target_db {
                        Some(db_name) if engine.use_database_for_server(db_name).is_ok() => {
                            Session::Following(db_name.clone())
                        }
                        _ => Session::Server,
                    };
                    println!("Running scheduled task '{}': {}", task.name, task.command);
                    match engine.execute(&mut session, &task.command, 0) {
                        Ok(out) => println!("Task '{}' result: {:?}", task.name, out),
                        Err(e) => eprintln!("Task '{}' error: {}", task.name, e),
                    }
                });
            }

            sleep(Duration::from_secs(1)).await;
        }
    }
}
