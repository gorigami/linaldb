use clap::{Parser, Subcommand};
use colored::*;
use linal::dsl::{execute_line, DslOutput};
use linal::engine::TensorDb;
use linal::server::start_server;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::fs;
use std::sync::{Arc, RwLock};
use toon_format::encode_default;

#[derive(Parser)]
#[command(name = "LINAL")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "LINAL: Linear Algebra Analytical Engine", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start REPL (default)
    Repl {
        /// Output format: 'display' (default, human-readable) or 'toon' (machine-readable)
        #[arg(long, default_value = "display")]
        format: String,
    },
    /// Run a script file
    Run {
        /// Path to the script file (.lnl)
        file: String,
        /// Output format: 'display' (default, human-readable) or 'toon' (machine-readable)
        #[arg(long, default_value = "display")]
        format: String,
    },
    /// Server management
    Server {
        #[command(subcommand)]
        action: Option<ServerAction>,
        #[arg(long, default_value_t = 8080)]
        port: u16,
    },
    /// Start HTTP server (shorthand for server)
    Serve {
        #[arg(long, default_value_t = 8080)]
        port: u16,
    },
    /// Initialize a new LINAL project structure
    Init,
    /// Load a Parquet file directly into a dataset
    Load {
        /// Path to the parquet file
        file: String,
        /// Target dataset name
        dataset: String,
    },
    /// Manage database instances
    Db {
        #[command(subcommand)]
        action: DbAction,
    },
    /// Run a query against a local or remote LINAL instance
    Query {
        /// The DSL command to execute
        dsl: String,
        /// Remote server URL (optional, e.g., http://localhost:8080)
        #[arg(long)]
        url: Option<String>,
        /// Database name to use
        #[arg(long, short)]
        db: Option<String>,
        /// Output format (display or toon)
        #[arg(long, default_value = "display")]
        format: String,
    },
    /// Execute a DSL command directly (embedded mode)
    Exec {
        /// The DSL command string
        command: String,
        /// Output format (display or toon)
        #[arg(long, default_value = "display")]
        format: String,
    },
    /// Manage background jobs on a running server
    Jobs {
        #[command(subcommand)]
        action: JobsAction,
        /// Server URL
        #[arg(long, default_value = "http://localhost:8080")]
        url: String,
    },
    /// Manage scheduled (recurring) tasks on a running server
    Schedule {
        #[command(subcommand)]
        action: ScheduleAction,
        /// Server URL
        #[arg(long, default_value = "http://localhost:8080")]
        url: String,
    },
}

#[derive(Subcommand)]
enum JobsAction {
    /// List all jobs
    List,
    /// Submit a DSL command as a background job
    Submit {
        /// The DSL command to run
        command: String,
        /// Database name to target
        #[arg(long, short)]
        db: Option<String>,
    },
    /// Get a job's status
    Get { id: String },
    /// Cancel a pending job
    Cancel { id: String },
    /// Get a completed job's result
    Result { id: String },
}

#[derive(Subcommand)]
enum ScheduleAction {
    /// List all scheduled tasks
    List,
    /// Create a new recurring task
    Create {
        /// Human-readable task name
        name: String,
        /// The DSL command to run on each interval
        command: String,
        /// Interval between runs, in seconds
        interval_secs: u64,
        /// Database name to target
        #[arg(long, short)]
        db: Option<String>,
    },
    /// Delete a scheduled task
    Delete { id: String },
}

#[derive(Subcommand)]
enum ServerAction {
    /// Start the server (default)
    Start,
    /// Check server status
    Status,
    /// Stop the server gracefully
    Stop,
}

#[derive(Subcommand)]
enum DbAction {
    /// List all databases
    List,
    /// Create a new database
    Create { name: String },
    /// Drop a database
    Drop { name: String },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let mut db = TensorDb::new();

    match cli.command {
        Some(Commands::Run { file, format }) => {
            let content = fs::read_to_string(&file)?;
            let use_toon = format == "toon";

            let mut current_cmd = String::new();
            let mut start_line = 0;
            let mut paren_balance = 0;

            for (idx, raw_line) in content.lines().enumerate() {
                let line = raw_line.trim();

                if current_cmd.is_empty() {
                    if line.is_empty()
                        || line.starts_with('#')
                        || line.starts_with("//")
                        || line.starts_with("--")
                    {
                        continue;
                    }
                    start_line = idx + 1;
                }

                if !current_cmd.is_empty() {
                    current_cmd.push(' ');
                }
                current_cmd.push_str(line);

                for c in line.chars() {
                    if c == '(' {
                        paren_balance += 1;
                    } else if c == ')' {
                        paren_balance -= 1;
                    }
                }

                if paren_balance == 0 {
                    match execute_line(&mut db, &current_cmd, start_line) {
                        Ok(output) => {
                            if !matches!(output, DslOutput::None) {
                                if use_toon {
                                    let toon = encode_default(&output)
                                        .unwrap_or_else(|e| format!("Error encoding TOON: {}", e));
                                    println!("{}", toon);
                                } else {
                                    println!("{}", output);
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("Error on line {}: {}", start_line, e);
                            std::process::exit(1);
                        }
                    }
                    current_cmd.clear();
                }
            }

            if !current_cmd.is_empty() {
                eprintln!(
                    "Error: Script ended with unbalanced parentheses starting at line {}",
                    start_line
                );
                std::process::exit(1);
            }
        }
        Some(Commands::Server { action, port }) => {
            let action = action.unwrap_or(ServerAction::Start);
            match action {
                ServerAction::Start => {
                    let db_arc = Arc::new(RwLock::new(db));
                    write_pid_file(port);
                    start_server(db_arc, port).await;
                    remove_pid_file(port);
                }
                ServerAction::Status => {
                    handle_server_status(port).await?;
                }
                ServerAction::Stop => {
                    handle_server_stop(port).await?;
                }
            }
        }
        Some(Commands::Serve { port }) => {
            let db_arc = Arc::new(RwLock::new(db));
            write_pid_file(port);
            start_server(db_arc, port).await;
            remove_pid_file(port);
        }
        Some(Commands::Init) => {
            handle_init()?;
        }
        Some(Commands::Load { file, dataset }) => {
            handle_load(&mut db, &file, &dataset)?;
        }
        Some(Commands::Db { action }) => {
            handle_db(&mut db, action)?;
        }
        Some(Commands::Query {
            dsl,
            url,
            db: target_db,
            format,
        }) => {
            handle_query(&mut db, dsl, url, target_db, format == "toon").await?;
        }
        Some(Commands::Exec { command, format }) => {
            handle_exec(&mut db, command, format == "toon")?;
        }
        Some(Commands::Jobs { action, url }) => {
            handle_jobs(action, url).await?;
        }
        Some(Commands::Schedule { action, url }) => {
            handle_schedule(action, url).await?;
        }
        Some(Commands::Repl { format }) => {
            run_repl(db, format == "toon")?;
        }
        None => {
            run_repl(db, false)?;
        }
    }

    Ok(())
}

fn handle_init() -> Result<(), Box<dyn std::error::Error>> {
    let data_dir = "./data";
    if !std::path::Path::new(data_dir).exists() {
        fs::create_dir_all(data_dir)?;
        println!("Created directory: {}", data_dir.green());
    }

    let config_path = "linal.toml";
    if !std::path::Path::new(config_path).exists() {
        let default_config = r#"[storage]
data_dir = "./data"
default_db = "default"
"#;
        fs::write(config_path, default_config)?;
        println!("Created default configuration: {}", config_path.green());
    } else {
        println!(
            "Configuration file already exists: {}",
            config_path.yellow()
        );
    }

    println!(
        "{}",
        "Initialization complete. Welcome to LINAL!".bold().blue()
    );
    Ok(())
}

fn handle_load(
    db: &mut TensorDb,
    file: &str,
    dataset: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let command = format!("LOAD DATASET {} FROM \"{}\"", dataset, file);
    match execute_line(db, &command, 1) {
        Ok(output) => {
            println!("{}", output.to_string().green());
            Ok(())
        }
        Err(e) => {
            eprintln!("{}: {}", "Error loading dataset".red(), e);
            Err(e.into())
        }
    }
}

fn run_repl(mut db: TensorDb, use_toon: bool) -> Result<(), Box<dyn std::error::Error>> {
    let mut rl = DefaultEditor::new()?;
    let history_path = ".linal_history";

    if rl.load_history(history_path).is_err() {
        // No history yet
    }

    println!("{}", "LINAL REPL v0.1".bold().blue());
    if use_toon {
        println!("Output format: {}", "TOON (machine-readable)".yellow());
    } else {
        println!("Output format: {}", "Display (human-readable)".yellow());
    }
    println!("Type 'EXIT' or use Ctrl-D to quit.");

    let mut current_cmd = String::new();
    let mut paren_balance = 0;

    loop {
        let active_db = db.active_db();
        let prompt = if paren_balance == 0 {
            format!("{} >_>  ", active_db.blue())
        } else {
            " ..  ".to_string()
        };
        let readline = rl.readline(&prompt);

        match readline {
            Ok(line) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }

                if trimmed.eq_ignore_ascii_case("EXIT") {
                    break;
                }

                // Handle meta-commands
                if trimmed.starts_with(".use ") {
                    let new_db = trimmed.strip_prefix(".use ").unwrap().trim();
                    match db.use_database(new_db) {
                        Ok(_) => println!("Switched to database: {}", new_db.green()),
                        Err(e) => eprintln!("{}: {}", "Error".red(), e),
                    }
                    continue;
                }

                rl.add_history_entry(trimmed)?;

                if !current_cmd.is_empty() {
                    current_cmd.push(' ');
                }
                current_cmd.push_str(trimmed);

                for c in trimmed.chars() {
                    if c == '(' {
                        paren_balance += 1;
                    } else if c == ')' {
                        paren_balance -= 1;
                    }
                }

                if paren_balance == 0 {
                    match execute_line(&mut db, &current_cmd, 1) {
                        Ok(output) => {
                            if !matches!(output, DslOutput::None) {
                                if use_toon {
                                    let toon = encode_default(&output)
                                        .unwrap_or_else(|e| format!("Error encoding TOON: {}", e));
                                    println!("{}", toon);
                                } else {
                                    println!("{}", output);
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("{}: {}", "Error".red(), e);
                        }
                    }
                    current_cmd.clear();
                }
            }
            Err(ReadlineError::Interrupted) => {
                println!("Interrupted");
                current_cmd.clear();
                paren_balance = 0;
                continue;
            }
            Err(ReadlineError::Eof) => {
                break;
            }
            Err(err) => {
                println!("Error: {:?}", err);
                break;
            }
        }
    }

    let _ = rl.save_history(history_path);
    Ok(())
}

fn handle_db(db: &mut TensorDb, action: DbAction) -> Result<(), Box<dyn std::error::Error>> {
    match action {
        DbAction::List => {
            let dbs = db.list_databases();
            println!("{}", "Databases:".bold().blue());
            for name in dbs {
                println!("  - {}", name.cyan());
            }
        }
        DbAction::Create { name } => {
            db.create_database(name.clone())?;
            println!("{} Database '{}' created.", "✓".green(), name.bold());
        }
        DbAction::Drop { name } => {
            db.drop_database(&name)?;
            println!("{} Database '{}' dropped.", "✓".yellow(), name.bold());
        }
    }
    Ok(())
}

async fn handle_query(
    db: &mut TensorDb,
    dsl: String,
    url: Option<String>,
    target_db: Option<String>,
    use_toon: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(server_url) = url {
        let client = reqwest::Client::new();
        let mut req = client.post(format!("{}/execute", server_url)).body(dsl);

        if let Some(db_name) = target_db {
            req = req.header("X-Linal-Database", db_name);
        }

        if use_toon {
            req = req.query(&[("format", "toon")]);
        } else {
            req = req.query(&[("format", "json")]);
        }

        let resp = req.send().await?;
        let status = resp.status();
        let body = resp.text().await?;

        if status.is_success() {
            println!("{}", body);
        } else {
            eprintln!("{} Remote error ({}): {}", "✗".red(), status, body.red());
        }
    } else {
        if let Some(db_name) = target_db {
            db.use_database(&db_name)?;
        }
        match execute_line(db, &dsl, 1) {
            Ok(output) => {
                if !matches!(output, DslOutput::None) {
                    if use_toon {
                        let toon = encode_default(&output)?;
                        println!("{}", toon);
                    } else {
                        println!("{}", output);
                    }
                }
            }
            Err(e) => {
                eprintln!("{}: {}", "Error".red(), e);
            }
        }
    }
    Ok(())
}

fn handle_exec(
    db: &mut TensorDb,
    command: String,
    use_toon: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    match execute_line(db, &command, 1) {
        Ok(output) => {
            if !matches!(output, DslOutput::None) {
                if use_toon {
                    println!(
                        "{}",
                        encode_default(&output)
                            .unwrap_or_else(|e| format!("Error encoding TOON: {}", e))
                    );
                } else {
                    println!("{}", output);
                }
            }
            Ok(())
        }
        Err(e) => {
            eprintln!("{}: {}", "Error".red(), e);
            Err(e.into())
        }
    }
}
async fn handle_server_status(port: u16) -> Result<(), Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    let url = format!("http://localhost:{}/health", port);
    match client.get(&url).send().await {
        Ok(resp) => {
            if resp.status().is_success() {
                println!("{} Server is running on port {}", "✓".green(), port);
                let body = resp.text().await?;
                println!("Status: {}", body.cyan());
            } else {
                println!(
                    "{} Server on port {} returned error: {}",
                    "✗".red(),
                    port,
                    resp.status()
                );
            }
        }
        Err(_) => {
            println!("{} Server is not running on port {}", "✗".red(), port);
        }
    }
    Ok(())
}

fn pid_file_path(port: u16) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("linal_server_{}.pid", port))
}

fn write_pid_file(port: u16) {
    let path = pid_file_path(port);
    let _ = fs::write(path, std::process::id().to_string());
}

fn remove_pid_file(port: u16) {
    let _ = fs::remove_file(pid_file_path(port));
}

async fn handle_server_stop(port: u16) -> Result<(), Box<dyn std::error::Error>> {
    let path = pid_file_path(port);
    let pid: u32 = match fs::read_to_string(&path).ok().and_then(|s| s.trim().parse().ok()) {
        Some(pid) => pid,
        None => {
            println!(
                "{} No PID file found for port {} (server may not be running, or was started outside this CLI).",
                "!".yellow(),
                port
            );
            return Ok(());
        }
    };

    #[cfg(unix)]
    let kill_result = std::process::Command::new("kill")
        .args(["-TERM", &pid.to_string()])
        .status();
    #[cfg(windows)]
    let kill_result = std::process::Command::new("taskkill")
        .args(["/PID", &pid.to_string(), "/F"])
        .status();

    match kill_result {
        Ok(status) if status.success() => {
            println!(
                "{} Sent stop signal to server (pid {}) on port {}.",
                "✓".green(),
                pid,
                port
            );
            let _ = fs::remove_file(&path);
        }
        _ => {
            println!(
                "{} Could not signal process {} (it may already be stopped). Cleaning up stale PID file.",
                "!".yellow(),
                pid
            );
            let _ = fs::remove_file(&path);
        }
    }
    Ok(())
}

async fn handle_jobs(action: JobsAction, url: String) -> Result<(), Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    match action {
        JobsAction::List => {
            let resp = client.get(format!("{}/jobs", url)).send().await?;
            print_json_response(resp).await?;
        }
        JobsAction::Submit { command, db } => {
            let mut req = client.post(format!("{}/jobs", url)).body(command);
            if let Some(db_name) = db {
                req = req.header("X-Linal-Database", db_name);
            }
            let resp = req.send().await?;
            print_json_response(resp).await?;
        }
        JobsAction::Get { id } => {
            let resp = client.get(format!("{}/jobs/{}", url, id)).send().await?;
            print_json_response(resp).await?;
        }
        JobsAction::Cancel { id } => {
            let resp = client.delete(format!("{}/jobs/{}", url, id)).send().await?;
            print_json_response(resp).await?;
        }
        JobsAction::Result { id } => {
            let resp = client
                .get(format!("{}/jobs/{}/result", url, id))
                .send()
                .await?;
            print_json_response(resp).await?;
        }
    }
    Ok(())
}

async fn handle_schedule(
    action: ScheduleAction,
    url: String,
) -> Result<(), Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    match action {
        ScheduleAction::List => {
            let resp = client.get(format!("{}/schedule", url)).send().await?;
            print_json_response(resp).await?;
        }
        ScheduleAction::Create {
            name,
            command,
            interval_secs,
            db,
        } => {
            let body = serde_json::json!({
                "name": name,
                "command": command,
                "interval_secs": interval_secs,
                "target_db": db,
            });
            let resp = client
                .post(format!("{}/schedule", url))
                .json(&body)
                .send()
                .await?;
            print_json_response(resp).await?;
        }
        ScheduleAction::Delete { id } => {
            let resp = client
                .delete(format!("{}/schedule/{}", url, id))
                .send()
                .await?;
            print_json_response(resp).await?;
        }
    }
    Ok(())
}

async fn print_json_response(resp: reqwest::Response) -> Result<(), Box<dyn std::error::Error>> {
    let status = resp.status();
    let body = resp.text().await?;
    let pretty = serde_json::from_str::<serde_json::Value>(&body)
        .map(|v| serde_json::to_string_pretty(&v).unwrap_or(body.clone()))
        .unwrap_or(body);
    if status.is_success() {
        println!("{}", pretty);
    } else {
        eprintln!("{} ({}): {}", "Error".red(), status, pretty.red());
    }
    Ok(())
}
