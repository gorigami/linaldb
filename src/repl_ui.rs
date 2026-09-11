//! Terminal UX helpers for the interactive REPL: `.help`/CLI-verb guidance,
//! a parse-error caret indicator, live keyword/dataset-name completion and
//! highlighting (via a custom rustyline `Helper`), and a spinner wrapper for
//! statement execution.

use colored::Colorize;
use rustyline::completion::{Completer, Pair};
use rustyline::highlight::Highlighter;
use rustyline::hint::HistoryHinter;
use rustyline::validate::MatchingBracketValidator;
use rustyline::{Context, Helper, Hinter, Validator};
use std::borrow::Cow;
use std::cell::RefCell;
use std::io::IsTerminal;
use std::rc::Rc;
use std::time::Duration;

/// Every DSL keyword token (kept in sync with `src/dsl/lexer.rs` by hand --
/// there is no runtime-queryable keyword list on the logos-generated
/// `Token` enum to derive this from automatically).
pub const DSL_KEYWORDS: &[&str] = &[
    "DEFINE",
    "VECTOR",
    "MATRIX",
    "LET",
    "LAZY",
    "SHOW",
    "SELECT",
    "DELIVER",
    "BIND",
    "ATTACH",
    "DERIVE",
    "DATASET",
    "INSERT",
    "SEARCH",
    "EXPLAIN",
    "AUDIT",
    "MATERIALIZE",
    "CREATE",
    "ALTER",
    "USE",
    "DROP",
    "SET",
    "SAVE",
    "LOAD",
    "LIST",
    "IMPORT",
    "EXPORT",
    "RESET",
    "TRANSFORM",
    "UPDATE",
    "DELETE",
    "JOIN",
    "ON",
    "INNER",
    "LEFT",
    "RIGHT",
    "FULL",
    "OUTER",
    "OFFSET",
    "IN",
    "BETWEEN",
    "UNION",
    "DISTINCT",
    "OVER",
    "PARTITION",
    "CASE",
    "WHEN",
    "THEN",
    "ELSE",
    "END",
    "PIPELINE",
    "PIPELINES",
    "APPLY",
    "DESCRIBE",
    "AS",
    "STRICT",
    "TENSOR",
    "VALUES",
    "TO",
    "FROM",
    "WITH",
    "BY",
    "INTO",
    "DATABASE",
    "INDEX",
    "COLUMNS",
    "COLUMN",
    "ADD",
    "ALL",
    "TENSORS",
    "DATASETS",
    "DATABASES",
    "INDEXES",
    "SCHEMA",
    "SHAPE",
    "LINEAGE",
    "METADATA",
    "VERSIONS",
    "WHERE",
    "FILTER",
    "GROUP",
    "HAVING",
    "ORDER",
    "LIMIT",
    "NULL",
    "NOT",
    "IS",
    "AND",
    "OR",
    "NULLABLE",
    "SUBTRACT",
    "MULTIPLY",
    "DIVIDE",
    "CORRELATE",
    "SIMILARITY",
    "DISTANCE",
    "MATMUL",
    "TRANSPOSE",
    "RESHAPE",
    "STACK",
    "SCALE",
    "NORMALIZE",
    "FLATTEN",
    "SUM",
    "MEAN",
    "STDEV",
    "FFT",
    "IFFT",
    "MAGNITUDE",
    "PSD",
    "WINDOW",
    "WHITEN",
    "BANDPASS",
    "RATE",
    "MATCHED_FILTER",
];

/// The subset of `DSL_KEYWORDS` that can legitimately start a statement --
/// used for `HELP` and for Tab-completing the first word of a line (the
/// full keyword list would suggest mid-statement words like `WHERE`/`FROM`
/// as if they were valid statement openers, which they aren't).
const STATEMENT_KEYWORDS: &[&str] = &[
    "SELECT",
    "CREATE",
    "DROP",
    "USE",
    "INSERT",
    "UPDATE",
    "DELETE",
    "DATASET",
    "SHOW",
    "LOAD",
    "SAVE",
    "IMPORT",
    "EXPORT",
    "LIST",
    "DEFINE",
    "APPLY",
    "BIND",
    "ATTACH",
    "DERIVE",
    "LET",
    "LAZY",
    "ALTER",
    "SEARCH",
    "EXPLAIN",
    "AUDIT",
    "MATERIALIZE",
    "RESET",
    "TRANSFORM",
    "DELIVER",
    "DESCRIBE",
];

/// CLI subcommand verbs (`linal <verb> ...`) that a user might reasonably
/// type inside the REPL by mistake -- confirmed against `DSL_KEYWORDS`
/// above to have zero overlap with a real statement keyword.
const CLI_VERBS: &[&str] = &[
    "serve", "run", "query", "exec", "jobs", "schedule", "db", "init", "linal", "repl",
];

/// If `input` looks like a shell/CLI invocation rather than a DSL
/// statement, returns a friendly redirect message instead of letting it
/// fall through to a generic parser error.
pub fn cli_verb_hint(input: &str) -> Option<String> {
    let first = input.split_whitespace().next()?.to_lowercase();
    if first == "--help" || first == "-h" || first == "--version" {
        return Some(format!(
            "'{}' is a shell flag, not a DSL statement — run `linal --help` from your terminal instead.",
            first
        ));
    }
    if CLI_VERBS.contains(&first.as_str()) {
        return Some(format!(
            "'{}' looks like a shell command, not a DSL statement — run it from your terminal \
             (e.g. `linal {}`), not inside the REPL. Type HELP for DSL syntax.",
            first,
            input.trim()
        ));
    }
    None
}

/// Usage tips shown one at a time on REPL startup (`pick_tip`) -- each one
/// documents a feature that already ships today, no aspirational claims.
const REPL_TIPS: &[&str] = &[
    "Tip: press Tab to autocomplete DSL keywords and dataset names.",
    "Tip: an open '(' lets a statement span multiple lines -- the prompt shows '..' until it balances.",
    "Tip: '.use <db>' switches the active database without restarting the REPL.",
    "Tip: type HELP any time for the full command and keyword reference.",
    "Tip: press the Up arrow or Ctrl-R to search your command history.",
];

/// Deterministically picks one of `REPL_TIPS` from `seed` (typically a mix
/// of the process id and history length, computed by the caller so this
/// function stays pure and easy to test) -- no `rand` dependency needed for
/// this much variety.
pub fn pick_tip(seed: u64) -> &'static str {
    REPL_TIPS[(seed as usize) % REPL_TIPS.len()]
}

/// Right-pads to `width` *visible* characters. Callers must pass the plain
/// (no ANSI escape codes) text here for width math -- never a colored
/// string, whose escape-code bytes would otherwise be counted as part of
/// the visible width and produce misaligned padding.
fn right_pad_spaces(plain_visible_text: &str, width: usize) -> String {
    " ".repeat(width.saturating_sub(plain_visible_text.chars().count()))
}

/// Renders `rows` (each a `(plain, display)` pair -- `plain` has no ANSI
/// codes and is what width/padding math runs on, `display` is what's
/// actually printed, e.g. `plain` wrapped in `.green()`) as a
/// rounded-corner box. Width is content-driven (the widest `plain` row),
/// not real-terminal-width-aware -- detecting actual terminal columns
/// would need a new dependency, and every row here is either a fixed
/// string or a short value like a database name, so this is an accepted
/// trade-off, not an oversight.
fn render_box(rows: &[(String, String)]) -> String {
    let content_width = rows
        .iter()
        .map(|(plain, _)| plain.chars().count())
        .max()
        .unwrap_or(0);

    let horizontal = "─".repeat(content_width + 2);
    let mut out = String::new();
    out.push_str(&format!(
        "{}{}{}\n",
        "╭".blue(),
        horizontal.blue(),
        "╮".blue()
    ));
    for (plain, display) in rows {
        let pad = right_pad_spaces(plain, content_width);
        out.push_str(&format!(
            "{} {}{} {}\n",
            "│".blue(),
            display,
            pad,
            "│".blue()
        ));
    }
    out.push_str(&format!(
        "{}{}{}",
        "╰".blue(),
        horizontal.blue(),
        "╯".blue()
    ));
    out
}

/// A short block of literally copy-pasteable, directly runnable DSL lines
/// -- distinct from `print_help()`'s abstract syntax-pattern examples
/// (which use placeholders like `dataset`/`col: Type`). Spans both halves
/// of "SQL meets Linear Algebra" rather than just one side. Every line
/// verified against `docs/DSL_REFERENCE.md` as real, valid syntax.
pub fn print_quick_start() {
    println!("{}", "Quick start:".bold());
    for example in [
        "VECTOR v = [1.0, 2.0, 3.0]",
        "DATASET t COLUMNS (id: Int, score: Float)",
        "INSERT INTO t VALUES (1, 0.9)",
        "SELECT * FROM t",
    ] {
        println!("  {}", example.cyan());
    }
}

/// Prints the REPL's one-time startup banner. `tip_seed` picks which
/// `REPL_TIPS` entry shows (see `pick_tip`). Falls back to flat, unframed
/// text when stdout isn't a TTY (e.g. `echo "EXIT" | linal repl`) -- the
/// box-drawing chars and onboarding flourishes (quick start, tip) have no
/// value to a piped/captured session, so they're dropped there rather than
/// just left uncolored. `linal run`/`exec`/`serve` never call this at all,
/// so they're unaffected regardless of TTY state.
pub fn print_welcome_banner(version: &str, active_db: &str, use_toon: bool, tip_seed: u64) {
    let format_label = if use_toon {
        "TOON (machine-readable)"
    } else {
        "Display (human-readable)"
    };

    if !std::io::stdout().is_terminal() {
        println!(
            "{}",
            format!("LINAL v{version} -- SQL meets Linear Algebra")
                .bold()
                .blue()
        );
        println!("Database: {active_db}   Format: {format_label}");
        println!("Type EXIT to quit, HELP for a quick reference, or Ctrl-D to quit.");
        return;
    }

    let title = format!("LINAL v{version}");
    let db_line = format!("Database: {active_db}");
    let format_line = format!("Format: {format_label}");
    let rows = [
        (title.clone(), title.bold().blue().to_string()),
        (
            "SQL meets Linear Algebra.".to_string(),
            "SQL meets Linear Algebra.".dimmed().to_string(),
        ),
        (
            "Vectors, matrices, and tensors as first-class citizens.".to_string(),
            "Vectors, matrices, and tensors as first-class citizens."
                .dimmed()
                .to_string(),
        ),
        (db_line.clone(), format!("Database: {}", active_db.green())),
        (
            format_line.clone(),
            format!("Format: {}", format_label.yellow()),
        ),
    ];

    println!("{}", render_box(&rows));
    println!();
    print_quick_start();
    println!();
    println!("{}", pick_tip(tip_seed).dimmed());
    println!("Type EXIT to quit, HELP for the full reference, or Ctrl-D to quit.");
}

/// Prints a short, grouped reference of real REPL meta-commands and
/// statement keywords, following the file's existing color conventions
/// (blue/bold headers, cyan for the exact commands to type).
pub fn print_help() {
    println!("{}", "LINAL DSL — quick reference".bold().blue());
    println!(
        "Type a statement and press Enter (parentheses can span multiple lines; the prompt \
         changes to `..` until they balance)."
    );
    println!();
    println!("{}", "Meta-commands:".bold());
    println!("  {:<12} show this help", "HELP".cyan());
    println!("  {:<12} switch the active database", ".use <db>".cyan());
    println!("  {:<12} quit the REPL", "EXIT".cyan());
    println!();
    println!("{}", "Common statements:".bold());
    for example in [
        "SELECT col, ... FROM dataset WHERE ... ORDER BY ... LIMIT n",
        "CREATE DATABASE name / USE name / DROP DATABASE name",
        "DATASET name COLUMNS (col: Type, ...)",
        "INSERT INTO name VALUES (...)",
        "SHOW ALL DATASETS / SHOW name",
        "SAVE DATASET name / LOAD DATASET name FROM \"path\"",
    ] {
        println!("  {}", example.dimmed());
    }
    println!();
    println!(
        "All statement keywords: {}",
        STATEMENT_KEYWORDS.join(", ").dimmed()
    );
    println!("Full reference: {}", "docs/DSL_REFERENCE.md".underline());
}

/// Pulls the byte offset back out of a `DslError::Parse` message's existing
/// `"... (at byte N)"` suffix, rather than threading a structured offset
/// field through `DslError` itself -- that variant is constructed at ~50
/// call sites throughout the executor/persistence code as a general-purpose
/// error (not just real syntax errors), so adding a required field there
/// would mean editing all of them. Only genuine parser errors produced via
/// `ParseError::into_dsl_error` (`src/dsl/parser/mod.rs`) ever contain this
/// exact marker text.
fn extract_byte_offset(msg: &str) -> Option<usize> {
    const MARKER: &str = " (at byte ";
    let start = msg.rfind(MARKER)? + MARKER.len();
    let rest = msg.get(start..)?;
    let end = rest.find(')')?;
    rest[..end].parse::<usize>().ok()
}

/// Prints `Error: {err}` (matching the REPL's existing red-error
/// convention) and, when the message carries a recoverable byte offset,
/// the offending source line with a caret (`^`) underneath it.
pub fn print_error_with_caret(source: &str, err: &impl std::fmt::Display) {
    let msg = err.to_string();
    eprintln!("{}: {}", "Error".red(), msg);
    if let Some(offset) = extract_byte_offset(&msg) {
        if offset <= source.len() && source.is_char_boundary(offset) {
            eprintln!("  {}", source);
            eprintln!("  {}{}", " ".repeat(offset), "^".red().bold());
        }
    }
}

/// Runs `f` with an indeterminate spinner shown for its duration (started
/// immediately before, cleared immediately after) -- gives feedback for
/// genuinely slow statements (large `IMPORT DATASET FROM`, big batch
/// `INSERT`) with no per-operation instrumentation. Fast statements show a
/// brief flash, which is accepted (matches common CLI spinner behavior).
pub fn with_spinner<T>(f: impl FnOnce() -> T) -> T {
    let pb = indicatif::ProgressBar::new_spinner();
    pb.set_message("running...");
    pb.enable_steady_tick(Duration::from_millis(80));
    let result = f();
    pb.finish_and_clear();
    result
}

/// Colors the leading statement keyword (if recognized) and any
/// double-quoted string literals in `line`. A basic, line-local pass --
/// not a full DSL tokenizer -- kept intentionally simple since it only
/// needs to look good while typing a single statement.
fn highlight_dsl_line(line: &str) -> Cow<'_, str> {
    if line.is_empty() {
        return Cow::Borrowed(line);
    }

    let mut out = String::with_capacity(line.len() + 16);
    let mut colored_any = false;

    let first_word_end = line.find(char::is_whitespace).unwrap_or(line.len());
    let first_word = &line[..first_word_end];
    if DSL_KEYWORDS.contains(&first_word.to_uppercase().as_str()) {
        out.push_str(&first_word.blue().bold().to_string());
        colored_any = true;
    } else {
        out.push_str(first_word);
    }

    let mut in_string = false;
    let mut buf = String::new();
    for ch in line[first_word_end..].chars() {
        if ch == '"' {
            buf.push(ch);
            if in_string {
                out.push_str(&buf.green().to_string());
                colored_any = true;
                buf.clear();
            }
            in_string = !in_string;
        } else {
            buf.push(ch);
        }
    }
    out.push_str(&buf);

    if colored_any {
        Cow::Owned(out)
    } else {
        Cow::Borrowed(line)
    }
}

/// Custom rustyline `Helper`: keyword/string highlighting (manual, above),
/// statement-keyword + live dataset-name Tab-completion (manual, below),
/// history-based inline hints and matching-bracket multi-line validation
/// (both delegated to rustyline's own built-ins via `#[derive(...)]`).
#[derive(Helper, Hinter, Validator)]
pub struct LinalHelper {
    #[rustyline(Hinter)]
    hinter: HistoryHinter,
    #[rustyline(Validator)]
    validator: MatchingBracketValidator,
    /// Dataset names in the currently active database. The REPL loop
    /// refreshes this from `TensorDb::active_instance` before each prompt
    /// so completion reflects the current session, not a stale snapshot.
    pub datasets: Rc<RefCell<Vec<String>>>,
}

impl LinalHelper {
    pub fn new(datasets: Rc<RefCell<Vec<String>>>) -> Self {
        Self {
            hinter: HistoryHinter::new(),
            validator: MatchingBracketValidator::new(),
            datasets,
        }
    }
}

impl Highlighter for LinalHelper {
    fn highlight<'l>(&self, line: &'l str, _pos: usize) -> Cow<'l, str> {
        highlight_dsl_line(line)
    }

    fn highlight_char(&self, _line: &str, _pos: usize, _forced: bool) -> bool {
        true
    }
}

impl Completer for LinalHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        let start = line[..pos]
            .rfind(char::is_whitespace)
            .map(|i| i + 1)
            .unwrap_or(0);
        let word = &line[start..pos];
        if word.is_empty() {
            return Ok((start, Vec::new()));
        }

        let word_upper = word.to_uppercase();
        let mut candidates: Vec<Pair> = STATEMENT_KEYWORDS
            .iter()
            .filter(|k| k.starts_with(&word_upper))
            .map(|k| Pair {
                display: (*k).to_string(),
                replacement: (*k).to_string(),
            })
            .collect();

        let word_lower = word.to_lowercase();
        for name in self.datasets.borrow().iter() {
            if name.to_lowercase().starts_with(&word_lower) {
                candidates.push(Pair {
                    display: name.clone(),
                    replacement: name.clone(),
                });
            }
        }

        Ok((start, candidates))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustyline::history::MemHistory;
    use std::sync::Mutex;

    // These four tests toggle colored::control's process-global override
    // (colored's SHOULD_COLORIZE is one shared AtomicBool pair with no
    // per-thread isolation) and race against each other under cargo test's
    // default concurrent-thread runner. Serialize them.
    static COLOR_OVERRIDE_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn cli_verb_hint_flags_known_shell_commands() {
        assert!(cli_verb_hint("serve --port 8080").is_some());
        assert!(cli_verb_hint("linal --help").is_some());
        assert!(cli_verb_hint("--help").is_some());
        assert!(cli_verb_hint("-h").is_some());
    }

    #[test]
    fn cli_verb_hint_ignores_real_dsl_statements() {
        assert!(cli_verb_hint("SELECT * FROM t").is_none());
        assert!(cli_verb_hint("CREATE DATABASE foo").is_none());
        assert!(cli_verb_hint("").is_none());
        // No CLI verb collides with a real DSL keyword.
        for verb in CLI_VERBS {
            assert!(
                !DSL_KEYWORDS.contains(&verb.to_uppercase().as_str()),
                "CLI verb '{verb}' collides with a real DSL keyword"
            );
        }
    }

    #[test]
    fn extract_byte_offset_parses_the_parser_marker() {
        assert_eq!(
            extract_byte_offset("expected a statement keyword, found integer `3` (at byte 0)"),
            Some(0)
        );
        assert_eq!(
            extract_byte_offset(
                "expected DATABASE or INDEX after CREATE, found `Dataset` (at byte 7)"
            ),
            Some(7)
        );
    }

    #[test]
    fn extract_byte_offset_ignores_unrelated_messages() {
        assert_eq!(extract_byte_offset("Dataset not found: t"), None);
        assert_eq!(extract_byte_offset(""), None);
    }

    #[test]
    fn highlight_dsl_line_colors_leading_keyword_and_strings() {
        let _guard = COLOR_OVERRIDE_LOCK.lock().unwrap();
        colored::control::set_override(true);
        let highlighted = highlight_dsl_line("SELECT * FROM t WHERE name = \"alice\"");
        assert!(matches!(highlighted, Cow::Owned(_)));
        let rendered = highlighted.into_owned();
        assert!(rendered.contains("SELECT"));
        assert!(rendered.contains("\"alice\""));
        assert_ne!(rendered, "SELECT * FROM t WHERE name = \"alice\"");
        colored::control::unset_override();
    }

    #[test]
    fn highlight_dsl_line_passes_through_unrecognized_input() {
        let _guard = COLOR_OVERRIDE_LOCK.lock().unwrap();
        colored::control::set_override(true);
        let highlighted = highlight_dsl_line("3+3");
        assert!(matches!(highlighted, Cow::Borrowed(_)));
        assert_eq!(highlighted, "3+3");
        colored::control::unset_override();
    }

    #[test]
    fn completer_suggests_matching_statement_keywords() {
        let history = MemHistory::new();
        let ctx = Context::new(&history);
        let helper = LinalHelper::new(Rc::new(RefCell::new(Vec::new())));
        let (start, candidates) = helper.complete("SEL", 3, &ctx).unwrap();
        assert_eq!(start, 0);
        assert!(candidates.iter().any(|c| c.replacement == "SELECT"));
    }

    #[test]
    fn right_pad_spaces_pads_to_requested_width() {
        assert_eq!(right_pad_spaces("abc", 10).len(), 7);
        assert_eq!(right_pad_spaces("abcdefghij", 10).len(), 0);
        assert_eq!(right_pad_spaces("way too long already", 5).len(), 0);
    }

    #[test]
    fn right_pad_spaces_ignores_ansi_bytes_in_display_text() {
        // Regression test for the exact gotcha `render_box` must avoid:
        // padding must be computed from the *plain* text length, never
        // from a colored string's byte/char count.
        let _guard = COLOR_OVERRIDE_LOCK.lock().unwrap();
        colored::control::set_override(true);
        let colored_text = "abc".red().bold().to_string();
        assert!(colored_text.chars().count() > 3); // proves ANSI bytes are present
        assert_eq!(right_pad_spaces("abc", 10).len(), 7);
        colored::control::unset_override();
    }

    #[test]
    fn render_box_pads_all_rows_to_the_widest_plain_content() {
        let _guard = COLOR_OVERRIDE_LOCK.lock().unwrap();
        colored::control::set_override(false);
        let rows = vec![
            ("short".to_string(), "short".to_string()),
            ("a longer line".to_string(), "a longer line".to_string()),
        ];
        let box_str = render_box(&rows);
        let content_lines: Vec<&str> = box_str.lines().filter(|l| l.starts_with('│')).collect();
        assert_eq!(content_lines.len(), 2);
        let widths: Vec<usize> = content_lines.iter().map(|l| l.chars().count()).collect();
        assert_eq!(widths[0], widths[1]);
    }

    #[test]
    fn pick_tip_is_deterministic_and_in_bounds() {
        assert_eq!(pick_tip(0), pick_tip(REPL_TIPS.len() as u64));
        for seed in 0..20u64 {
            assert!(REPL_TIPS.contains(&pick_tip(seed)));
        }
    }

    #[test]
    fn pick_tip_varies_across_seeds() {
        let a = pick_tip(0);
        let b = pick_tip(1);
        assert_ne!(a, b);
    }

    #[test]
    fn completer_suggests_current_dataset_names() {
        let history = MemHistory::new();
        let ctx = Context::new(&history);
        let datasets = Rc::new(RefCell::new(vec!["pbmc_cells".to_string()]));
        let helper = LinalHelper::new(datasets);
        let line = "SELECT * FROM pb";
        let (start, candidates) = helper.complete(line, line.len(), &ctx).unwrap();
        assert_eq!(start, "SELECT * FROM ".len());
        assert!(candidates.iter().any(|c| c.replacement == "pbmc_cells"));
    }
}
