// tests/parser_error_surfacing_test.rs
// Regression tests for a DX gap flagged during the round-2 consistency audit
// (CONSISTENCY_PLAN.md, historical — deleted once round 2 closed out): the
// parser's structured ParseError (byte offset + expectation detail) was
// silently discarded on failure in execute_line_with_context, replaced with
// a generic "Unknown command: <raw line>" message no matter what actually
// went wrong. Fixed by matching on the parse Result instead of `if let Ok`
// and converting the real ParseError via into_dsl_error (which now also
// preserves the byte offset, previously dropped there too).
//
// Fixing this also surfaced a second bug in the same function: pure `--`
// comment lines errored out as "Unknown command" when reached directly via
// execute_line (REPL, server /execute) — the blank/comment fallback only
// checked `#`/`//`, not `--`, even though the lexer treats all three as
// comment styles and execute_script's own pre-filter already checked all
// three. Fixed by checking `--` too.

use linal::dsl::{execute_line, DslError, DslOutput};
use linal::engine::TensorDb;

#[test]
fn test_malformed_statement_surfaces_real_parser_error() {
    let mut db = TensorDb::new();
    let result = execute_line(&mut db, "GET * FROM users", 1);
    match result {
        Err(DslError::Parse { line, msg }) => {
            assert_eq!(line, 1);
            assert!(
                !msg.starts_with("Unknown command"),
                "expected the real parser error, not the old generic fallback, got: {msg}"
            );
            assert!(
                msg.contains("GET") || msg.contains("statement"),
                "expected the error to reference what actually went wrong, got: {msg}"
            );
            assert!(
                msg.contains("at byte"),
                "expected the byte offset to survive the conversion into DslError::Parse, got: {msg}"
            );
        }
        other => panic!("Expected DslError::Parse, got: {other:?}"),
    }
}

#[test]
fn test_old_paren_tensor_syntax_surfaces_specific_error() {
    let mut db = TensorDb::new();
    let result = execute_line(&mut db, "DEFINE t AS TENSOR(2,2) VALUES [1,2,3,4]", 1);
    match result {
        Err(DslError::Parse { msg, .. }) => {
            assert!(
                !msg.starts_with("Unknown command"),
                "expected a specific parse error, not the old generic fallback, got: {msg}"
            );
        }
        other => panic!("Expected DslError::Parse, got: {other:?}"),
    }
}

#[test]
fn test_genuinely_unparseable_line_still_errors() {
    // Guard against over-broad fixes: a truly invalid line must still error,
    // just with a better message than before.
    let mut db = TensorDb::new();
    let result = execute_line(&mut db, "@@@ not valid dsl @@@", 1);
    assert!(result.is_err(), "invalid input must still be an error");
}

// ── Comment-only lines must be a no-op via execute_line directly (REPL,
//    server /execute), not just through execute_script's own pre-filter ──

#[test]
fn test_double_dash_comment_line_is_noop_via_execute_line() {
    let mut db = TensorDb::new();
    let result = execute_line(&mut db, "-- pure comment", 1);
    match result {
        Ok(DslOutput::None) => {}
        other => panic!("Expected Ok(DslOutput::None) for a `--` comment line, got: {other:?}"),
    }
}

#[test]
fn test_hash_comment_line_is_noop_via_execute_line() {
    let mut db = TensorDb::new();
    let result = execute_line(&mut db, "# pure comment", 1);
    match result {
        Ok(DslOutput::None) => {}
        other => panic!("Expected Ok(DslOutput::None) for a `#` comment line, got: {other:?}"),
    }
}

#[test]
fn test_slash_slash_comment_line_is_noop_via_execute_line() {
    let mut db = TensorDb::new();
    let result = execute_line(&mut db, "// pure comment", 1);
    match result {
        Ok(DslOutput::None) => {}
        other => panic!("Expected Ok(DslOutput::None) for a `//` comment line, got: {other:?}"),
    }
}

#[test]
fn test_blank_line_is_noop_via_execute_line() {
    let mut db = TensorDb::new();
    let result = execute_line(&mut db, "   ", 1);
    match result {
        Ok(DslOutput::None) => {}
        other => panic!("Expected Ok(DslOutput::None) for a blank line, got: {other:?}"),
    }
}

#[test]
fn test_valid_statement_after_fix_still_executes_normally() {
    // Guard against over-broad fixes: real statements must be unaffected.
    let mut db = TensorDb::new();
    let out = execute_line(&mut db, "VECTOR v = [1.0, 2.0, 3.0]", 1)
        .unwrap_or_else(|e| panic!("Unexpected error: {e:?}"));
    match out {
        DslOutput::Message(_) => {}
        other => panic!("Expected Ok(Message), got: {other:?}"),
    }
}
