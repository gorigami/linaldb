//! Splits a multi-statement `.lnl` script into individual statements, exactly
//! matching the CLI's `linal run` behavior: a statement can span multiple
//! lines and only ends once its parenthesis count balances back to zero.
//! Comment (`#`/`--`/`//`) and blank lines are skipped only *between*
//! statements -- one inside an already-open parenthesis is appended into the
//! statement text as-is, matching `linal run` exactly.
//!
//! This used to be duplicated inline in `main.rs`'s `Run` command; it's
//! extracted here so the server's `POST /execute/batch` endpoint can share
//! the identical algorithm instead of risking a second, subtly different
//! reimplementation (see CHANGELOG for the bug class this guards against:
//! `linal-hub`'s playground had to reimplement this same joiner in Python
//! client-side because no server-side batch endpoint existed yet).

/// One statement recovered from a script, with the 1-based source line it
/// started on (for error messages).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScriptStatement {
    pub text: String,
    pub start_line: usize,
}

/// A script ended with an unclosed parenthesis -- `start_line` is where the
/// unterminated statement began.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnbalancedParensError {
    pub start_line: usize,
}

impl std::fmt::Display for UnbalancedParensError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Script ended with unbalanced parentheses starting at line {}",
            self.start_line
        )
    }
}

impl std::error::Error for UnbalancedParensError {}

/// Split `content` into individual statements, joining lines whose
/// parentheses haven't yet balanced. See module docs for the exact rules.
pub fn split_script(content: &str) -> Result<Vec<ScriptStatement>, UnbalancedParensError> {
    let mut statements = Vec::new();
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
            statements.push(ScriptStatement {
                text: std::mem::take(&mut current_cmd),
                start_line,
            });
        }
    }

    if !current_cmd.is_empty() {
        return Err(UnbalancedParensError { start_line });
    }

    Ok(statements)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_line_statements() {
        let stmts = split_script("VECTOR v = [1,2,3]\nSHOW v").unwrap();
        assert_eq!(stmts.len(), 2);
        assert_eq!(stmts[0].text, "VECTOR v = [1,2,3]");
        assert_eq!(stmts[0].start_line, 1);
        assert_eq!(stmts[1].text, "SHOW v");
        assert_eq!(stmts[1].start_line, 2);
    }

    #[test]
    fn multi_line_statement_joins_on_paren_balance() {
        let script = "DATASET users COLUMNS (\n    id: INT,\n    age: INT\n)\nSHOW SCHEMA users";
        let stmts = split_script(script).unwrap();
        assert_eq!(stmts.len(), 2);
        assert_eq!(stmts[0].text, "DATASET users COLUMNS ( id: INT, age: INT )");
        assert_eq!(stmts[0].start_line, 1);
        assert_eq!(stmts[1].text, "SHOW SCHEMA users");
        assert_eq!(stmts[1].start_line, 5);
    }

    #[test]
    fn comments_and_blank_lines_skipped_between_statements() {
        let script = "// header comment\n\n# also a comment\nSHOW ALL DATASETS";
        let stmts = split_script(script).unwrap();
        assert_eq!(stmts.len(), 1);
        assert_eq!(stmts[0].text, "SHOW ALL DATASETS");
    }

    #[test]
    fn comment_inside_open_paren_is_not_skipped() {
        // Matches `linal run`'s own behavior: the comment-skip only applies
        // between statements, so a `//` line while a paren is still open
        // gets appended into the statement text verbatim.
        let script = "MATRIX m = (\n// not a comment here\n1, 2)";
        let stmts = split_script(script).unwrap();
        assert_eq!(stmts.len(), 1);
        assert_eq!(stmts[0].text, "MATRIX m = ( // not a comment here 1, 2)");
    }

    #[test]
    fn unbalanced_parens_errors_with_start_line() {
        let err = split_script("DATASET users COLUMNS (\n    id: INT").unwrap_err();
        assert_eq!(err.start_line, 1);
    }

    #[test]
    fn empty_script_yields_no_statements() {
        assert_eq!(split_script("\n\n// only comments\n").unwrap(), vec![]);
    }
}
