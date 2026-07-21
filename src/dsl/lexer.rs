use logos::{Logos, Span};

/// All tokens produced by the LINALDB DSL lexer.
///
/// Keywords are uppercase and case-sensitive, matching the existing DSL convention.
/// Whitespace and comment styles (`--`, `#`, `//`) are skipped automatically.
/// Identifiers have lowest priority — any uppercase word that matches a keyword
/// token first is lexed as that keyword, not as an `Ident`.
#[derive(Logos, Debug, Clone, PartialEq)]
#[logos(skip r"[ \t\r\n]+")]
#[logos(skip r"--[^\n]*")]
#[logos(skip r"#[^\n]*")]
#[logos(skip r"//[^\n]*")]
pub enum Token {
    // ─── Top-level command keywords ──────────────────────────────────────────
    #[token("DEFINE")]
    Define,
    #[token("VECTOR")]
    Vector,
    #[token("MATRIX")]
    Matrix,
    #[token("LET")]
    Let,
    #[token("LAZY")]
    Lazy,
    #[token("SHOW")]
    Show,
    #[token("SELECT")]
    Select,
    #[token("DELIVER")]
    Deliver,
    #[token("BIND")]
    Bind,
    #[token("ATTACH")]
    Attach,
    #[token("DERIVE")]
    Derive,
    #[token("DATASET")]
    Dataset,
    #[token("INSERT")]
    Insert,
    #[token("SEARCH")]
    Search,
    #[token("EXPLAIN")]
    Explain,
    #[token("AUDIT")]
    Audit,
    #[token("MATERIALIZE")]
    Materialize,
    #[token("CREATE")]
    Create,
    #[token("ALTER")]
    Alter,
    #[token("USE")]
    Use,
    #[token("DROP")]
    Drop,
    #[token("SET")]
    Set,
    #[token("SAVE")]
    Save,
    #[token("LOAD")]
    Load,
    #[token("LIST")]
    List,
    #[token("IMPORT")]
    Import,
    #[token("EXPORT")]
    Export,
    #[token("RESET")]
    Reset,
    #[token("TRANSFORM")]
    Transform,
    #[token("UPDATE")]
    Update,
    #[token("DELETE")]
    Delete,
    #[token("JOIN")]
    Join,
    #[token("ON")]
    On,
    #[token("INNER")]
    Inner,
    #[token("LEFT")]
    Left,
    #[token("RIGHT")]
    Right,
    #[token("FULL")]
    Full,
    #[token("OUTER")]
    Outer,
    #[token("OFFSET")]
    Offset,
    #[token("IN")]
    In,
    #[token("BETWEEN")]
    Between,
    #[token("UNION")]
    Union,
    #[token("DISTINCT")]
    Distinct,
    #[token("OVER")]
    Over,
    #[token("PARTITION")]
    Partition,
    #[token("CASE")]
    Case,
    #[token("WHEN")]
    When,
    #[token("THEN")]
    Then,
    #[token("ELSE")]
    Else,
    #[token("END")]
    End,
    #[token("PIPELINE")]
    Pipeline,
    #[token("PIPELINES")]
    Pipelines,
    #[token("APPLY")]
    Apply,
    #[token("DESCRIBE")]
    Describe,

    // ─── Structural / sub-command keywords ───────────────────────────────────
    #[token("AS")]
    As,
    #[token("STRICT")]
    Strict,
    #[token("TENSOR")]
    Tensor,
    #[token("VALUES")]
    Values,
    #[token("TO")]
    To,
    #[token("FROM")]
    From,
    #[token("WITH")]
    With,
    #[token("BY")]
    By,
    #[token("INTO")]
    Into,
    #[token("DATABASE")]
    Database,
    #[token("INDEX")]
    Index,
    #[token("COLUMNS")]
    Columns,
    #[token("COLUMN")]
    Column,
    #[token("ADD")]
    Add,
    #[token("ALL")]
    All,
    #[token("TENSORS")]
    Tensors,
    #[token("DATASETS")]
    Datasets,
    #[token("DATABASES")]
    Databases,
    #[token("INDEXES")]
    Indexes,
    #[token("SCHEMA")]
    Schema,
    #[token("SHAPE")]
    Shape,
    #[token("LINEAGE")]
    Lineage,
    #[token("METADATA")]
    Metadata,
    #[token("VERSIONS")]
    Versions,
    #[token("WHERE")]
    Where,
    #[token("FILTER")]
    Filter,
    #[token("GROUP")]
    Group,
    #[token("HAVING")]
    Having,
    #[token("ORDER")]
    Order,
    #[token("LIMIT")]
    Limit,
    #[token("NULL")]
    Null,
    #[token("NOT")]
    Not,
    #[token("IS")]
    Is,
    #[token("AND")]
    And,
    #[token("OR")]
    Or,
    #[token("NULLABLE")]
    Nullable,

    // ─── Operation keywords (within LET / DERIVE expressions) ─────────────────
    #[token("SUBTRACT")]
    Subtract,
    #[token("MULTIPLY")]
    Multiply,
    #[token("DIVIDE")]
    Divide,
    #[token("CORRELATE")]
    Correlate,
    #[token("SIMILARITY")]
    Similarity,
    #[token("DISTANCE")]
    Distance,
    #[token("MATMUL")]
    Matmul,
    #[token("TRANSPOSE")]
    Transpose,
    #[token("RESHAPE")]
    Reshape,
    #[token("STACK")]
    Stack,
    #[token("SCALE")]
    Scale,
    #[token("NORMALIZE")]
    Normalize,
    #[token("FLATTEN")]
    Flatten,
    #[token("SUM")]
    Sum,
    #[token("MEAN")]
    Mean,
    #[token("STDEV")]
    Stdev,
    #[token("FFT")]
    Fft,
    #[token("IFFT")]
    Ifft,

    // ─── Punctuation & operators ──────────────────────────────────────────────
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token(",")]
    Comma,
    #[token(":")]
    Colon,
    #[token("=")]
    Eq,
    #[token(".")]
    Dot,
    #[token(">=")]
    GtEq,
    #[token("<=")]
    LtEq,
    #[token("!=")]
    NotEq,
    #[token("~=")]
    ApproxEq,
    #[token(">")]
    Gt,
    #[token("<")]
    Lt,
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("?")]
    Question,

    // ─── Literals ─────────────────────────────────────────────────────────────
    /// Float literal: `3.14`, `1.0e-5`. Requires a leading digit (`0.5`, not `.5`).
    #[regex(r"[0-9]+\.[0-9]+([eE][+-]?[0-9]+)?", |lex| lex.slice().parse::<f64>().ok())]
    Float(f64),

    /// Integer literal: `42`, `0`, `1024`.
    #[regex(r"[0-9]+", |lex| lex.slice().parse::<i64>().ok())]
    Int(i64),

    /// Double- or single-quoted string literal: `"hello"` / `'hello'`.
    #[regex(r#""[^"]*""#, |lex| {
        let s = lex.slice();
        Some(s[1..s.len() - 1].to_string())
    })]
    #[regex(r"'[^']*'", |lex| {
        let s = lex.slice();
        Some(s[1..s.len() - 1].to_string())
    })]
    Str(String),

    /// Identifier: tensor/dataset/variable names. Lowest priority — matched only
    /// when no keyword applies.
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*", |lex| lex.slice().to_string())]
    Ident(String),
}

impl std::fmt::Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Token::Ident(s) => write!(f, "identifier `{}`", s),
            Token::Str(s) => write!(f, "string `\"{}\"`", s),
            Token::Int(n) => write!(f, "integer `{}`", n),
            Token::Float(n) => write!(f, "float `{}`", n),
            Token::Eq => write!(f, "`=`"),
            Token::Comma => write!(f, "`,`"),
            Token::Colon => write!(f, "`:`"),
            Token::Dot => write!(f, "`.`"),
            Token::LBracket => write!(f, "`[`"),
            Token::RBracket => write!(f, "`]`"),
            Token::LParen => write!(f, "`(`"),
            Token::RParen => write!(f, "`)`"),
            Token::GtEq => write!(f, "`>=`"),
            Token::LtEq => write!(f, "`<=`"),
            Token::NotEq => write!(f, "`!=`"),
            Token::Gt => write!(f, "`>`"),
            Token::Lt => write!(f, "`<`"),
            Token::Plus => write!(f, "`+`"),
            Token::Minus => write!(f, "`-`"),
            Token::Star => write!(f, "`*`"),
            Token::Slash => write!(f, "`/`"),
            other => {
                // For keyword tokens use the Debug name
                write!(f, "`{:?}`", other)
            }
        }
    }
}

/// A token with its byte-range span in the original source string.
pub type Spanned = (Token, Span);

/// Tokenize `source` into a list of `(Token, Span)` pairs.
///
/// Returns `Err(byte_offset)` at the first character that does not match any
/// token pattern.
pub fn tokenize(source: &str) -> Result<Vec<Spanned>, usize> {
    let mut lexer = Token::lexer(source);
    let mut tokens = Vec::new();
    while let Some(result) = lexer.next() {
        match result {
            Ok(tok) => tokens.push((tok, lexer.span())),
            Err(_) => return Err(lexer.span().start),
        }
    }
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(src: &str) -> Vec<Token> {
        tokenize(src).unwrap().into_iter().map(|(t, _)| t).collect()
    }

    #[test]
    fn keywords_take_priority_over_ident() {
        assert_eq!(tok("VECTOR"), vec![Token::Vector]);
        assert_eq!(tok("LET"), vec![Token::Let]);
        assert_eq!(tok("LAZY"), vec![Token::Lazy]);
        assert_eq!(tok("STRICT"), vec![Token::Strict]);
        assert_eq!(tok("ADD"), vec![Token::Add]);
    }

    #[test]
    fn identifiers_for_unknown_names() {
        assert_eq!(tok("my_tensor"), vec![Token::Ident("my_tensor".into())]);
        assert_eq!(tok("v1"), vec![Token::Ident("v1".into())]);
    }

    #[test]
    fn numeric_literals() {
        assert_eq!(tok("42"), vec![Token::Int(42)]);
        assert_eq!(tok("3.14"), vec![Token::Float(3.14)]);
    }

    #[test]
    fn string_literal() {
        assert_eq!(tok("\"hello\""), vec![Token::Str("hello".into())]);
    }

    #[test]
    fn comments_are_skipped() {
        assert_eq!(
            tok("LET -- this is a comment\nx = ADD a b"),
            tok("LET x = ADD a b")
        );
        assert_eq!(
            tok("# hash comment\nLET x = ADD a b"),
            tok("LET x = ADD a b")
        );
    }

    #[test]
    fn simple_let_statement() {
        let tokens = tok("LET result = ADD a b");
        assert_eq!(
            tokens,
            vec![
                Token::Let,
                Token::Ident("result".into()),
                Token::Eq,
                Token::Add,
                Token::Ident("a".into()),
                Token::Ident("b".into()),
            ]
        );
    }

    #[test]
    fn infix_expression() {
        let tokens = tok("LET c = a + b");
        assert_eq!(
            tokens,
            vec![
                Token::Let,
                Token::Ident("c".into()),
                Token::Eq,
                Token::Ident("a".into()),
                Token::Plus,
                Token::Ident("b".into()),
            ]
        );
    }

    #[test]
    fn lazy_let_variants() {
        assert_eq!(tok("LAZY LET")[..2], [Token::Lazy, Token::Let]);
        assert_eq!(tok("LET LAZY")[..2], [Token::Let, Token::Lazy]);
    }

    #[test]
    fn show_schema_statement() {
        let tokens = tok("SHOW SCHEMA my_dataset");
        assert_eq!(
            tokens,
            vec![
                Token::Show,
                Token::Schema,
                Token::Ident("my_dataset".into()),
            ]
        );
    }

    #[test]
    fn unknown_char_returns_error() {
        assert!(tokenize("LET x = @bad").is_err());
    }
}
