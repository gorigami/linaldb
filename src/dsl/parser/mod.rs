use logos::Span;

use crate::dsl::ast::*;
use crate::dsl::lexer::{tokenize, Token};

mod dataset;
mod expr;
mod introspection;
mod persistence;

// ─── Error ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ParseError {
    pub offset: usize,
    pub msg: String,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (at byte {})", self.msg, self.offset)
    }
}

impl ParseError {
    pub fn into_dsl_error(self, line: usize) -> crate::dsl::DslError {
        crate::dsl::DslError::Parse {
            line,
            // Uses Display (msg + byte offset), not just self.msg, so the
            // offset survives the conversion into DslError::Parse's plain
            // String — DslError::Parse has no dedicated offset field, so
            // this is the only place that information can travel.
            msg: self.to_string(),
        }
    }
}

// ─── Parser ───────────────────────────────────────────────────────────────────

struct Parser {
    tokens: Vec<(Token, Span)>,
    pos: usize,
    end: usize,
}

impl Parser {
    fn new(tokens: Vec<(Token, Span)>, source_len: usize) -> Self {
        Self {
            tokens,
            pos: 0,
            end: source_len,
        }
    }

    // ── Cursor primitives ────────────────────────────────────────────────────

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos).map(|(t, _)| t)
    }

    fn peek_at(&self, offset: usize) -> Option<&Token> {
        self.tokens.get(self.pos + offset).map(|(t, _)| t)
    }

    fn current_span(&self) -> Span {
        self.tokens
            .get(self.pos)
            .map(|(_, s)| s.clone())
            .unwrap_or(self.end..self.end)
    }

    fn advance(&mut self) -> Option<Token> {
        if self.pos < self.tokens.len() {
            let (tok, _) = self.tokens[self.pos].clone();
            self.pos += 1;
            Some(tok)
        } else {
            None
        }
    }

    fn at(&self, expected: &Token) -> bool {
        self.peek()
            .map(|t| same_variant(t, expected))
            .unwrap_or(false)
    }

    fn at_ident(&self, name: &str) -> bool {
        matches!(self.peek(), Some(Token::Ident(s)) if s == name)
    }

    fn at_any_ident(&self) -> bool {
        matches!(self.peek(), Some(Token::Ident(_)))
    }

    fn eof(&self) -> bool {
        self.pos >= self.tokens.len()
    }

    // ── Consuming helpers ────────────────────────────────────────────────────

    fn eat(&mut self, expected: &Token) -> Result<Span, ParseError> {
        if self.at(expected) {
            let span = self.current_span();
            self.pos += 1;
            Ok(span)
        } else {
            Err(self.unexpected(&format!("{}", expected)))
        }
    }

    fn eat_ident(&mut self) -> Result<String, ParseError> {
        match self.advance_if_ident() {
            Some(s) => Ok(s),
            None => Err(self.unexpected("identifier")),
        }
    }

    fn advance_if_ident(&mut self) -> Option<String> {
        if matches!(self.peek(), Some(Token::Ident(_))) {
            if let Some(Token::Ident(s)) = self.advance() {
                return Some(s);
            }
        }
        None
    }

    fn eat_str(&mut self) -> Result<String, ParseError> {
        if matches!(self.peek(), Some(Token::Str(_))) {
            if let Some(Token::Str(s)) = self.advance() {
                return Ok(s);
            }
        }
        Err(self.unexpected("string literal"))
    }

    fn eat_int(&mut self) -> Result<i64, ParseError> {
        if matches!(self.peek(), Some(Token::Int(_))) {
            if let Some(Token::Int(n)) = self.advance() {
                return Ok(n);
            }
        }
        Err(self.unexpected("integer literal"))
    }

    fn eat_number(&mut self) -> Result<f64, ParseError> {
        match self.peek() {
            Some(Token::Float(_)) => {
                if let Some(Token::Float(f)) = self.advance() {
                    return Ok(f);
                }
                unreachable!()
            }
            Some(Token::Int(_)) => {
                if let Some(Token::Int(n)) = self.advance() {
                    return Ok(n as f64);
                }
                unreachable!()
            }
            Some(Token::Minus) => {
                self.advance();
                Ok(-self.eat_number()?)
            }
            _ => Err(self.unexpected("numeric literal")),
        }
    }

    fn eat_usize(&mut self) -> Result<usize, ParseError> {
        let n = self.eat_int()?;
        n.try_into()
            .map_err(|_| self.error(format!("expected non-negative integer, got {}", n)))
    }

    // ── Error helpers ────────────────────────────────────────────────────────

    fn error(&self, msg: impl Into<String>) -> ParseError {
        ParseError {
            offset: self.current_span().start,
            msg: msg.into(),
        }
    }

    fn unexpected(&self, expected: &str) -> ParseError {
        let found = match self.peek() {
            Some(t) => format!("{}", t),
            None => "end of input".to_string(),
        };
        self.error(format!("expected {}, found {}", expected, found))
    }
}

fn same_variant(a: &Token, b: &Token) -> bool {
    std::mem::discriminant(a) == std::mem::discriminant(b)
}

// ─── Public API ───────────────────────────────────────────────────────────────

pub fn parse(source: &str) -> Result<Statement, ParseError> {
    let tokens = tokenize(source).map_err(|offset| ParseError {
        offset,
        msg: format!("unrecognised character at byte {}", offset),
    })?;

    let mut p = Parser::new(tokens, source.len());

    if p.eof() {
        return Err(ParseError {
            offset: 0,
            msg: "empty input".to_string(),
        });
    }

    let stmt = p.parse_statement()?;

    if !p.eof() {
        return Err(p.error(format!(
            "unexpected token after statement: {}",
            p.peek().map(|t| format!("{}", t)).unwrap_or_default()
        )));
    }

    Ok(stmt)
}

// ─── Statement dispatch ───────────────────────────────────────────────────────

impl Parser {
    fn parse_statement(&mut self) -> Result<Statement, ParseError> {
        match self.peek() {
            Some(Token::Define) => self.parse_define_tensor(),
            Some(Token::Vector) => self.parse_vector(),
            Some(Token::Matrix) => self.parse_matrix(),
            Some(Token::Let) | Some(Token::Lazy) => self.parse_let(),
            Some(Token::Derive) => self.parse_derive(),
            Some(Token::Show) => self.parse_show(),
            Some(Token::Bind) => self.parse_bind(),
            Some(Token::Attach) => self.parse_attach(),
            Some(Token::Dataset) => self.parse_create_dataset(),
            Some(Token::Insert) => self.parse_insert_into(),
            Some(Token::Select) => self.parse_select(),
            Some(Token::With) => self.parse_cte_select(),
            Some(Token::Materialize) => self.parse_materialize(),
            Some(Token::Deliver) => self.parse_deliver(),
            Some(Token::Explain) => self.parse_explain(),
            Some(Token::Audit) => self.parse_audit(),
            Some(Token::Save) => self.parse_save(),
            Some(Token::Load) => self.parse_load(),
            Some(Token::List) => self.parse_list(),
            Some(Token::Import) => self.parse_import(),
            Some(Token::Export) => self.parse_export(),
            Some(Token::Create) => self.parse_create(),
            Some(Token::Alter) => self.parse_alter(),
            Some(Token::Drop) => self.parse_drop(),
            Some(Token::Use) => self.parse_use(),
            Some(Token::Set) => self.parse_set(),
            Some(Token::Search) => self.parse_search(),
            Some(Token::Update) => self.parse_update(),
            Some(Token::Delete) => self.parse_delete(),
            Some(Token::Transform) => self.parse_transform(),
            Some(Token::Apply) => self.parse_apply_pipeline(),
            Some(Token::Describe) => self.parse_describe_pipeline(),
            Some(Token::Reset) => {
                self.advance();
                if self.at_ident("SESSION") {
                    self.advance();
                }
                Ok(Statement::Reset)
            }
            Some(Token::Ident(_)) if self.peek_at(1) == Some(&Token::Dot) => {
                self.parse_method_call()
            }
            _ => Err(self.unexpected("a statement keyword")),
        }
    }
}

// ─── Small statement parsers ──────────────────────────────────────────────────

impl Parser {
    // DEFINE PIPELINE <name> AS step [THEN step ...]
    // DEFINE <name> AS [STRICT] TENSOR [dims] VALUES [values]
    fn parse_define_tensor(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Define)?;
        // Check for DEFINE PIPELINE
        if self.at(&Token::Pipeline) {
            return self.parse_define_pipeline();
        }
        let name = self.eat_ident()?;
        self.eat(&Token::As)?;

        let kind = if self.at(&Token::Strict) {
            self.advance();
            self.eat(&Token::Tensor)?;
            TensorKindAst::Strict
        } else {
            self.eat(&Token::Tensor)?;
            TensorKindAst::Normal
        };

        let shape = self.parse_usize_list()?;
        self.eat(&Token::Values)?;
        let values = self.parse_f64_list()?;

        Ok(Statement::DefineTensor(DefineTensorStmt {
            name,
            kind,
            shape,
            values,
        }))
    }

    // VECTOR <name> = [values]
    fn parse_vector(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Vector)?;
        let name = self.eat_ident()?;
        self.eat(&Token::Eq)?;
        let values = self.parse_f64_list()?;
        Ok(Statement::Vector(VectorStmt { name, values }))
    }

    // MATRIX <name> = [[r0], [r1], ...]
    fn parse_matrix(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Matrix)?;
        let name = self.eat_ident()?;
        self.eat(&Token::Eq)?;

        self.eat(&Token::LBracket)?;
        let mut rows_data: Vec<Vec<f64>> = Vec::new();
        while !self.at(&Token::RBracket) && !self.eof() {
            rows_data.push(self.parse_f64_list()?);
            if self.at(&Token::Comma) {
                self.advance();
            }
        }
        self.eat(&Token::RBracket)?;

        if rows_data.is_empty() {
            return Err(self.error("matrix must have at least one row"));
        }
        let cols = rows_data[0].len();
        for (i, row) in rows_data.iter().enumerate() {
            if row.len() != cols {
                return Err(self.error(format!(
                    "row {} has {} values, expected {}",
                    i,
                    row.len(),
                    cols
                )));
            }
        }
        let rows = rows_data.len();
        let values: Vec<f64> = rows_data.into_iter().flatten().collect();

        Ok(Statement::Matrix(MatrixStmt {
            name,
            rows,
            cols,
            values,
        }))
    }

    // LET <name> = <expr>
    // LAZY LET <name> = <expr>
    // LET LAZY <name> = <expr>
    fn parse_let(&mut self) -> Result<Statement, ParseError> {
        let lazy = match self.peek() {
            Some(Token::Lazy) => {
                self.advance();
                self.eat(&Token::Let)?;
                true
            }
            Some(Token::Let) => {
                self.advance();
                let lazy = self.at(&Token::Lazy);
                if lazy {
                    self.advance();
                }
                lazy
            }
            _ => return Err(self.unexpected("LET or LAZY LET")),
        };

        let name = self.eat_ident()?;
        self.eat(&Token::Eq)?;
        let expr = self.parse_expr()?;

        Ok(Statement::Let(LetStmt { name, lazy, expr }))
    }

    // DERIVE <name> FROM <expr>
    fn parse_derive(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Derive)?;
        let name = self.eat_ident()?;
        self.eat(&Token::From)?;
        let source_expr = self.parse_expr()?;
        Ok(Statement::Derive(DeriveStmt { name, source_expr }))
    }

    // BIND <alias> TO <target>
    fn parse_bind(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Bind)?;
        let alias = self.eat_ident()?;
        self.eat(&Token::To)?;
        let target = self.eat_ident()?;
        Ok(Statement::Bind(BindStmt { alias, target }))
    }

    // ATTACH <tensor> TO <dataset>.<column>
    fn parse_attach(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Attach)?;
        let tensor = self.eat_ident()?;
        self.eat(&Token::To)?;
        let dataset = self.eat_ident()?;
        self.eat(&Token::Dot)?;
        let column = self.eat_ident()?;
        Ok(Statement::Attach(AttachStmt {
            tensor,
            dataset,
            column,
        }))
    }

    // dataset.add_column("col", tensor) — method-call syntax for tensor-first datasets
    fn parse_method_call(&mut self) -> Result<Statement, ParseError> {
        let dataset = self.eat_ident()?;
        self.eat(&Token::Dot)?;
        let method = self.eat_ident()?;
        if method != "add_column" {
            return Err(self.error(format!("Unknown method '{}'", method)));
        }
        self.eat(&Token::LParen)?;
        let column = if self.at(&Token::Str("".into())) {
            self.eat_str()?
        } else {
            self.eat_ident()?
        };
        self.eat(&Token::Comma)?;
        let tensor = self.eat_ident()?;
        self.eat(&Token::RParen)?;
        Ok(Statement::Attach(AttachStmt {
            tensor,
            dataset,
            column,
        }))
    }

    // CREATE DATABASE [IF NOT EXISTS] <name>
    // CREATE [VECTOR] INDEX <idx_name> ON <dataset>(<column>)
    fn parse_create(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Create)?;
        match self.peek().cloned() {
            Some(Token::Database) => {
                self.advance();
                let if_not_exists = if self.at_ident("IF") {
                    self.advance();
                    self.eat(&Token::Not)?;
                    if !self.at_ident("EXISTS") {
                        return Err(self
                            .error("expected EXISTS after NOT in CREATE DATABASE IF NOT EXISTS"));
                    }
                    self.advance();
                    true
                } else {
                    false
                };
                let name = self.eat_ident()?;
                Ok(Statement::CreateDatabase(CreateDatabaseStmt {
                    name,
                    if_not_exists,
                }))
            }
            Some(Token::Index) => {
                self.advance();
                let (dataset, column, kind) = self.parse_index_target(IndexKindAst::Default)?;
                Ok(Statement::CreateIndex(CreateIndexStmt {
                    dataset,
                    column,
                    kind,
                }))
            }
            Some(Token::Vector) => {
                self.advance();
                self.eat(&Token::Index)?;
                let (dataset, column, _) = self.parse_index_target(IndexKindAst::Default)?;
                Ok(Statement::CreateIndex(CreateIndexStmt {
                    dataset,
                    column,
                    kind: IndexKindAst::Vector,
                }))
            }
            _ => Err(self.unexpected("DATABASE or INDEX after CREATE")),
        }
    }

    // <idx_name> ON <dataset>(<column>)
    fn parse_index_target(
        &mut self,
        default_kind: IndexKindAst,
    ) -> Result<(String, String, IndexKindAst), ParseError> {
        let _idx_name = if self.at_any_ident() && !self.at(&Token::On) {
            Some(self.eat_ident()?)
        } else {
            None
        };

        if self.at(&Token::On) {
            self.advance();
        }

        let dataset = self.eat_ident()?;
        self.eat(&Token::LParen)?;
        let column = self.eat_ident()?;
        self.eat(&Token::RParen)?;

        Ok((dataset, column, default_kind))
    }

    // DROP DATABASE [IF EXISTS] <name>
    // DROP PIPELINE <name>
    fn parse_drop(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Drop)?;
        if self.at(&Token::Pipeline) {
            self.advance();
            let name = self.eat_ident()?;
            return Ok(Statement::DropPipeline(name));
        }
        self.eat(&Token::Database)?;
        let if_exists = if self.at_ident("IF") {
            self.advance();
            if !self.at_ident("EXISTS") {
                return Err(self.error("expected EXISTS after IF in DROP DATABASE IF EXISTS"));
            }
            self.advance();
            true
        } else {
            false
        };
        let name = self.eat_ident()?;
        Ok(Statement::DropDatabase(DropDatabaseStmt {
            name,
            if_exists,
        }))
    }

    // TRANSFORM <source> SELECT ... [WHERE ...] [INTO <target>]
    fn parse_transform(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Transform)?;
        let source = self.eat_ident()?;
        self.eat(&Token::Select)?;
        let columns = if self.at(&Token::Star) {
            self.advance();
            SelectColumns::All
        } else {
            let mut cols = vec![self.parse_select_expr()?];
            while self.at(&Token::Comma) {
                self.advance();
                cols.push(self.parse_select_expr()?);
            }
            SelectColumns::Named(cols)
        };
        let filter = if matches!(self.peek(), Some(Token::Where) | Some(Token::Filter)) {
            self.advance();
            Some(self.parse_expr()?)
        } else {
            None
        };
        let target = if self.at(&Token::Into) {
            self.advance();
            Some(self.eat_ident()?)
        } else {
            None
        };
        Ok(Statement::Transform(TransformStmt {
            source,
            columns,
            filter,
            target,
        }))
    }

    // (Already ate DEFINE) PIPELINE <name> AS step [THEN step ...]
    fn parse_define_pipeline(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Pipeline)?;
        let name = self.eat_ident()?;
        self.eat(&Token::As)?;
        let mut steps = vec![self.parse_pipeline_step()?];
        while self.at(&Token::Then) {
            self.advance();
            steps.push(self.parse_pipeline_step()?);
        }
        Ok(Statement::DefinePipeline(DefinePipelineStmt {
            name,
            steps,
            source: String::new(), // populated by execute_line_with_context
        }))
    }

    fn parse_pipeline_step(&mut self) -> Result<PipelineStep, ParseError> {
        match self.peek() {
            Some(Token::Select) => {
                self.advance();
                let mut exprs = vec![self.parse_select_expr()?];
                while self.at(&Token::Comma) {
                    self.advance();
                    exprs.push(self.parse_select_expr()?);
                }
                Ok(PipelineStep::Select(exprs))
            }
            Some(Token::Where) | Some(Token::Filter) => {
                self.advance();
                Ok(PipelineStep::Filter(self.parse_expr()?))
            }
            Some(Token::Order) => {
                self.advance();
                self.eat(&Token::By)?;
                let mut cols = vec![];
                loop {
                    let col = self.eat_ident()?;
                    let ascending = if self.at_ident("DESC") {
                        self.advance();
                        false
                    } else {
                        if self.at_ident("ASC") {
                            self.advance();
                        }
                        true
                    };
                    cols.push((col, ascending));
                    if !self.at(&Token::Comma) {
                        break;
                    }
                    self.advance();
                }
                Ok(PipelineStep::OrderBy(cols))
            }
            Some(Token::Limit) => {
                self.advance();
                let n = self.eat_usize()?;
                Ok(PipelineStep::Limit(n))
            }
            Some(Token::Normalize) => {
                self.advance();
                let col = self.eat_ident()?;
                Ok(PipelineStep::NormalizeCol(col))
            }
            _ => Err(self.unexpected("SELECT, WHERE, FILTER, ORDER BY, LIMIT, or NORMALIZE")),
        }
    }

    // APPLY PIPELINE <name> ON <source> [INTO <target>]
    fn parse_apply_pipeline(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Apply)?;
        self.eat(&Token::Pipeline)?;
        let pipeline = self.eat_ident()?;
        self.eat(&Token::On)?;
        let source = self.eat_ident()?;
        let into = if self.at(&Token::Into) {
            self.advance();
            Some(self.eat_ident()?)
        } else {
            None
        };
        Ok(Statement::ApplyPipeline(ApplyPipelineStmt {
            pipeline,
            source,
            into,
        }))
    }

    // DESCRIBE PIPELINE <name>
    fn parse_describe_pipeline(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Describe)?;
        self.eat(&Token::Pipeline)?;
        let name = self.eat_ident()?;
        Ok(Statement::DescribePipeline(name))
    }

    // SET DATASET <name> [METADATA] <key> = <value>
    fn parse_set(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Set)?;
        self.eat(&Token::Dataset)?;
        let dataset = self.eat_ident()?;
        if self.at(&Token::Metadata) {
            self.advance();
        }
        let key = self.eat_ident()?;
        self.eat(&Token::Eq)?;
        let value = self.eat_str()?;
        Ok(Statement::SetMetadata(SetMetadataStmt {
            dataset,
            key,
            value,
        }))
    }
}

// ─── Helper parsers ───────────────────────────────────────────────────────────

impl Parser {
    fn parse_usize_list(&mut self) -> Result<Vec<usize>, ParseError> {
        self.eat(&Token::LBracket)?;
        let mut dims = Vec::new();
        while !self.at(&Token::RBracket) && !self.eof() {
            dims.push(self.eat_usize()?);
            if self.at(&Token::Comma) {
                self.advance();
            }
        }
        self.eat(&Token::RBracket)?;
        Ok(dims)
    }

    fn parse_f64_list(&mut self) -> Result<Vec<f64>, ParseError> {
        self.eat(&Token::LBracket)?;
        let mut vals = Vec::new();
        while !self.at(&Token::RBracket) && !self.eof() {
            vals.push(self.eat_number()?);
            if self.at(&Token::Comma) {
                self.advance();
            }
        }
        self.eat(&Token::RBracket)?;
        Ok(vals)
    }

    fn parse_index_specs(&mut self) -> Result<Vec<IndexSpec>, ParseError> {
        let mut specs = Vec::new();
        while !self.at(&Token::RBracket) && !self.eof() {
            let spec = if self.at(&Token::Star) || self.at(&Token::Colon) {
                self.advance();
                IndexSpec::All
            } else {
                let start = self.eat_usize()?;
                if self.at(&Token::Colon) {
                    self.advance();
                    let end = self.eat_usize()?;
                    IndexSpec::Range(start, end)
                } else {
                    IndexSpec::Index(start)
                }
            };
            specs.push(spec);
            if self.at(&Token::Comma) {
                self.advance();
            }
        }
        Ok(specs)
    }

    fn parse_column_list(&mut self) -> Result<Vec<ColumnDef>, ParseError> {
        let has_parens = self.at(&Token::LParen);
        if has_parens {
            self.advance();
        }
        let mut cols = Vec::new();
        loop {
            if self.eof() {
                break;
            }
            if has_parens && self.at(&Token::RParen) {
                break;
            }
            if !has_parens && !matches!(self.peek(), Some(Token::Ident(_))) {
                break;
            }
            cols.push(self.parse_column_def()?);
            if self.at(&Token::Comma) {
                self.advance();
            }
        }
        if has_parens {
            self.eat(&Token::RParen)?;
        }
        Ok(cols)
    }

    fn parse_column_def(&mut self) -> Result<ColumnDef, ParseError> {
        let name = self.eat_ident()?;
        self.eat(&Token::Colon)?;
        let col_type = self.parse_col_type()?;

        let nullable = if self.at(&Token::Question) {
            self.advance();
            true
        } else if self.at(&Token::Not) {
            self.advance();
            self.eat(&Token::Nullable)?;
            false
        } else if self.at(&Token::Nullable) {
            self.advance();
            true
        } else {
            false
        };

        let default_val = if self.at_ident("DEFAULT") {
            self.advance();
            Some(self.parse_filter_value()?)
        } else {
            None
        };

        Ok(ColumnDef {
            name,
            col_type,
            nullable,
            default_val,
        })
    }

    fn parse_col_type(&mut self) -> Result<ColType, ParseError> {
        match self.peek() {
            Some(Token::Vector) => {
                self.advance();
                self.eat(&Token::LParen)?;
                let n = self.eat_usize()?;
                self.eat(&Token::RParen)?;
                Ok(ColType::Vector(n))
            }
            Some(Token::Matrix) => {
                self.advance();
                self.eat(&Token::LParen)?;
                let rows = self.eat_usize()?;
                self.eat(&Token::Comma)?;
                let cols = self.eat_usize()?;
                self.eat(&Token::RParen)?;
                Ok(ColType::Matrix(rows, cols))
            }
            Some(Token::Tensor) => {
                self.advance();
                let dims = if self.at(&Token::LParen) {
                    self.eat(&Token::LParen)?;
                    let mut dims = Vec::new();
                    while !self.at(&Token::RParen) && !self.eof() {
                        dims.push(self.eat_usize()?);
                        if self.at(&Token::Comma) {
                            self.advance();
                        }
                    }
                    self.eat(&Token::RParen)?;
                    dims
                } else {
                    self.parse_usize_list()?
                };
                Ok(ColType::Tensor(dims))
            }
            Some(Token::Ident(_)) => {
                let t = self.eat_ident()?;
                match t.to_uppercase().as_str() {
                    "INT" | "INTEGER" | "INT32" | "INT64" => Ok(ColType::Int),
                    "FLOAT" | "FLOAT32" | "FLOAT64" | "DOUBLE" => Ok(ColType::Float),
                    "STRING" | "TEXT" | "VARCHAR" => Ok(ColType::String),
                    "BOOL" | "BOOLEAN" => Ok(ColType::Bool),
                    "VECTOR" => {
                        self.eat(&Token::LParen)?;
                        let n = self.eat_usize()?;
                        self.eat(&Token::RParen)?;
                        Ok(ColType::Vector(n))
                    }
                    "MATRIX" => {
                        self.eat(&Token::LParen)?;
                        let rows = self.eat_usize()?;
                        self.eat(&Token::Comma)?;
                        let cols = self.eat_usize()?;
                        self.eat(&Token::RParen)?;
                        Ok(ColType::Matrix(rows, cols))
                    }
                    "TENSOR" => {
                        let dims = if self.at(&Token::LParen) {
                            self.eat(&Token::LParen)?;
                            let mut dims = Vec::new();
                            while !self.at(&Token::RParen) && !self.eof() {
                                dims.push(self.eat_usize()?);
                                if self.at(&Token::Comma) {
                                    self.advance();
                                }
                            }
                            self.eat(&Token::RParen)?;
                            dims
                        } else {
                            self.parse_usize_list()?
                        };
                        Ok(ColType::Tensor(dims))
                    }
                    other => Err(self.error(format!("unknown column type '{}'", other))),
                }
            }
            _ => Err(self.unexpected(
                "a column type: Int, Float, String, Bool, Vector(n), Matrix(r,c), Tensor[...]",
            )),
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_ok(src: &str) -> Statement {
        parse(src).unwrap_or_else(|e| panic!("parse failed for {:?}: {}", src, e))
    }

    fn parse_err(src: &str) -> ParseError {
        parse(src).expect_err(&format!("expected parse error for {:?}", src))
    }

    #[test]
    fn vector_literal() {
        let stmt = parse_ok("VECTOR v = [1, 2, 3]");
        let Statement::Vector(s) = stmt else {
            panic!("expected Vector")
        };
        assert_eq!(s.name, "v");
        assert_eq!(s.values, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn matrix_literal() {
        let stmt = parse_ok("MATRIX m = [[1, 2], [3, 4]]");
        let Statement::Matrix(s) = stmt else {
            panic!("expected Matrix")
        };
        assert_eq!(s.name, "m");
        assert_eq!(s.rows, 2);
        assert_eq!(s.cols, 2);
        assert_eq!(s.values, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn define_tensor_normal() {
        let stmt = parse_ok("DEFINE v AS TENSOR [3] VALUES [1, 0, 0]");
        let Statement::DefineTensor(s) = stmt else {
            panic!("expected DefineTensor")
        };
        assert_eq!(s.name, "v");
        assert_eq!(s.kind, TensorKindAst::Normal);
        assert_eq!(s.shape, vec![3]);
        assert_eq!(s.values, vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn define_tensor_strict() {
        let stmt = parse_ok("DEFINE v AS STRICT TENSOR [3] VALUES [1, 0, 0]");
        let Statement::DefineTensor(s) = stmt else {
            panic!("expected DefineTensor")
        };
        assert_eq!(s.kind, TensorKindAst::Strict);
    }

    #[test]
    fn let_named_op() {
        let stmt = parse_ok("LET result = ADD a b");
        let Statement::Let(s) = stmt else {
            panic!("expected Let")
        };
        assert_eq!(s.name, "result");
        assert!(!s.lazy);
        assert!(matches!(s.expr, Expr::Call(CallExpr::Add(..))));
    }

    #[test]
    fn let_lazy_prefix() {
        let stmt = parse_ok("LAZY LET trend = STDEV sensor");
        let Statement::Let(s) = stmt else {
            panic!("expected Let")
        };
        assert!(s.lazy);
        assert!(matches!(s.expr, Expr::Call(CallExpr::Stdev(..))));
    }

    #[test]
    fn let_lazy_suffix() {
        let stmt = parse_ok("LET LAZY trend = MEAN sensor");
        let Statement::Let(s) = stmt else {
            panic!("expected Let")
        };
        assert!(s.lazy);
    }

    #[test]
    fn let_infix_add() {
        let stmt = parse_ok("LET c = a + b");
        let Statement::Let(s) = stmt else {
            panic!("expected Let")
        };
        assert!(matches!(
            s.expr,
            Expr::Infix {
                op: InfixOp::Add,
                ..
            }
        ));
    }

    #[test]
    fn let_infix_precedence() {
        let stmt = parse_ok("LET c = a + b * 2.0");
        let Statement::Let(s) = stmt else { panic!() };
        let Expr::Infix {
            op: InfixOp::Add,
            lhs,
            rhs,
        } = s.expr
        else {
            panic!("expected Add at root")
        };
        assert!(matches!(*lhs, Expr::Ref(_)));
        assert!(matches!(
            *rhs,
            Expr::Infix {
                op: InfixOp::Multiply,
                ..
            }
        ));
    }

    #[test]
    fn let_subscript() {
        let stmt = parse_ok("LET x = t[0, 1]");
        let Statement::Let(s) = stmt else { panic!() };
        let Expr::Index { base, indices } = s.expr else {
            panic!("expected Index")
        };
        assert!(matches!(*base, Expr::Ref(ref n) if n == "t"));
        assert_eq!(indices.len(), 2);
        assert!(matches!(indices[0], IndexSpec::Index(0)));
        assert!(matches!(indices[1], IndexSpec::Index(1)));
    }

    #[test]
    fn let_range_subscript() {
        let stmt = parse_ok("LET x = t[0:5, *]");
        let Statement::Let(s) = stmt else { panic!() };
        let Expr::Index { indices, .. } = s.expr else {
            panic!()
        };
        assert!(matches!(indices[0], IndexSpec::Range(0, 5)));
        assert!(matches!(indices[1], IndexSpec::All));
    }

    #[test]
    fn let_field_access() {
        let stmt = parse_ok("LET x = ds.col");
        let Statement::Let(s) = stmt else { panic!() };
        assert!(matches!(s.expr, Expr::Field { .. }));
    }

    #[test]
    fn let_correlate() {
        let stmt = parse_ok("LET sim = CORRELATE a WITH b");
        let Statement::Let(s) = stmt else { panic!() };
        assert!(matches!(s.expr, Expr::Call(CallExpr::Correlate(..))));
    }

    #[test]
    fn let_scale() {
        let stmt = parse_ok("LET s = SCALE a BY 0.5");
        let Statement::Let(s) = stmt else { panic!() };
        let Expr::Call(CallExpr::Scale { factor, .. }) = s.expr else {
            panic!()
        };
        assert!((factor - 0.5).abs() < 1e-9);
    }

    #[test]
    fn let_reshape() {
        let stmt = parse_ok("LET r = RESHAPE a TO [2, 3]");
        let Statement::Let(s) = stmt else { panic!() };
        let Expr::Call(CallExpr::Reshape { shape, .. }) = s.expr else {
            panic!()
        };
        assert_eq!(shape, vec![2, 3]);
    }

    #[test]
    fn let_stack() {
        let stmt = parse_ok("LET s = STACK a b c");
        let Statement::Let(s) = stmt else { panic!() };
        let Expr::Call(CallExpr::Stack(ops)) = s.expr else {
            panic!()
        };
        assert_eq!(ops.len(), 3);
    }

    #[test]
    fn let_dataset_constructor() {
        let stmt = parse_ok(r#"LET ds = dataset("my_ds")"#);
        let Statement::Let(s) = stmt else { panic!() };
        assert!(matches!(s.expr, Expr::DatasetRef(ref n) if n == "my_ds"));
    }

    #[test]
    fn bind_statement() {
        let stmt = parse_ok("BIND alias TO source");
        let Statement::Bind(s) = stmt else { panic!() };
        assert_eq!(s.alias, "alias");
        assert_eq!(s.target, "source");
    }

    #[test]
    fn attach_statement() {
        let stmt = parse_ok("ATTACH my_tensor TO my_dataset.my_col");
        let Statement::Attach(s) = stmt else { panic!() };
        assert_eq!(s.tensor, "my_tensor");
        assert_eq!(s.dataset, "my_dataset");
        assert_eq!(s.column, "my_col");
    }

    #[test]
    fn derive_statement() {
        let stmt = parse_ok("DERIVE clean FROM NORMALIZE raw");
        let Statement::Derive(s) = stmt else { panic!() };
        assert_eq!(s.name, "clean");
        assert!(matches!(s.source_expr, Expr::Call(CallExpr::Normalize(..))));
    }

    #[test]
    fn show_schema() {
        let stmt = parse_ok("SHOW SCHEMA my_dataset");
        let Statement::Show(s) = stmt else { panic!() };
        assert!(matches!(s.target, ShowTarget::Schema(ref n) if n == "my_dataset"));
    }

    #[test]
    fn show_all() {
        let stmt = parse_ok("SHOW ALL");
        let Statement::Show(s) = stmt else { panic!() };
        assert!(matches!(s.target, ShowTarget::All));
    }

    #[test]
    fn show_all_datasets() {
        let stmt = parse_ok("SHOW ALL DATASETS");
        let Statement::Show(s) = stmt else { panic!() };
        assert!(matches!(s.target, ShowTarget::AllDatasets));
    }

    #[test]
    fn show_named() {
        let stmt = parse_ok("SHOW my_tensor");
        let Statement::Show(s) = stmt else { panic!() };
        assert!(matches!(s.target, ShowTarget::Named(ref n) if n == "my_tensor"));
    }

    #[test]
    fn show_dataset_metadata() {
        let stmt = parse_ok("SHOW DATASET METADATA foo");
        let Statement::Show(s) = stmt else { panic!() };
        assert!(matches!(s.target, ShowTarget::DatasetMetadata(ref n) if n == "foo"));
    }

    #[test]
    fn create_dataset_columns() {
        let stmt = parse_ok("DATASET diagnostics COLUMNS (id: Int, emb: Vector(128))");
        let Statement::CreateDataset(s) = stmt else {
            panic!()
        };
        assert_eq!(s.name, "diagnostics");
        assert_eq!(s.columns.len(), 2);
        assert_eq!(s.columns[0].name, "id");
        assert!(matches!(s.columns[0].col_type, ColType::Int));
        assert!(matches!(s.columns[1].col_type, ColType::Vector(128)));
    }

    #[test]
    fn create_dataset_matrix_column() {
        let stmt = parse_ok("DATASET t COLUMNS (feat: Matrix(4, 4))");
        let Statement::CreateDataset(s) = stmt else {
            panic!()
        };
        assert!(matches!(s.columns[0].col_type, ColType::Matrix(4, 4)));
    }

    #[test]
    fn select_basic() {
        let stmt = parse_ok("SELECT col1, col2 FROM my_ds");
        let Statement::Select(s) = stmt else { panic!() };
        assert!(matches!(&s.source, DatasetSource::Named(n) if n == "my_ds"));
        let SelectColumns::Named(cols) = s.columns else {
            panic!()
        };
        assert!(matches!(&cols[0], SelectExpr::Column(c) if c == "col1"));
        assert!(matches!(&cols[1], SelectExpr::Column(c) if c == "col2"));
    }

    #[test]
    fn select_with_limit() {
        let stmt = parse_ok("SELECT * FROM my_ds LIMIT 10");
        let Statement::Select(s) = stmt else { panic!() };
        assert!(matches!(s.columns, SelectColumns::All));
        assert_eq!(s.limit, Some(10));
    }

    #[test]
    fn save_tensor() {
        let stmt = parse_ok(r#"SAVE TENSOR my_t TO "data/my_t.json""#);
        let Statement::Save(s) = stmt else { panic!() };
        assert!(matches!(s.kind, PersistKind::Tensor));
        assert_eq!(s.name, "my_t");
        assert_eq!(s.path, Some("data/my_t.json".into()));
    }

    #[test]
    fn load_dataset() {
        let stmt = parse_ok(r#"LOAD DATASET ds FROM "data/ds""#);
        let Statement::Load(s) = stmt else { panic!() };
        assert!(matches!(s.kind, PersistKind::Dataset));
        assert_eq!(s.path, Some("data/ds".into()));
    }

    #[test]
    fn use_database() {
        let stmt = parse_ok("USE analytics");
        let Statement::UseDatabase(s) = stmt else {
            panic!()
        };
        assert_eq!(s.name, "analytics");
    }

    #[test]
    fn use_dataset_from() {
        let stmt = parse_ok(r#"USE DATASET FROM "data/foo.h5ad""#);
        let Statement::Import(s) = stmt else { panic!() };
        assert!(s.ephemeral);
        assert_eq!(s.path, "data/foo.h5ad");
    }

    #[test]
    fn import_dataset_from() {
        let stmt = parse_ok(r#"IMPORT DATASET FROM "data/foo.parquet""#);
        let Statement::Import(s) = stmt else { panic!() };
        assert!(!s.ephemeral);
    }

    #[test]
    fn create_database() {
        let stmt = parse_ok("CREATE DATABASE mydb");
        let Statement::CreateDatabase(s) = stmt else {
            panic!()
        };
        assert_eq!(s.name, "mydb");
        assert!(!s.if_not_exists);
    }

    #[test]
    fn create_database_if_not_exists() {
        let stmt = parse_ok("CREATE DATABASE IF NOT EXISTS mydb");
        let Statement::CreateDatabase(s) = stmt else {
            panic!()
        };
        assert_eq!(s.name, "mydb");
        assert!(s.if_not_exists);
    }

    #[test]
    fn drop_database() {
        let stmt = parse_ok("DROP DATABASE mydb");
        let Statement::DropDatabase(s) = stmt else {
            panic!()
        };
        assert_eq!(s.name, "mydb");
        assert!(!s.if_exists);
    }

    #[test]
    fn drop_database_if_exists() {
        let stmt = parse_ok("DROP DATABASE IF EXISTS mydb");
        let Statement::DropDatabase(s) = stmt else {
            panic!()
        };
        assert_eq!(s.name, "mydb");
        assert!(s.if_exists);
    }

    #[test]
    fn reset_session() {
        let stmt = parse_ok("RESET SESSION");
        assert!(matches!(stmt, Statement::Reset));
    }

    #[test]
    fn create_vector_index() {
        let stmt = parse_ok("CREATE VECTOR INDEX v_idx ON vecs(col)");
        let Statement::CreateIndex(s) = stmt else {
            panic!()
        };
        assert_eq!(s.dataset, "vecs");
        assert_eq!(s.column, "col");
        assert!(matches!(s.kind, IndexKindAst::Vector));
    }

    #[test]
    fn explain_dataset() {
        let stmt = parse_ok("EXPLAIN DATASET users");
        let Statement::Explain(s) = stmt else {
            panic!()
        };
        assert!(matches!(s.target, ExplainTarget::Dataset(ref n) if n == "users"));
    }

    #[test]
    fn explain_bare_ident() {
        let stmt = parse_ok("EXPLAIN foo");
        let Statement::Explain(s) = stmt else {
            panic!()
        };
        assert!(matches!(s.target, ExplainTarget::Dataset(ref n) if n == "foo"));
    }

    #[test]
    fn reset_statement() {
        let stmt = parse_ok("RESET");
        assert!(matches!(stmt, Statement::Reset));
    }

    #[test]
    fn search_statement() {
        let stmt = parse_ok("SEARCH embeddings ON vec QUERY q LIMIT 5");
        let Statement::Search(s) = stmt else { panic!() };
        assert_eq!(s.dataset, "embeddings");
        assert_eq!(s.column, "vec");
        assert!(matches!(s.query, SearchQuery::TensorRef(ref n) if n == "q"));
        assert_eq!(s.top_k, 5);
        assert!(s.target.is_none());
    }

    #[test]
    fn search_with_into() {
        let stmt = parse_ok("SEARCH embeddings ON vec QUERY q LIMIT 10 INTO results");
        let Statement::Search(s) = stmt else { panic!() };
        assert_eq!(s.target, Some("results".to_string()));
    }

    #[test]
    fn search_inline_vector() {
        let stmt = parse_ok("SEARCH embeddings ON vec QUERY [1.0, 2.0, 3.0] LIMIT 5");
        let Statement::Search(s) = stmt else { panic!() };
        assert_eq!(s.dataset, "embeddings");
        assert_eq!(s.column, "vec");
        assert!(matches!(s.query, SearchQuery::Inline(ref v) if v == &vec![1.0, 2.0, 3.0]));
        assert_eq!(s.top_k, 5);
    }

    #[test]
    fn search_inline_vector_into() {
        let stmt = parse_ok("SEARCH emb ON col QUERY [0.5, -1.0] LIMIT 3 INTO out");
        let Statement::Search(s) = stmt else { panic!() };
        assert!(matches!(s.query, SearchQuery::Inline(_)));
        assert_eq!(s.target, Some("out".to_string()));
    }

    #[test]
    fn dataset_from_source() {
        let stmt = parse_ok("DATASET filtered FROM employees");
        let Statement::CreateDataset(s) = stmt else {
            panic!()
        };
        assert_eq!(s.name, "filtered");
        let from = s.from.unwrap();
        assert_eq!(from.source, "employees");
        assert!(from.filter.is_none());
        assert!(from.select.is_none());
        assert!(from.order_by.is_none());
        assert!(from.limit.is_none());
    }

    #[test]
    fn dataset_from_with_filter() {
        let stmt = parse_ok("DATASET seniors FROM employees FILTER age >= 60");
        let Statement::CreateDataset(s) = stmt else {
            panic!()
        };
        let from = s.from.unwrap();
        assert_eq!(from.source, "employees");
        let f = from.filter.unwrap();
        assert!(matches!(
            f,
            Expr::Infix {
                op: InfixOp::GtEq,
                ..
            }
        ));
    }

    #[test]
    fn dataset_from_full_clauses() {
        let stmt = parse_ok("DATASET top FROM orders SELECT name ORDER BY total DESC LIMIT 10");
        let Statement::CreateDataset(s) = stmt else {
            panic!()
        };
        let from = s.from.unwrap();
        assert_eq!(from.source, "orders");
        let ord = from.order_by.unwrap();
        assert_eq!(ord.columns.len(), 1);
        assert_eq!(ord.columns[0].0, "total");
        assert!(!ord.columns[0].1);
        assert_eq!(from.limit, Some(10));
    }

    #[test]
    fn select_aggregate_columns() {
        let stmt = parse_ok("SELECT SUM(price), AVG(qty) FROM orders GROUP BY category");
        let Statement::Select(s) = stmt else { panic!() };
        assert_eq!(s.group_by, vec!["category"]);
        let SelectColumns::Named(cols) = s.columns else {
            panic!()
        };
        assert!(
            matches!(&cols[0], SelectExpr::Aggregate { func: AggFuncAst::Sum, expr, .. } if matches!(expr.as_ref(), Expr::Ref(c) if c == "price"))
        );
        assert!(
            matches!(&cols[1], SelectExpr::Aggregate { func: AggFuncAst::Avg, expr, .. } if matches!(expr.as_ref(), Expr::Ref(c) if c == "qty"))
        );
    }

    #[test]
    fn select_count_star() {
        let stmt = parse_ok("SELECT COUNT(*) FROM orders GROUP BY category");
        let Statement::Select(s) = stmt else { panic!() };
        let SelectColumns::Named(cols) = s.columns else {
            panic!()
        };
        assert!(
            matches!(&cols[0], SelectExpr::Aggregate { func: AggFuncAst::Count, expr, .. } if matches!(expr.as_ref(), Expr::Ref(c) if c == "*"))
        );
    }

    #[test]
    fn select_mixed_plain_and_agg() {
        let stmt = parse_ok("SELECT category, MAX(price) FROM orders GROUP BY category");
        let Statement::Select(s) = stmt else { panic!() };
        let SelectColumns::Named(cols) = s.columns else {
            panic!()
        };
        assert!(matches!(&cols[0], SelectExpr::Column(c) if c == "category"));
        assert!(
            matches!(&cols[1], SelectExpr::Aggregate { func: AggFuncAst::Max, expr, .. } if matches!(expr.as_ref(), Expr::Ref(c) if c == "price"))
        );
    }

    #[test]
    fn is_read_only() {
        assert!(!parse_ok("SHOW ALL").is_read_only());
        assert!(parse_ok("EXPLAIN foo").is_read_only());
        assert!(parse_ok("LIST TENSORS").is_read_only());
        assert!(parse_ok("DELIVER users").is_read_only());
        assert!(!parse_ok("LET x = ADD a b").is_read_only());
        assert!(!parse_ok("VECTOR v = [1, 2]").is_read_only());
    }

    #[test]
    fn is_read_only_pipeline_mutations_require_write_lock() {
        // CONSISTENCY_PLAN.md Track C / C5: DEFINE/APPLY/SAVE/LOAD/DROP
        // PIPELINE all mutate db.pipelines (or persist/restore it) and must
        // NOT be classified read-only, or the server would run them under a
        // read lock (src/server/mod.rs consults is_read_only to pick
        // db_arc.read() vs db_arc.write()).
        assert!(!parse_ok("DEFINE PIPELINE clean AS WHERE active = 1").is_read_only());
        assert!(!parse_ok("APPLY PIPELINE clean ON products").is_read_only());
        assert!(!parse_ok("APPLY PIPELINE clean ON products INTO top_products").is_read_only());
        assert!(!parse_ok("SAVE PIPELINE clean").is_read_only());
        assert!(!parse_ok("LOAD PIPELINE clean").is_read_only());
        assert!(!parse_ok("DROP PIPELINE clean").is_read_only());
    }

    #[test]
    fn unknown_command_errors() {
        let e = parse_err("UNKNOWN foo bar");
        assert!(e.msg.contains("expected a statement keyword"));
    }

    #[test]
    fn empty_input_errors() {
        let e = parse_err("");
        assert!(e.msg.contains("empty input"));
    }

    #[test]
    fn missing_eq_in_let_errors() {
        let e = parse_err("LET x ADD a b");
        assert!(e.msg.contains('=') || e.msg.contains("expected"));
    }

    #[test]
    fn trailing_tokens_error() {
        let e = parse_err("RESET RESET");
        assert!(e.msg.contains("unexpected token"));
    }

    #[test]
    fn select_where_and() {
        let stmt = parse_ok("SELECT * FROM users WHERE age > 30 AND active = 1");
        let Statement::Select(s) = stmt else {
            panic!("expected Select")
        };
        assert!(matches!(s.filter, Some(Expr::And(..))));
    }

    #[test]
    fn select_where_or() {
        let stmt = parse_ok("SELECT * FROM users WHERE status = 1 OR role = 2");
        let Statement::Select(s) = stmt else {
            panic!("expected Select")
        };
        assert!(matches!(s.filter, Some(Expr::Or(..))));
    }

    #[test]
    fn select_where_not() {
        let stmt = parse_ok("SELECT * FROM users WHERE NOT active");
        let Statement::Select(s) = stmt else {
            panic!("expected Select")
        };
        assert!(matches!(s.filter, Some(Expr::Not(..))));
    }

    #[test]
    fn select_where_and_or_precedence() {
        // `a AND b OR c` should parse as `(a AND b) OR c`
        let stmt = parse_ok("SELECT * FROM t WHERE x > 1 AND y > 2 OR z > 3");
        let Statement::Select(s) = stmt else {
            panic!("expected Select")
        };
        // OR at top level
        assert!(matches!(s.filter, Some(Expr::Or(..))));
        if let Some(Expr::Or(lhs, _)) = s.filter {
            // AND on the left
            assert!(matches!(*lhs, Expr::And(..)));
        }
    }

    #[test]
    fn select_where_compound_three_and() {
        let stmt = parse_ok("SELECT * FROM t WHERE a > 1 AND b > 2 AND c > 3");
        let Statement::Select(s) = stmt else {
            panic!("expected Select")
        };
        assert!(matches!(s.filter, Some(Expr::And(..))));
    }

    #[test]
    fn select_where_is_null() {
        let stmt = parse_ok("SELECT * FROM users WHERE email IS NULL");
        let Statement::Select(s) = stmt else {
            panic!("expected Select")
        };
        assert!(matches!(s.filter, Some(Expr::IsNull(..))));
    }

    #[test]
    fn select_where_is_not_null() {
        let stmt = parse_ok("SELECT * FROM users WHERE email IS NOT NULL");
        let Statement::Select(s) = stmt else {
            panic!("expected Select")
        };
        assert!(matches!(s.filter, Some(Expr::IsNotNull(..))));
    }

    #[test]
    fn select_inner_join() {
        let stmt =
            parse_ok("SELECT * FROM orders INNER JOIN users ON user_id = id WHERE total > 100");
        let Statement::Select(s) = stmt else {
            panic!("expected Select")
        };
        assert!(matches!(&s.source, DatasetSource::Named(n) if n == "orders"));
        assert_eq!(s.joins.len(), 1);
        assert_eq!(s.joins[0].kind, JoinKind::Inner);
        assert_eq!(s.joins[0].dataset, "users");
        assert_eq!(s.joins[0].left_col, "user_id");
        assert_eq!(s.joins[0].right_col, "id");
        assert!(s.filter.is_some());
    }

    #[test]
    fn select_left_join() {
        let stmt = parse_ok("SELECT * FROM orders LEFT JOIN users ON orders.user_id = users.id");
        let Statement::Select(s) = stmt else {
            panic!("expected Select")
        };
        assert_eq!(s.joins[0].kind, JoinKind::Left);
    }

    #[test]
    fn select_bare_join() {
        let stmt = parse_ok("SELECT * FROM a JOIN b ON a_id = b_id");
        let Statement::Select(s) = stmt else {
            panic!("expected Select")
        };
        assert_eq!(s.joins[0].kind, JoinKind::Inner);
    }

    #[test]
    fn update_statement() {
        let stmt = parse_ok("UPDATE products SET price = 99 WHERE stock > 0");
        let Statement::Update(s) = stmt else {
            panic!("expected Update")
        };
        assert_eq!(s.dataset, "products");
        assert_eq!(s.assignments.len(), 1);
        assert_eq!(s.assignments[0].0, "price");
        assert!(s.filter.is_some());
    }

    #[test]
    fn update_multiple_cols() {
        let stmt = parse_ok("UPDATE products SET price = 10, stock = 0");
        let Statement::Update(s) = stmt else {
            panic!("expected Update")
        };
        assert_eq!(s.assignments.len(), 2);
        assert!(s.filter.is_none());
    }

    #[test]
    fn delete_with_filter() {
        let stmt = parse_ok("DELETE FROM orders WHERE status = 0");
        let Statement::Delete(s) = stmt else {
            panic!("expected Delete")
        };
        assert_eq!(s.dataset, "orders");
        assert!(s.filter.is_some());
    }

    #[test]
    fn delete_all() {
        let stmt = parse_ok("DELETE FROM temp_data");
        let Statement::Delete(s) = stmt else {
            panic!("expected Delete")
        };
        assert!(s.filter.is_none());
    }

    #[test]
    fn select_limit_offset() {
        let stmt = parse_ok("SELECT * FROM users LIMIT 10 OFFSET 5");
        let Statement::Select(s) = stmt else {
            panic!("expected Select")
        };
        assert_eq!(s.limit, Some(10));
        assert_eq!(s.offset, Some(5));
    }

    #[test]
    fn select_multi_col_order_by() {
        let stmt = parse_ok("SELECT * FROM users ORDER BY age DESC, name ASC");
        let Statement::Select(s) = stmt else {
            panic!("expected Select")
        };
        let ord = s.order_by.unwrap();
        assert_eq!(ord.columns.len(), 2);
        assert_eq!(ord.columns[0].0, "age");
        assert!(!ord.columns[0].1);
        assert_eq!(ord.columns[1].0, "name");
        assert!(ord.columns[1].1);
    }

    #[test]
    fn select_where_in() {
        let stmt = parse_ok("SELECT * FROM users WHERE status IN (1, 2, 3)");
        let Statement::Select(s) = stmt else {
            panic!("expected Select")
        };
        assert!(matches!(s.filter, Some(Expr::In { .. })));
    }

    #[test]
    fn select_where_between() {
        let stmt = parse_ok("SELECT * FROM users WHERE age BETWEEN 18 AND 65");
        let Statement::Select(s) = stmt else {
            panic!("expected Select")
        };
        assert!(matches!(s.filter, Some(Expr::Between { .. })));
    }

    #[test]
    fn select_right_join() {
        let stmt = parse_ok("SELECT * FROM orders RIGHT JOIN users ON user_id = id");
        let Statement::Select(s) = stmt else {
            panic!("expected Select")
        };
        assert_eq!(s.joins[0].kind, JoinKind::Right);
    }

    #[test]
    fn select_right_outer_join() {
        let stmt = parse_ok("SELECT * FROM orders RIGHT OUTER JOIN users ON user_id = id");
        let Statement::Select(s) = stmt else {
            panic!("expected Select")
        };
        assert_eq!(s.joins[0].kind, JoinKind::Right);
    }

    #[test]
    fn select_full_outer_join() {
        let stmt = parse_ok("SELECT * FROM orders FULL OUTER JOIN users ON user_id = id");
        let Statement::Select(s) = stmt else {
            panic!("expected Select")
        };
        assert_eq!(s.joins[0].kind, JoinKind::Full);
    }

    #[test]
    fn select_full_join() {
        let stmt = parse_ok("SELECT * FROM a FULL JOIN b ON a_id = b_id");
        let Statement::Select(s) = stmt else {
            panic!("expected Select")
        };
        assert_eq!(s.joins[0].kind, JoinKind::Full);
    }

    #[test]
    fn select_subquery() {
        let stmt =
            parse_ok("SELECT * FROM (SELECT age, name FROM employees WHERE age > 30) AS sub");
        let Statement::Select(s) = stmt else {
            panic!("expected Select")
        };
        match s.source {
            DatasetSource::Subquery { alias, .. } => assert_eq!(alias, "sub"),
            _ => panic!("expected Subquery source"),
        }
    }

    #[test]
    fn select_order_by_single_no_direction() {
        let stmt = parse_ok("SELECT * FROM orders ORDER BY total");
        let Statement::Select(s) = stmt else {
            panic!("expected Select")
        };
        let ord = s.order_by.unwrap();
        assert_eq!(ord.columns.len(), 1);
        assert_eq!(ord.columns[0].0, "total");
        assert!(ord.columns[0].1); // ascending by default
    }

    #[test]
    fn select_between_compound() {
        let stmt = parse_ok("SELECT * FROM t WHERE age BETWEEN 18 AND 65 AND active = 1");
        let Statement::Select(s) = stmt else {
            panic!("expected Select")
        };
        // AND at top level with Between on left
        assert!(matches!(s.filter, Some(Expr::And(..))));
        if let Some(Expr::And(lhs, _)) = s.filter {
            assert!(matches!(*lhs, Expr::Between { .. }));
        }
    }

    // ─── v0.1.29: CTEs, Window Functions, UNION, CASE WHEN, DISTINCT, COALESCE, ScalarFn, CAST ──

    #[test]
    fn select_distinct() {
        let stmt = parse_ok("SELECT DISTINCT col FROM my_ds");
        let Statement::Select(s) = stmt else { panic!() };
        assert!(s.distinct);
        let SelectColumns::Named(cols) = s.columns else {
            panic!()
        };
        assert!(matches!(&cols[0], SelectExpr::Column(c) if c == "col"));
    }

    #[test]
    fn select_distinct_star() {
        let stmt = parse_ok("SELECT DISTINCT * FROM my_ds");
        let Statement::Select(s) = stmt else { panic!() };
        assert!(s.distinct);
        assert!(matches!(s.columns, SelectColumns::All));
    }

    #[test]
    fn select_union() {
        let stmt = parse_ok("SELECT a FROM t1 UNION SELECT a FROM t2");
        let Statement::Select(s) = stmt else { panic!() };
        assert!(s.union.is_some());
        let (kind, _) = s.union.unwrap();
        assert!(matches!(kind, SetOpKind::Union));
    }

    #[test]
    fn select_union_all() {
        let stmt = parse_ok("SELECT a FROM t1 UNION ALL SELECT a FROM t2");
        let Statement::Select(s) = stmt else { panic!() };
        let (kind, _) = s.union.unwrap();
        assert!(matches!(kind, SetOpKind::UnionAll));
    }

    #[test]
    fn cte_basic() {
        let stmt = parse_ok("WITH cte AS (SELECT * FROM employees) SELECT * FROM cte");
        let Statement::Select(s) = stmt else { panic!() };
        assert_eq!(s.ctes.len(), 1);
        assert_eq!(s.ctes[0].0, "cte");
        assert!(matches!(&s.source, DatasetSource::Named(n) if n == "cte"));
    }

    #[test]
    fn cte_multiple() {
        let stmt =
            parse_ok("WITH a AS (SELECT * FROM t1), b AS (SELECT * FROM t2) SELECT * FROM a");
        let Statement::Select(s) = stmt else { panic!() };
        assert_eq!(s.ctes.len(), 2);
        assert_eq!(s.ctes[0].0, "a");
        assert_eq!(s.ctes[1].0, "b");
    }

    #[test]
    fn case_when_simple() {
        let stmt =
            parse_ok("SELECT CASE WHEN age > 30 THEN \"senior\" ELSE \"junior\" END FROM emp");
        let Statement::Select(s) = stmt else { panic!() };
        let SelectColumns::Named(cols) = s.columns else {
            panic!()
        };
        assert!(
            matches!(&cols[0], SelectExpr::Computed { expr, .. } if matches!(expr.as_ref(), Expr::Case { .. }))
        );
    }

    #[test]
    fn case_when_no_else() {
        let stmt = parse_ok("SELECT CASE WHEN x > 0 THEN \"pos\" END FROM t");
        let Statement::Select(s) = stmt else { panic!() };
        let SelectColumns::Named(cols) = s.columns else {
            panic!()
        };
        assert!(matches!(&cols[0], SelectExpr::Computed { .. }));
    }

    #[test]
    fn coalesce_expr() {
        let stmt = parse_ok("SELECT COALESCE(a, b, 0) FROM t");
        let Statement::Select(s) = stmt else { panic!() };
        let SelectColumns::Named(cols) = s.columns else {
            panic!()
        };
        assert!(
            matches!(&cols[0], SelectExpr::Computed { expr, .. } if matches!(expr.as_ref(), Expr::Coalesce(_)))
        );
    }

    #[test]
    fn nullif_expr() {
        let stmt = parse_ok("SELECT NULLIF(a, 0) FROM t");
        let Statement::Select(s) = stmt else { panic!() };
        let SelectColumns::Named(cols) = s.columns else {
            panic!()
        };
        assert!(
            matches!(&cols[0], SelectExpr::Computed { expr, .. } if matches!(expr.as_ref(), Expr::Nullif(..)))
        );
    }

    #[test]
    fn scalar_fn_upper() {
        let stmt = parse_ok("SELECT UPPER(name) FROM emp");
        let Statement::Select(s) = stmt else { panic!() };
        let SelectColumns::Named(cols) = s.columns else {
            panic!()
        };
        assert!(
            matches!(&cols[0], SelectExpr::Computed { expr, .. } if matches!(expr.as_ref(), Expr::ScalarFn { func: ScalarFnKind::Upper, .. }))
        );
    }

    #[test]
    fn scalar_fn_lower() {
        let stmt = parse_ok("SELECT LOWER(name) AS lower_name FROM emp");
        let Statement::Select(s) = stmt else { panic!() };
        let SelectColumns::Named(cols) = s.columns else {
            panic!()
        };
        assert!(
            matches!(&cols[0], SelectExpr::Computed { alias: Some(a), .. } if a == "lower_name")
        );
    }

    #[test]
    fn scalar_fn_length() {
        let stmt = parse_ok("SELECT LENGTH(name) FROM emp");
        let Statement::Select(s) = stmt else { panic!() };
        let SelectColumns::Named(cols) = s.columns else {
            panic!()
        };
        assert!(
            matches!(&cols[0], SelectExpr::Computed { expr, .. } if matches!(expr.as_ref(), Expr::ScalarFn { func: ScalarFnKind::Length, .. }))
        );
    }

    #[test]
    fn cast_expr() {
        let stmt = parse_ok("SELECT CAST(age AS FLOAT) AS f_age FROM emp");
        let Statement::Select(s) = stmt else { panic!() };
        let SelectColumns::Named(cols) = s.columns else {
            panic!()
        };
        assert!(
            matches!(&cols[0], SelectExpr::Computed { expr, alias: Some(a) } if a == "f_age" && matches!(expr.as_ref(), Expr::Cast { to: CastTarget::Float, .. }))
        );
    }

    #[test]
    fn cast_to_int() {
        let stmt = parse_ok("SELECT CAST(score AS INT) FROM t");
        let Statement::Select(s) = stmt else { panic!() };
        let SelectColumns::Named(cols) = s.columns else {
            panic!()
        };
        assert!(
            matches!(&cols[0], SelectExpr::Computed { expr, .. } if matches!(expr.as_ref(), Expr::Cast { to: CastTarget::Int, .. }))
        );
    }

    #[test]
    fn window_row_number() {
        let stmt = parse_ok(
            "SELECT ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC) AS rn FROM emp",
        );
        let Statement::Select(s) = stmt else { panic!() };
        let SelectColumns::Named(cols) = s.columns else {
            panic!()
        };
        assert!(
            matches!(&cols[0], SelectExpr::Window { func: WindowFunc::RowNumber, alias, .. } if alias == "rn")
        );
    }

    #[test]
    fn window_rank() {
        let stmt =
            parse_ok("SELECT RANK() OVER (PARTITION BY dept ORDER BY salary) AS rnk FROM emp");
        let Statement::Select(s) = stmt else { panic!() };
        let SelectColumns::Named(cols) = s.columns else {
            panic!()
        };
        assert!(matches!(
            &cols[0],
            SelectExpr::Window {
                func: WindowFunc::Rank,
                ..
            }
        ));
    }

    #[test]
    fn window_dense_rank() {
        let stmt = parse_ok("SELECT DENSE_RANK() OVER (ORDER BY score DESC) AS dr FROM t");
        let Statement::Select(s) = stmt else { panic!() };
        let SelectColumns::Named(cols) = s.columns else {
            panic!()
        };
        assert!(matches!(
            &cols[0],
            SelectExpr::Window {
                func: WindowFunc::DenseRank,
                ..
            }
        ));
    }

    #[test]
    fn window_lag() {
        let stmt = parse_ok("SELECT LAG(salary, 1) OVER (ORDER BY date) AS prev_sal FROM emp");
        let Statement::Select(s) = stmt else { panic!() };
        let SelectColumns::Named(cols) = s.columns else {
            panic!()
        };
        assert!(
            matches!(&cols[0], SelectExpr::Window { func: WindowFunc::Lag { col, offset: 1 }, .. } if col == "salary")
        );
    }

    #[test]
    fn window_lead() {
        let stmt = parse_ok("SELECT LEAD(salary, 2) OVER (ORDER BY date) AS next_sal FROM emp");
        let Statement::Select(s) = stmt else { panic!() };
        let SelectColumns::Named(cols) = s.columns else {
            panic!()
        };
        assert!(
            matches!(&cols[0], SelectExpr::Window { func: WindowFunc::Lead { col, offset: 2 }, .. } if col == "salary")
        );
    }

    #[test]
    fn window_sum_over() {
        let stmt =
            parse_ok("SELECT SUM(amount) OVER (PARTITION BY customer) AS running_sum FROM orders");
        let Statement::Select(s) = stmt else { panic!() };
        let SelectColumns::Named(cols) = s.columns else {
            panic!()
        };
        assert!(matches!(
            &cols[0],
            SelectExpr::Window {
                func: WindowFunc::Sum(_),
                ..
            }
        ));
    }

    #[test]
    fn window_avg_over() {
        let stmt = parse_ok("SELECT AVG(score) OVER (PARTITION BY dept) AS dept_avg FROM t");
        let Statement::Select(s) = stmt else { panic!() };
        let SelectColumns::Named(cols) = s.columns else {
            panic!()
        };
        assert!(matches!(
            &cols[0],
            SelectExpr::Window {
                func: WindowFunc::Avg(_),
                ..
            }
        ));
    }

    #[test]
    fn select_no_distinct_by_default() {
        let stmt = parse_ok("SELECT col FROM ds");
        let Statement::Select(s) = stmt else { panic!() };
        assert!(!s.distinct);
        assert!(s.union.is_none());
        assert!(s.ctes.is_empty());
    }
}
