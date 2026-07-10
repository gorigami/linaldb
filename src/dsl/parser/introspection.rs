use super::{ParseError, Parser};
use crate::dsl::ast::*;
use crate::dsl::lexer::Token;

impl Parser {
    // SHOW <target>
    pub(super) fn parse_show(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Show)?;

        let target = match self.peek() {
            Some(Token::All) => {
                self.advance();
                match self.peek() {
                    Some(Token::Datasets) => {
                        self.advance();
                        ShowTarget::AllDatasets
                    }
                    Some(Token::Databases) => {
                        self.advance();
                        ShowTarget::AllDatabases
                    }
                    Some(Token::Tensors) => {
                        self.advance();
                        ShowTarget::All
                    }
                    _ => ShowTarget::All,
                }
            }
            Some(Token::Databases) => {
                self.advance();
                ShowTarget::AllDatabases
            }
            Some(Token::Schema) => {
                self.advance();
                ShowTarget::Schema(self.eat_ident()?)
            }
            Some(Token::Shape) => {
                self.advance();
                ShowTarget::Shape(self.eat_ident()?)
            }
            Some(Token::Lineage) => {
                self.advance();
                ShowTarget::Lineage(self.eat_ident()?)
            }
            Some(Token::Indexes) => {
                self.advance();
                let ds = if self.at_any_ident() {
                    Some(self.eat_ident()?)
                } else {
                    None
                };
                ShowTarget::Indexes(ds)
            }
            Some(Token::Dataset) => {
                self.advance();
                match self.peek() {
                    Some(Token::Metadata) => {
                        self.advance();
                        ShowTarget::DatasetMetadata(self.eat_ident()?)
                    }
                    Some(Token::Versions) => {
                        self.advance();
                        ShowTarget::DatasetVersions(self.eat_ident()?)
                    }
                    _ => return Err(self.unexpected("METADATA or VERSIONS after SHOW DATASET")),
                }
            }
            Some(Token::Pipelines) => {
                self.advance();
                ShowTarget::Pipelines
            }
            Some(Token::Str(_)) => ShowTarget::StringLiteral(self.eat_str()?),
            Some(Token::Ident(_)) => ShowTarget::Named(self.eat_ident()?),
            _ => return Err(self.unexpected("a SHOW target")),
        };

        Ok(Statement::Show(ShowStmt { target }))
    }

    // EXPLAIN [PLAN] DATASET <name>
    // EXPLAIN [PLAN] SEARCH <ds> ON <col> QUERY <q> LIMIT <k>
    // EXPLAIN [PLAN] SELECT ...
    // EXPLAIN <bare_ident>
    pub(super) fn parse_explain(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Explain)?;
        if self.at_ident("PLAN") {
            self.advance();
        }
        let target = match self.peek().cloned() {
            Some(Token::Dataset) => {
                self.advance();
                let name = self.eat_ident()?;
                if self.at(&Token::From) {
                    self.advance();
                    let source = self.eat_ident()?;
                    let from = self.parse_dataset_from_clause(source)?;
                    ExplainTarget::DatasetQuery { name, from }
                } else {
                    ExplainTarget::Dataset(name)
                }
            }
            Some(Token::Search) => {
                let stmt = self.parse_search()?;
                let inner = if let Statement::Search(s) = stmt {
                    s
                } else {
                    unreachable!()
                };
                ExplainTarget::Search(inner)
            }
            Some(Token::Select) => {
                let stmt = self.parse_select()?;
                let inner = if let Statement::Select(s) = stmt {
                    s
                } else {
                    unreachable!()
                };
                ExplainTarget::Select(inner)
            }
            Some(Token::Ident(_)) => ExplainTarget::Dataset(self.eat_ident()?),
            _ => return Err(self.error("EXPLAIN expects DATASET, SEARCH, SELECT, or a name")),
        };
        Ok(Statement::Explain(ExplainStmt { target }))
    }

    // AUDIT DATASET <name>  or  AUDIT <name>
    pub(super) fn parse_audit(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Audit)?;
        if self.at(&Token::Dataset) {
            self.advance();
        }
        let target = self.eat_ident()?;
        Ok(Statement::Audit(AuditStmt { target }))
    }

    // DELIVER <dataset> [TO <path>]
    pub(super) fn parse_deliver(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Deliver)?;
        let dataset = self.eat_ident()?;
        let path = if self.at(&Token::To) {
            self.advance();
            Some(self.eat_str()?)
        } else {
            None
        };
        Ok(Statement::Deliver(DeliverStmt { dataset, path }))
    }
}
