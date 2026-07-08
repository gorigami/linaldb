use super::{ParseError, Parser};
use crate::dsl::ast::*;
use crate::dsl::lexer::Token;

impl Parser {
    // SAVE TENSOR <name> [TO <path>]
    // SAVE DATASET <name> [TO <path>]
    pub(super) fn parse_save(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Save)?;
        let kind = self.parse_persist_kind()?;
        let name = self.eat_ident()?;
        let path = if self.at(&Token::To) {
            self.advance();
            Some(self.eat_str()?)
        } else {
            None
        };
        Ok(Statement::Save(SaveStmt { kind, name, path }))
    }

    // LOAD TENSOR <name> [FROM <path>]
    // LOAD DATASET <name> [FROM <path>]
    pub(super) fn parse_load(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Load)?;
        let kind = self.parse_persist_kind()?;
        let name = self.eat_ident()?;
        let path = if self.at(&Token::From) {
            self.advance();
            Some(self.eat_str()?)
        } else {
            None
        };
        Ok(Statement::Load(LoadStmt { kind, name, path }))
    }

    pub(super) fn parse_persist_kind(&mut self) -> Result<PersistKind, ParseError> {
        match self.peek() {
            Some(Token::Tensor) => {
                self.advance();
                Ok(PersistKind::Tensor)
            }
            Some(Token::Dataset) => {
                self.advance();
                Ok(PersistKind::Dataset)
            }
            _ => Err(self.unexpected("TENSOR or DATASET")),
        }
    }

    // LIST TENSORS | LIST DATASETS | LIST DATASET VERSIONS <name>
    pub(super) fn parse_list(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::List)?;
        let target = match self.peek() {
            Some(Token::Tensors) => {
                self.advance();
                ListTarget::Tensors
            }
            Some(Token::Datasets) => {
                self.advance();
                ListTarget::Datasets
            }
            Some(Token::Dataset) => {
                self.advance();
                if self.at(&Token::Versions) {
                    self.advance();
                    ListTarget::DatasetVersions(self.eat_ident()?)
                } else if self.at_ident("PACKAGES") {
                    self.advance();
                    ListTarget::DatasetPackages
                } else {
                    ListTarget::Datasets
                }
            }
            _ => return Err(self.unexpected("TENSORS, DATASETS, or DATASET VERSIONS")),
        };
        Ok(Statement::List(ListStmt { target }))
    }

    // IMPORT DATASET FROM <path> [AS <name>]
    // IMPORT CSV FROM <path> [AS <name>]
    pub(super) fn parse_import(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Import)?;
        if self.at_ident("CSV") {
            self.advance();
            self.eat(&Token::From)?;
            let path = self.eat_str()?;
            let name = if self.at(&Token::As) {
                self.advance();
                Some(self.eat_ident()?)
            } else {
                None
            };
            return Ok(Statement::ImportCsv(ImportCsvStmt { path, name }));
        }
        self.eat(&Token::Dataset)?;
        self.eat(&Token::From)?;
        let path = self.eat_str()?;
        let name = if self.at(&Token::As) {
            self.advance();
            Some(self.eat_ident()?)
        } else {
            None
        };
        Ok(Statement::Import(ImportStmt {
            ephemeral: false,
            path,
            name,
        }))
    }

    // EXPORT [CSV] <name> TO <path>
    pub(super) fn parse_export(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Export)?;
        if self.at_ident("CSV") {
            self.advance();
        }
        let name = self.eat_ident()?;
        self.eat(&Token::To)?;
        let path = self.eat_str()?;
        Ok(Statement::Export(ExportStmt { name, path }))
    }

    // USE <database_name>
    // USE DATASET FROM <path> [AS <name>]  (ephemeral import)
    pub(super) fn parse_use(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Use)?;
        match self.peek() {
            Some(Token::Dataset) => {
                self.advance();
                self.eat(&Token::From)?;
                let path = self.eat_str()?;
                let name = if self.at(&Token::As) {
                    self.advance();
                    Some(self.eat_ident()?)
                } else {
                    None
                };
                Ok(Statement::Import(ImportStmt {
                    ephemeral: true,
                    path,
                    name,
                }))
            }
            Some(Token::Ident(_)) => Ok(Statement::UseDatabase(UseDatabaseStmt {
                name: self.eat_ident()?,
            })),
            _ => Err(self.unexpected("a database name or DATASET FROM")),
        }
    }
}
