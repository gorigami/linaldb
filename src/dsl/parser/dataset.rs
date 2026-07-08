use super::{ParseError, Parser};
use crate::dsl::ast::*;
use crate::dsl::lexer::Token;

impl Parser {
    // Shared helper: parse everything after `FROM <source>` for DATASET and EXPLAIN.
    pub(super) fn parse_dataset_from_clause(
        &mut self,
        source: String,
    ) -> Result<DatasetFromClause, ParseError> {
        let mut filter = None;
        let mut select = None;
        let mut group_by = vec![];
        let mut having = None;
        let mut order_by = None;
        let mut limit = None;
        let mut offset = None;

        loop {
            match self.peek() {
                Some(Token::Filter) | Some(Token::Where) => {
                    self.advance();
                    filter = Some(self.parse_expr()?);
                }
                Some(Token::Select) => {
                    self.advance();
                    let mut exprs = vec![];
                    loop {
                        exprs.push(self.parse_select_expr()?);
                        if self.at(&Token::Comma) {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                    select = Some(exprs);
                }
                Some(Token::Group) => {
                    self.advance();
                    self.eat(&Token::By)?;
                    let mut cols = vec![];
                    loop {
                        cols.push(self.eat_ident()?);
                        if self.at(&Token::Comma) {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                    group_by = cols;
                }
                Some(Token::Having) => {
                    self.advance();
                    having = Some(self.parse_expr()?);
                }
                Some(Token::Order) => {
                    self.advance();
                    self.eat(&Token::By)?;
                    let mut columns = vec![];
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
                        columns.push((col, ascending));
                        if self.at(&Token::Comma) {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                    order_by = Some(OrderByClause { columns });
                }
                Some(Token::Limit) => {
                    self.advance();
                    limit = Some(self.eat_usize()?);
                    if self.at(&Token::Offset) {
                        self.advance();
                        offset = Some(self.eat_usize()?);
                    }
                }
                Some(Token::Offset) => {
                    self.advance();
                    offset = Some(self.eat_usize()?);
                }
                _ => break,
            }
        }

        Ok(DatasetFromClause {
            source,
            filter,
            select,
            group_by,
            having,
            order_by,
            limit,
            offset,
        })
    }

    // DATASET <name> COLUMNS (...) | DATASET <name> FROM <src>
    pub(super) fn parse_create_dataset(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Dataset)?;
        let name = self.eat_ident()?;

        match self.peek() {
            Some(Token::Columns) => {
                self.advance();
                let columns = self.parse_column_list()?;
                Ok(Statement::CreateDataset(CreateDatasetStmt {
                    name,
                    columns,
                    from: None,
                }))
            }
            Some(Token::From) => {
                self.advance();
                let source = self.eat_ident()?;
                let from = self.parse_dataset_from_clause(source)?;
                Ok(Statement::CreateDataset(CreateDatasetStmt {
                    name,
                    columns: vec![],
                    from: Some(from),
                }))
            }
            Some(Token::Add) => {
                self.advance();
                self.eat(&Token::Column)?;
                let col_name = self.eat_ident()?;
                if self.at(&Token::Eq) {
                    self.advance();
                    let expr = self.parse_expr()?;
                    let lazy = if self.at(&Token::Lazy) {
                        self.advance();
                        true
                    } else {
                        false
                    };
                    Ok(Statement::AlterDataset(AlterDatasetStmt {
                        dataset: name,
                        operation: AlterOp::AddComputedColumn {
                            name: col_name,
                            expr: Box::new(expr),
                            lazy,
                        },
                    }))
                } else {
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
                    Ok(Statement::AlterDataset(AlterDatasetStmt {
                        dataset: name,
                        operation: AlterOp::AddColumn(ColumnDef {
                            name: col_name,
                            col_type,
                            nullable,
                            default_val,
                        }),
                    }))
                }
            }
            _ => Err(self.unexpected("COLUMNS, FROM, or ADD COLUMN after DATASET name")),
        }
    }

    pub(super) fn parse_filter_value(&mut self) -> Result<FilterValue, ParseError> {
        match self.peek() {
            Some(Token::Int(_)) => {
                if let Some(Token::Int(n)) = self.advance() {
                    return Ok(FilterValue::Int(n));
                }
                unreachable!()
            }
            Some(Token::Float(_)) => {
                if let Some(Token::Float(f)) = self.advance() {
                    return Ok(FilterValue::Float(f));
                }
                unreachable!()
            }
            Some(Token::Str(_)) => {
                if let Some(Token::Str(s)) = self.advance() {
                    return Ok(FilterValue::Str(s));
                }
                unreachable!()
            }
            Some(Token::Minus) => {
                self.advance();
                match self.peek() {
                    Some(Token::Int(_)) => {
                        if let Some(Token::Int(n)) = self.advance() {
                            return Ok(FilterValue::Int(-n));
                        }
                        unreachable!()
                    }
                    Some(Token::Float(_)) => {
                        if let Some(Token::Float(f)) = self.advance() {
                            return Ok(FilterValue::Float(-f));
                        }
                        unreachable!()
                    }
                    _ => Err(self.unexpected("number after `-`")),
                }
            }
            Some(Token::Ident(_)) => {
                if self.at_ident("true") {
                    self.advance();
                    Ok(FilterValue::Bool(true))
                } else if self.at_ident("false") {
                    self.advance();
                    Ok(FilterValue::Bool(false))
                } else {
                    Err(self.unexpected("filter value (integer, float, string, or boolean)"))
                }
            }
            _ => Err(self.unexpected("filter value (integer, float, string, or boolean)")),
        }
    }

    // ALTER DATASET <name> ADD COLUMN <col> = <expr> [LAZY]
    // ALTER DATASET <name> ADD COLUMN <col>: TYPE [DEFAULT val]
    pub(super) fn parse_alter(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Alter)?;
        self.eat(&Token::Dataset)?;
        let dataset = self.eat_ident()?;
        self.eat(&Token::Add)?;
        self.eat(&Token::Column)?;
        let col_name = self.eat_ident()?;
        if self.at(&Token::Eq) {
            self.advance();
            let expr = self.parse_expr()?;
            let lazy = if self.at(&Token::Lazy) {
                self.advance();
                true
            } else {
                false
            };
            Ok(Statement::AlterDataset(AlterDatasetStmt {
                dataset,
                operation: AlterOp::AddComputedColumn {
                    name: col_name,
                    expr: Box::new(expr),
                    lazy,
                },
            }))
        } else {
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
            Ok(Statement::AlterDataset(AlterDatasetStmt {
                dataset,
                operation: AlterOp::AddColumn(ColumnDef {
                    name: col_name,
                    col_type,
                    nullable,
                    default_val,
                }),
            }))
        }
    }

    // INSERT INTO <dataset> VALUES (v1, v2, ...) | INSERT INTO <dataset> (col = val, ...)
    pub(super) fn parse_insert_into(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Insert)?;
        self.eat(&Token::Into)?;
        let dataset = self.eat_ident()?;

        let row = if self.at(&Token::Values) {
            self.advance();
            self.eat(&Token::LParen)?;
            let mut vals = Vec::new();
            while !self.at(&Token::RParen) && !self.eof() {
                vals.push(self.parse_insert_value()?);
                if self.at(&Token::Comma) {
                    self.advance();
                }
            }
            self.eat(&Token::RParen)?;
            InsertRow::Positional(vals)
        } else {
            self.eat(&Token::LParen)?;
            let mut named = Vec::new();
            while !self.at(&Token::RParen) && !self.eof() {
                let col = self.eat_ident()?;
                self.eat(&Token::Eq)?;
                let val = self.parse_insert_value()?;
                named.push((col, val));
                if self.at(&Token::Comma) {
                    self.advance();
                }
            }
            self.eat(&Token::RParen)?;
            InsertRow::Named(named)
        };

        Ok(Statement::InsertInto(InsertIntoStmt { dataset, row }))
    }

    pub(super) fn parse_insert_value(&mut self) -> Result<InsertValue, ParseError> {
        match self.peek() {
            Some(Token::Null) => {
                self.advance();
                Ok(InsertValue::Null)
            }
            Some(Token::Str(_)) => Ok(InsertValue::Text(self.eat_str()?)),
            Some(Token::Float(_)) | Some(Token::Int(_)) | Some(Token::Minus) => {
                Ok(InsertValue::Scalar(self.eat_number()?))
            }
            Some(Token::Ident(_)) => {
                if self.at_ident("true") {
                    self.advance();
                    Ok(InsertValue::Bool(true))
                } else if self.at_ident("false") {
                    self.advance();
                    Ok(InsertValue::Bool(false))
                } else {
                    Ok(InsertValue::TensorRef(self.eat_ident()?))
                }
            }
            Some(Token::LBracket) => {
                self.advance();
                if self.at(&Token::LBracket) {
                    let mut rows: Vec<Vec<f64>> = Vec::new();
                    while !self.at(&Token::RBracket) && !self.eof() {
                        self.eat(&Token::LBracket)?;
                        let mut row: Vec<f64> = Vec::new();
                        while !self.at(&Token::RBracket) && !self.eof() {
                            row.push(self.eat_number()?);
                            if self.at(&Token::Comma) {
                                self.advance();
                            }
                        }
                        self.eat(&Token::RBracket)?;
                        rows.push(row);
                        if self.at(&Token::Comma) {
                            self.advance();
                        }
                    }
                    self.eat(&Token::RBracket)?;
                    Ok(InsertValue::Matrix(rows))
                } else {
                    let mut vals: Vec<f64> = Vec::new();
                    while !self.at(&Token::RBracket) && !self.eof() {
                        vals.push(self.eat_number()?);
                        if self.at(&Token::Comma) {
                            self.advance();
                        }
                    }
                    self.eat(&Token::RBracket)?;
                    Ok(InsertValue::Vector(vals))
                }
            }
            _ => Err(self.unexpected(
                "a value (number, string, identifier, NULL, or vector/matrix literal)",
            )),
        }
    }

    // SELECT [* | col, ...] FROM <dataset|subquery> [JOIN ...] [WHERE expr]
    //        [GROUP BY ...] [HAVING expr] [ORDER BY col [ASC|DESC], ...] [LIMIT n [OFFSET m]]
    pub(super) fn parse_select(&mut self) -> Result<Statement, ParseError> {
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

        self.eat(&Token::From)?;

        // FROM (SELECT ...) AS alias  OR  FROM dataset_name
        let source = if self.at(&Token::LParen) {
            self.advance();
            let inner_stmt = self.parse_select()?;
            self.eat(&Token::RParen)?;
            self.eat(&Token::As)?;
            let alias = self.eat_ident()?;
            if let Statement::Select(inner) = inner_stmt {
                DatasetSource::Subquery {
                    query: Box::new(inner),
                    alias,
                }
            } else {
                return Err(self.error("Expected SELECT inside subquery"));
            }
        } else {
            DatasetSource::Named(self.eat_ident()?)
        };

        // Parse zero or more JOIN clauses
        let mut joins = Vec::new();
        while matches!(
            self.peek(),
            Some(Token::Join)
                | Some(Token::Inner)
                | Some(Token::Left)
                | Some(Token::Right)
                | Some(Token::Full)
        ) {
            joins.push(self.parse_join_clause()?);
        }

        let mut filter = None;
        let mut group_by = Vec::new();
        let mut having = None;
        let mut order_by = None;
        let mut limit = None;
        let mut offset = None;

        loop {
            match self.peek() {
                Some(Token::Where) | Some(Token::Filter) => {
                    self.advance();
                    filter = Some(self.parse_expr()?);
                }
                Some(Token::Group) => {
                    self.advance();
                    if self.at(&Token::By) {
                        self.advance();
                    }
                    group_by.push(self.eat_ident()?);
                    while self.at(&Token::Comma) {
                        self.advance();
                        group_by.push(self.eat_ident()?);
                    }
                }
                Some(Token::Having) => {
                    self.advance();
                    having = Some(self.parse_expr()?);
                }
                Some(Token::Order) => {
                    self.advance();
                    if self.at(&Token::By) {
                        self.advance();
                    }
                    let mut columns = vec![];
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
                        columns.push((col, ascending));
                        if self.at(&Token::Comma) {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                    order_by = Some(OrderByClause { columns });
                }
                Some(Token::Limit) => {
                    self.advance();
                    limit = Some(self.eat_usize()?);
                    if self.at(&Token::Offset) {
                        self.advance();
                        offset = Some(self.eat_usize()?);
                    }
                }
                Some(Token::Offset) => {
                    self.advance();
                    offset = Some(self.eat_usize()?);
                }
                _ => break,
            }
        }

        Ok(Statement::Select(SelectStmt {
            source,
            joins,
            columns,
            filter,
            group_by,
            having,
            order_by,
            limit,
            offset,
        }))
    }

    // [INNER | LEFT [OUTER] | RIGHT [OUTER] | FULL [OUTER]] JOIN <dataset> ON <col> = <col>
    fn parse_join_clause(&mut self) -> Result<JoinClause, ParseError> {
        let kind = if self.at(&Token::Left) {
            self.advance();
            if self.at(&Token::Outer) {
                self.advance();
            }
            self.eat(&Token::Join)?;
            JoinKind::Left
        } else if self.at(&Token::Right) {
            self.advance();
            if self.at(&Token::Outer) {
                self.advance();
            }
            self.eat(&Token::Join)?;
            JoinKind::Right
        } else if self.at(&Token::Full) {
            self.advance();
            if self.at(&Token::Outer) {
                self.advance();
            }
            self.eat(&Token::Join)?;
            JoinKind::Full
        } else {
            if self.at(&Token::Inner) {
                self.advance();
            }
            self.eat(&Token::Join)?;
            JoinKind::Inner
        };

        let right_dataset = self.eat_ident()?;
        self.eat(&Token::On)?;

        // ON <left_ref> = <right_ref>
        // Each ref is either `col` or `table.col`
        let left_col = self.parse_join_col_ref()?;
        self.eat(&Token::Eq)?;
        let right_col = self.parse_join_col_ref()?;

        Ok(JoinClause {
            kind,
            dataset: right_dataset,
            left_col,
            right_col,
        })
    }

    // Parse `col` or `table.col`, returning only the column part.
    fn parse_join_col_ref(&mut self) -> Result<String, ParseError> {
        let first = self.eat_ident()?;
        if self.at(&Token::Dot) {
            self.advance();
            Ok(self.eat_ident()?)
        } else {
            Ok(first)
        }
    }

    // UPDATE <dataset> SET col = expr [, col = expr]* [WHERE expr]
    pub(super) fn parse_update(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Update)?;
        let dataset = self.eat_ident()?;
        self.eat(&Token::Set)?;

        let mut assignments = Vec::new();
        loop {
            let col = self.eat_ident()?;
            self.eat(&Token::Eq)?;
            let expr = self.parse_expr()?;
            assignments.push((col, expr));
            if self.at(&Token::Comma) {
                self.advance();
            } else {
                break;
            }
        }

        let filter = if matches!(self.peek(), Some(Token::Where) | Some(Token::Filter)) {
            self.advance();
            Some(self.parse_expr()?)
        } else {
            None
        };

        Ok(Statement::Update(UpdateStmt {
            dataset,
            assignments,
            filter,
        }))
    }

    // DELETE FROM <dataset> [WHERE expr]
    pub(super) fn parse_delete(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Delete)?;
        self.eat(&Token::From)?;
        let dataset = self.eat_ident()?;

        let filter = if matches!(self.peek(), Some(Token::Where) | Some(Token::Filter)) {
            self.advance();
            Some(self.parse_expr()?)
        } else {
            None
        };

        Ok(Statement::Delete(DeleteStmt { dataset, filter }))
    }

    // MATERIALIZE <name> | MATERIALIZE <dataset>.<column>
    pub(super) fn parse_materialize(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Materialize)?;
        let name = self.eat_ident()?;
        let target = if self.at(&Token::Dot) {
            self.advance();
            let col = self.eat_ident()?;
            format!("{}.{}", name, col)
        } else {
            name
        };
        Ok(Statement::Materialize(MaterializeStmt { target }))
    }

    // SELECT column list item: plain column or aggregate call
    pub(super) fn parse_select_expr(&mut self) -> Result<SelectExpr, ParseError> {
        match self.peek() {
            Some(Token::Sum) => {
                self.advance();
                self.parse_agg_call(AggFuncAst::Sum)
            }
            Some(Token::Ident(s)) if s == "AVG" => {
                self.advance();
                self.parse_agg_call(AggFuncAst::Avg)
            }
            Some(Token::Ident(s)) if s == "COUNT" => {
                self.advance();
                self.parse_agg_call(AggFuncAst::Count)
            }
            Some(Token::Ident(s)) if s == "MIN" => {
                self.advance();
                self.parse_agg_call(AggFuncAst::Min)
            }
            Some(Token::Ident(s)) if s == "MAX" => {
                self.advance();
                self.parse_agg_call(AggFuncAst::Max)
            }
            _ => Ok(SelectExpr::Column(self.eat_ident()?)),
        }
    }

    pub(super) fn parse_agg_call(&mut self, func: AggFuncAst) -> Result<SelectExpr, ParseError> {
        self.eat(&Token::LParen)?;
        let expr = if self.at(&Token::Star) {
            self.advance();
            Expr::Ref("*".to_string())
        } else {
            self.parse_expr()?
        };
        self.eat(&Token::RParen)?;
        Ok(SelectExpr::Aggregate {
            func,
            expr: Box::new(expr),
        })
    }

    // SEARCH <dataset> ON <column> QUERY <tensor_name|[vector]> LIMIT <k> [INTO <target>]
    pub(super) fn parse_search(&mut self) -> Result<Statement, ParseError> {
        self.eat(&Token::Search)?;
        let first = self.eat_ident()?;

        if self.at(&Token::From) {
            // Legacy syntax: SEARCH target FROM source QUERY [...] ON col K=k
            self.advance();
            let source = self.eat_ident()?;
            if !self.at_ident("QUERY") {
                return Err(self.error("Expected QUERY after source in SEARCH ... FROM"));
            }
            self.advance();
            let query = if self.at(&Token::LBracket) {
                SearchQuery::Inline(self.parse_f64_list()?)
            } else {
                SearchQuery::TensorRef(self.eat_ident()?)
            };
            if !self.at(&Token::On) {
                return Err(self.error("Expected ON after query vector in SEARCH ... FROM"));
            }
            self.advance();
            let column = self.eat_ident()?;
            self.eat_ident()?; // consume "K"
            self.eat(&Token::Eq)?;
            let top_k = self.eat_usize()?;
            Ok(Statement::Search(SearchStmt {
                dataset: source,
                column,
                query,
                top_k,
                target: Some(first),
            }))
        } else if self.at(&Token::Where) {
            // WHERE syntax: SEARCH source WHERE col ~= [...] LIMIT k
            self.advance();
            let column = self.eat_ident()?;
            self.eat(&Token::ApproxEq)?;
            let query = SearchQuery::Inline(self.parse_f64_list()?);
            self.eat(&Token::Limit)?;
            let top_k = self.eat_usize()?;
            Ok(Statement::Search(SearchStmt {
                dataset: first,
                column,
                query,
                top_k,
                target: None,
            }))
        } else {
            // Modern syntax: SEARCH dataset ON col QUERY [...|name] LIMIT k [INTO target]
            if !self.at(&Token::On) {
                return Err(self.error("Expected FROM, ON, or WHERE after dataset name in SEARCH"));
            }
            self.advance();
            let column = self.eat_ident()?;
            if !self.at_ident("QUERY") {
                return Err(self.error("Expected QUERY after column name in SEARCH"));
            }
            self.advance();
            let query = if self.at(&Token::LBracket) {
                SearchQuery::Inline(self.parse_f64_list()?)
            } else {
                SearchQuery::TensorRef(self.eat_ident()?)
            };
            self.eat(&Token::Limit)?;
            let top_k = self.eat_usize()?;
            let target = if self.at(&Token::Into) {
                self.advance();
                Some(self.eat_ident()?)
            } else {
                None
            };
            Ok(Statement::Search(SearchStmt {
                dataset: first,
                column,
                query,
                top_k,
                target,
            }))
        }
    }
}
