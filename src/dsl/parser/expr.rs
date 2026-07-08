use super::{ParseError, Parser};
use crate::dsl::ast::*;
use crate::dsl::lexer::Token;

impl Parser {
    pub(super) fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        self.parse_pratt(0)
    }

    /// Pratt parser — `min_bp` is the minimum left binding power to continue.
    pub(super) fn parse_pratt(&mut self, min_bp: u8) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_expr_atom()?;

        loop {
            // Postfix: field access, subscript, IS NULL / IS NOT NULL
            match self.peek() {
                Some(Token::Dot) => {
                    self.advance();
                    let field = self.eat_ident()?;
                    lhs = Expr::Field {
                        base: Box::new(lhs),
                        field,
                    };
                    continue;
                }
                Some(Token::LBracket) => {
                    self.advance();
                    let indices = self.parse_index_specs()?;
                    self.eat(&Token::RBracket)?;
                    lhs = Expr::Index {
                        base: Box::new(lhs),
                        indices,
                    };
                    continue;
                }
                Some(Token::Is) => {
                    self.advance();
                    if self.at(&Token::Not) {
                        self.advance();
                        self.eat(&Token::Null)?;
                        lhs = Expr::IsNotNull(Box::new(lhs));
                    } else {
                        self.eat(&Token::Null)?;
                        lhs = Expr::IsNull(Box::new(lhs));
                    }
                    continue;
                }
                _ => {}
            }

            // Infix operators with precedence (lowest → highest):
            // OR(1) < AND(3) < comparison(5) < +/-(7) < *//,(9)
            let (left_bp, right_bp) = match self.peek() {
                Some(Token::Or) => (1u8, 2u8),
                Some(Token::And) => (3, 4),
                Some(Token::Eq) | Some(Token::NotEq) | Some(Token::Gt) | Some(Token::Lt)
                | Some(Token::GtEq) | Some(Token::LtEq) => (5, 6),
                Some(Token::Plus) | Some(Token::Minus) => (7, 8),
                Some(Token::Star) | Some(Token::Slash) => (9, 10),
                _ => break,
            };

            if left_bp < min_bp {
                break;
            }

            let tok = self.advance();
            match tok {
                Some(Token::Or) => {
                    let rhs = self.parse_pratt(right_bp)?;
                    lhs = Expr::Or(Box::new(lhs), Box::new(rhs));
                }
                Some(Token::And) => {
                    let rhs = self.parse_pratt(right_bp)?;
                    lhs = Expr::And(Box::new(lhs), Box::new(rhs));
                }
                Some(tok) => {
                    let op = match &tok {
                        Token::Eq => InfixOp::Eq,
                        Token::NotEq => InfixOp::NotEq,
                        Token::Gt => InfixOp::Gt,
                        Token::Lt => InfixOp::Lt,
                        Token::GtEq => InfixOp::GtEq,
                        Token::LtEq => InfixOp::LtEq,
                        Token::Plus => InfixOp::Add,
                        Token::Minus => InfixOp::Subtract,
                        Token::Star => InfixOp::Multiply,
                        Token::Slash => InfixOp::Divide,
                        _ => unreachable!(),
                    };
                    let rhs = self.parse_pratt(right_bp)?;
                    lhs = Expr::Infix {
                        op,
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    };
                }
                None => break,
            };
        }

        Ok(lhs)
    }

    pub(super) fn parse_expr_atom(&mut self) -> Result<Expr, ParseError> {
        match self.peek() {
            Some(Token::Float(_)) => {
                if let Some(Token::Float(f)) = self.advance() {
                    return Ok(Expr::Scalar(f));
                }
                unreachable!()
            }
            Some(Token::Int(_)) => {
                if let Some(Token::Int(n)) = self.advance() {
                    return Ok(Expr::Int(n));
                }
                unreachable!()
            }
            Some(Token::Str(_)) => {
                return Ok(Expr::StringLit(self.eat_str()?));
            }
            Some(Token::Minus) => {
                self.advance();
                let inner = self.parse_pratt(11)?;
                return Ok(match inner {
                    Expr::Int(n) => Expr::Int(-n),
                    Expr::Scalar(n) => Expr::Scalar(-n),
                    other => Expr::Call(CallExpr::Scale {
                        input: Box::new(other),
                        factor: -1.0,
                    }),
                });
            }
            Some(Token::Not) => {
                self.advance();
                let inner = self.parse_pratt(11)?;
                return Ok(Expr::Not(Box::new(inner)));
            }
            Some(Token::LParen) => {
                self.advance();
                let e = self.parse_pratt(0)?;
                self.eat(&Token::RParen)?;
                return Ok(e);
            }
            Some(Token::Add)
            | Some(Token::Subtract)
            | Some(Token::Multiply)
            | Some(Token::Divide)
            | Some(Token::Correlate)
            | Some(Token::Similarity)
            | Some(Token::Distance)
            | Some(Token::Matmul)
            | Some(Token::Normalize)
            | Some(Token::Transpose)
            | Some(Token::Flatten)
            | Some(Token::Sum)
            | Some(Token::Mean)
            | Some(Token::Stdev)
            | Some(Token::Scale)
            | Some(Token::Reshape)
            | Some(Token::Stack) => return self.parse_call_expr(),
            Some(Token::Ident(_)) => {
                let name = self.eat_ident()?;
                if name == "dataset" && self.at(&Token::LParen) {
                    self.advance();
                    let ds = self.eat_str()?;
                    self.eat(&Token::RParen)?;
                    return Ok(Expr::DatasetRef(ds));
                }
                let upper = name.to_uppercase();
                if matches!(upper.as_str(), "COUNT" | "SUM" | "AVG" | "MIN" | "MAX")
                    && self.at(&Token::LParen)
                {
                    self.advance();
                    let arg = if self.at(&Token::Star) {
                        self.advance();
                        "*".to_string()
                    } else {
                        self.eat_ident()?
                    };
                    self.eat(&Token::RParen)?;
                    return Ok(Expr::Ref(format!("{}({})", upper, arg)));
                }
                return Ok(Expr::Ref(name));
            }
            _ => {}
        }
        Err(self.unexpected("an expression"))
    }

    pub(super) fn parse_call_expr(&mut self) -> Result<Expr, ParseError> {
        let call = match self.advance() {
            Some(Token::Add) => {
                let a = self.parse_simple_expr()?;
                let b = self.parse_simple_expr()?;
                CallExpr::Add(Box::new(a), Box::new(b))
            }
            Some(Token::Subtract) => {
                let a = self.parse_simple_expr()?;
                let b = self.parse_simple_expr()?;
                CallExpr::Subtract(Box::new(a), Box::new(b))
            }
            Some(Token::Multiply) => {
                let a = self.parse_simple_expr()?;
                let b = self.parse_simple_expr()?;
                CallExpr::Multiply(Box::new(a), Box::new(b))
            }
            Some(Token::Divide) => {
                let a = self.parse_simple_expr()?;
                let b = self.parse_simple_expr()?;
                CallExpr::Divide(Box::new(a), Box::new(b))
            }
            Some(Token::Correlate) => {
                let a = self.parse_simple_expr()?;
                self.eat(&Token::With)?;
                let b = self.parse_simple_expr()?;
                CallExpr::Correlate(Box::new(a), Box::new(b))
            }
            Some(Token::Similarity) => {
                let a = self.parse_simple_expr()?;
                self.eat(&Token::With)?;
                let b = self.parse_simple_expr()?;
                CallExpr::Similarity(Box::new(a), Box::new(b))
            }
            Some(Token::Distance) => {
                let a = self.parse_simple_expr()?;
                self.eat(&Token::To)?;
                let b = self.parse_simple_expr()?;
                CallExpr::Distance(Box::new(a), Box::new(b))
            }
            Some(Token::Matmul) => {
                let a = self.parse_simple_expr()?;
                let b = self.parse_simple_expr()?;
                CallExpr::Matmul(Box::new(a), Box::new(b))
            }
            Some(Token::Normalize) => CallExpr::Normalize(Box::new(self.parse_simple_expr()?)),
            Some(Token::Transpose) => CallExpr::Transpose(Box::new(self.parse_simple_expr()?)),
            Some(Token::Flatten) => CallExpr::Flatten(Box::new(self.parse_simple_expr()?)),
            Some(Token::Sum) => CallExpr::Sum(Box::new(self.parse_simple_expr()?)),
            Some(Token::Mean) => CallExpr::Mean(Box::new(self.parse_simple_expr()?)),
            Some(Token::Stdev) => CallExpr::Stdev(Box::new(self.parse_simple_expr()?)),
            Some(Token::Scale) => {
                let input = self.parse_simple_expr()?;
                self.eat(&Token::By)?;
                let factor = self.eat_number()?;
                CallExpr::Scale {
                    input: Box::new(input),
                    factor,
                }
            }
            Some(Token::Reshape) => {
                let input = self.parse_simple_expr()?;
                self.eat(&Token::To)?;
                let shape = self.parse_usize_list()?;
                CallExpr::Reshape {
                    input: Box::new(input),
                    shape,
                }
            }
            Some(Token::Stack) => {
                let mut operands = vec![self.parse_simple_expr()?];
                while self.can_start_simple_expr() {
                    operands.push(self.parse_simple_expr()?);
                }
                if operands.len() < 2 {
                    return Err(self.error("STACK requires at least 2 operands"));
                }
                CallExpr::Stack(operands)
            }
            _ => return Err(self.unexpected("a named operation")),
        };
        Ok(Expr::Call(call))
    }

    pub(super) fn parse_simple_expr(&mut self) -> Result<Expr, ParseError> {
        let mut expr = match self.peek() {
            Some(Token::Float(_)) => {
                if let Some(Token::Float(f)) = self.advance() {
                    Expr::Scalar(f)
                } else {
                    unreachable!()
                }
            }
            Some(Token::Int(_)) => {
                if let Some(Token::Int(n)) = self.advance() {
                    Expr::Int(n)
                } else {
                    unreachable!()
                }
            }
            Some(Token::Minus) => {
                self.advance();
                Expr::Scalar(-self.eat_number()?)
            }
            Some(Token::Ident(_)) => Expr::Ref(self.eat_ident()?),
            Some(Token::LParen) => {
                self.advance();
                let e = self.parse_pratt(0)?;
                self.eat(&Token::RParen)?;
                e
            }
            _ => {
                return Err(self.unexpected(
                    "a simple expression (identifier, literal, or parenthesized expression)",
                ))
            }
        };

        if self.at(&Token::LBracket) {
            self.advance();
            let indices = self.parse_index_specs()?;
            self.eat(&Token::RBracket)?;
            expr = Expr::Index {
                base: Box::new(expr),
                indices,
            };
        } else if self.at(&Token::Dot) {
            self.advance();
            let field = self.eat_ident()?;
            expr = Expr::Field {
                base: Box::new(expr),
                field,
            };
        }

        Ok(expr)
    }

    pub(super) fn can_start_simple_expr(&self) -> bool {
        matches!(
            self.peek(),
            Some(Token::Ident(_))
                | Some(Token::Int(_))
                | Some(Token::Float(_))
                | Some(Token::Minus)
                | Some(Token::LParen)
        )
    }
}
