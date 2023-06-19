use syn::{Block, Expr, Pat, Stmt};

pub trait CommandsTransformer {
    fn transform_block(&mut self, block: &Block, is_inloop: bool) -> Block {
        Block {
            brace_token: block.brace_token.clone(),
            stmts: block
                .stmts
                .iter()
                .map(move |stmt| self.transform_stmt(stmt, is_inloop))
                .collect(),
        }
    }

    fn transform_pattern(&mut self, pat: &Pat, is_inloop: bool) -> Pat {
        match pat {
            Pat::Ident(ident) => Pat::Ident(syn::PatIdent {
                subpat: ident.subpat.as_ref().map(|(at, subpat)| {
                    (
                        at.clone(),
                        Box::new(self.transform_pattern(subpat, is_inloop)),
                    )
                }),
                ..ident.clone()
            }),
            Pat::Lit(lit) => Pat::Lit(syn::PatLit {
                attrs: lit.attrs.clone(),
                lit: lit.lit.clone(),
            }),
            Pat::Or(clause) => Pat::Or(syn::PatOr {
                cases: clause
                    .cases
                    .iter()
                    .map(|pat| self.transform_pattern(pat, is_inloop))
                    .collect(),
                ..clause.clone()
            }),
            Pat::Range(range) => Pat::Range(syn::PatRange {
                start: range
                    .start
                    .as_ref()
                    .map(|start| Box::new(self.transform_expr(&start, is_inloop))),
                end: range
                    .end
                    .as_ref()
                    .map(|end| Box::new(self.transform_expr(&end, is_inloop))),
                ..range.clone()
            }),
            Pat::Reference(r) => Pat::Reference(syn::PatReference {
                pat: Box::new(self.transform_pattern(&r.pat, is_inloop)),
                ..r.clone()
            }),
            Pat::Slice(slice) => Pat::Slice(syn::PatSlice {
                elems: slice
                    .elems
                    .iter()
                    .map(|pat| self.transform_pattern(pat, is_inloop))
                    .collect(),
                ..slice.clone()
            }),
            Pat::Struct(s) => Pat::Struct(syn::PatStruct {
                fields: s
                    .fields
                    .iter()
                    .map(|f| syn::FieldPat {
                        pat: Box::new(self.transform_pattern(&f.pat, is_inloop)),
                        ..f.clone()
                    })
                    .collect(),
                ..s.clone()
            }),
            Pat::Tuple(tuple) => Pat::Tuple(syn::PatTuple {
                elems: tuple
                    .elems
                    .iter()
                    .map(|pat| self.transform_pattern(pat, is_inloop))
                    .collect(),
                ..tuple.clone()
            }),
            Pat::TupleStruct(tuple) => Pat::TupleStruct(syn::PatTupleStruct {
                elems: tuple
                    .elems
                    .iter()
                    .map(|pat| self.transform_pattern(pat, is_inloop))
                    .collect(),
                ..tuple.clone()
            }),
            Pat::Type(ty) => Pat::Type(syn::PatType {
                pat: Box::new(self.transform_pattern(&ty.pat, is_inloop)),
                ..ty.clone()
            }),
            _ => pat.clone(),
        }
    }
    fn transform_expr(&mut self, expr: &Expr, is_inloop: bool) -> Expr {
        match expr {
            Expr::Array(arr) => Expr::Array(syn::ExprArray {
                elems: arr
                    .elems
                    .iter()
                    .map(|expr| self.transform_expr(expr, is_inloop))
                    .collect(),
                ..arr.clone()
            }),
            Expr::Assign(assign) => Expr::Assign(syn::ExprAssign {
                left: Box::new(self.transform_expr(&*assign.left, is_inloop)),
                right: Box::new(self.transform_expr(&*assign.right, is_inloop)),
                ..assign.clone()
            }),
            Expr::Binary(binary) => Expr::Binary(syn::ExprBinary {
                left: Box::new(self.transform_expr(&*binary.left, is_inloop)),
                right: Box::new(self.transform_expr(&*binary.right, is_inloop)),
                ..binary.clone()
            }),
            Expr::Block(block) => Expr::Block(syn::ExprBlock {
                block: self.transform_block(&block.block, is_inloop),
                ..block.clone()
            }),
            Expr::Call(call) => Expr::Call(syn::ExprCall {
                func: Box::new(self.transform_expr(&*call.func, is_inloop)),
                args: call
                    .args
                    .iter()
                    .map(|expr| self.transform_expr(expr, is_inloop))
                    .collect(),
                ..call.clone()
            }),
            Expr::Cast(cast) => Expr::Cast(syn::ExprCast {
                expr: Box::new(self.transform_expr(&*cast.expr, is_inloop)),
                ..cast.clone()
            }),
            Expr::Field(field) => Expr::Field(syn::ExprField {
                base: Box::new(self.transform_expr(&*field.base, is_inloop)),
                ..field.clone()
            }),
            Expr::ForLoop(l) => Expr::ForLoop(syn::ExprForLoop {
                pat: Box::new(self.transform_pattern(&l.pat, is_inloop)),
                expr: Box::new(self.transform_expr(&*l.expr, is_inloop)),
                body: self.transform_block(&l.body, true),
                ..l.clone()
            }),
            Expr::Group(group) => Expr::Group(syn::ExprGroup {
                expr: Box::new(self.transform_expr(&group.expr, is_inloop)),
                ..group.clone()
            }),
            Expr::If(if_stmt) => Expr::If(syn::ExprIf {
                cond: Box::new(self.transform_expr(&if_stmt.cond, is_inloop)),
                then_branch: self.transform_block(&if_stmt.then_branch, is_inloop),
                else_branch: if_stmt.else_branch.as_ref().map(|(else_token, expr)| {
                    (
                        else_token.clone(),
                        Box::new(self.transform_expr(&expr, is_inloop)),
                    )
                }),
                ..if_stmt.clone()
            }),
            Expr::Index(index_expr) => Expr::Index(syn::ExprIndex {
                expr: Box::new(self.transform_expr(&*index_expr.expr, is_inloop)),
                index: Box::new(self.transform_expr(&*index_expr.index, is_inloop)),
                ..index_expr.clone()
            }),
            Expr::Let(l) => Expr::Let(syn::ExprLet {
                pat: Box::new(self.transform_pattern(&l.pat, is_inloop)),
                expr: Box::new(self.transform_expr(&*l.expr, is_inloop)),
                ..l.clone()
            }),
            Expr::Loop(loop_stmt) => Expr::Loop(syn::ExprLoop {
                body: self.transform_block(&loop_stmt.body, is_inloop),
                ..loop_stmt.clone()
            }),
            Expr::Macro(m) => self.macro_transform_expr(m, is_inloop),
            Expr::Match(match_expr) => Expr::Match(syn::ExprMatch {
                expr: Box::new(self.transform_expr(&*match_expr.expr, is_inloop)),
                arms: match_expr
                    .arms
                    .iter()
                    .map(|arm| syn::Arm {
                        pat: self.transform_pattern(&arm.pat, is_inloop),
                        guard: arm.guard.as_ref().map(|(guard_token, expr)| {
                            (
                                guard_token.clone(),
                                Box::new(self.transform_expr(&expr, is_inloop)),
                            )
                        }),
                        body: Box::new(self.transform_expr(&arm.body, is_inloop)),
                        ..arm.clone()
                    })
                    .collect(),
                ..match_expr.clone()
            }),
            Expr::MethodCall(call) => Expr::MethodCall(syn::ExprMethodCall {
                receiver: Box::new(self.transform_expr(&call.receiver, is_inloop)),
                args: call
                    .args
                    .iter()
                    .map(|expr| self.transform_expr(expr, is_inloop))
                    .collect(),
                ..call.clone()
            }),
            Expr::Paren(paren) => Expr::Paren(syn::ExprParen {
                expr: Box::new(self.transform_expr(&paren.expr, is_inloop)),
                ..paren.clone()
            }),
            Expr::Range(range) => Expr::Range(syn::ExprRange {
                start: range
                    .start
                    .as_ref()
                    .map(|f| Box::new(self.transform_expr(&f, is_inloop))),
                end: range
                    .end
                    .as_ref()
                    .map(|t| Box::new(self.transform_expr(&t, is_inloop))),
                ..range.clone()
            }),
            Expr::Reference(reference) => Expr::Reference(syn::ExprReference {
                expr: Box::new(self.transform_expr(&reference.expr, is_inloop)),
                ..reference.clone()
            }),
            Expr::Repeat(repeat) => Expr::Repeat(syn::ExprRepeat {
                expr: Box::new(self.transform_expr(&repeat.expr, is_inloop)),
                len: Box::new(self.transform_expr(&repeat.len, is_inloop)),
                ..repeat.clone()
            }),
            Expr::Return(ret) => {
                if let Some(expr) = self.return_transform(ret) {
                    expr
                } else {
                    Expr::Return(syn::ExprReturn {
                        expr: ret
                            .expr
                            .as_ref()
                            .map(|e| Box::new(self.transform_expr(&e, is_inloop))),
                        ..ret.clone()
                    })
                }
            }
            Expr::Struct(s) => Expr::Struct(syn::ExprStruct {
                fields: s
                    .fields
                    .iter()
                    .map(|f| syn::FieldValue {
                        member: f.member.clone(),
                        expr: self.transform_expr(&f.expr, is_inloop),
                        ..f.clone()
                    })
                    .collect(),
                ..s.clone()
            }),
            Expr::Try(s) => Expr::Try(syn::ExprTry {
                expr: Box::new(self.transform_expr(&s.expr, is_inloop)),
                ..s.clone()
            }),
            Expr::Tuple(tuple) => Expr::Tuple(syn::ExprTuple {
                elems: tuple
                    .elems
                    .iter()
                    .map(|expr| self.transform_expr(expr, is_inloop))
                    .collect(),
                ..tuple.clone()
            }),
            Expr::Unary(unary) => Expr::Unary(syn::ExprUnary {
                expr: Box::new(self.transform_expr(&unary.expr, is_inloop)),
                ..unary.clone()
            }),
            Expr::Unsafe(unsafe_stmt) => Expr::Unsafe(syn::ExprUnsafe {
                block: self.transform_block(&unsafe_stmt.block, is_inloop),
                ..unsafe_stmt.clone()
            }),
            Expr::While(while_stmt) => Expr::While(syn::ExprWhile {
                cond: Box::new(self.transform_expr(&while_stmt.cond, true)),
                body: self.transform_block(&while_stmt.body, true),
                ..while_stmt.clone()
            }),
            Expr::Await(await_expr) => self.async_transform(await_expr, is_inloop),
            _ => expr.clone(),
        }
    }
    fn transform_stmt(&mut self, stmt: &Stmt, is_inloop: bool) -> Stmt {
        match stmt {
            Stmt::Local(local) => Stmt::Local(syn::Local {
                pat: self.transform_pattern(&local.pat, is_inloop),
                init: local.init.as_ref().map(|local_init| syn::LocalInit {
                    eq_token: local_init.eq_token.clone(),
                    expr: Box::new(self.transform_expr(&local_init.expr, is_inloop)),
                    diverge: local_init.diverge.as_ref().map(|(else_token, expr)| {
                        (
                            else_token.clone(),
                            Box::new(self.transform_expr(&expr, is_inloop)),
                        )
                    }),
                }),
                ..local.clone()
            }),
            Stmt::Expr(expr, semi) => {
                Stmt::Expr(self.transform_expr(&expr, is_inloop), semi.clone())
            }
            Stmt::Macro(mac) => self.macro_transform_stmt(mac, is_inloop),
            _ => stmt.clone(),
        }
    }
    fn async_transform(&mut self, input: &syn::ExprAwait, is_inloop: bool) -> syn::Expr;
    fn macro_transform_stmt(&mut self, mac: &syn::StmtMacro, is_inloop: bool) -> syn::Stmt;
    fn macro_transform_expr(&mut self, mac: &syn::ExprMacro, is_inloop: bool) -> syn::Expr;
    fn return_transform(&mut self, ret: &syn::ExprReturn) -> Option<syn::Expr>;
}
