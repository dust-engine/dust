#![feature(track_path, proc_macro_span, proc_macro_diagnostic, extend_one)]
#![feature(let_chains)]
#![feature(extract_if)]

use syn::parse::{Parse, ParseStream};

extern crate proc_macro;
mod commands;
mod commands_join;
mod gpu;
mod push_constant;
mod transformer;

struct ExprGpuAsync {
    pub mv: Option<syn::Token![move]>,
    pub stmts: Vec<syn::Stmt>,
}
impl Parse for ExprGpuAsync {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(ExprGpuAsync {
            mv: input.parse()?,
            stmts: syn::Block::parse_within(input)?,
        })
    }
}

#[proc_macro]
pub fn commands(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    commands::proc_macro_commands(input.into()).into()
}

#[proc_macro]
pub fn gpu(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    gpu::proc_macro_gpu(input.into()).into()
}

#[proc_macro]
pub fn join(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    commands_join::proc_macro_join(input.into()).into()
}

#[proc_macro_derive(PushConstants, attributes(stage))]
pub fn push_constant(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    push_constant::push_constant(input.into()).into()
}
