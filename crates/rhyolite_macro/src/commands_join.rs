use quote::ToTokens;
use syn::parse::{Parse, ParseStream};

struct MacroJoin {
    pub exprs: syn::punctuated::Punctuated<syn::Expr, syn::Token![,]>,
}
impl Parse for MacroJoin {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(MacroJoin {
            exprs: syn::punctuated::Punctuated::parse_separated_nonempty(input)?,
        })
    }
}

pub fn proc_macro_join(input: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    let input = match syn::parse2::<MacroJoin>(input) {
        Ok(input) => input,
        Err(err) => return err.to_compile_error(),
    };
    if input.exprs.len() == 0 {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "Expects at least one argument",
        )
        .to_compile_error();
    }
    if input.exprs.len() == 1 {
        return input.exprs[0].clone().into_token_stream();
    }

    let mut token_stream = proc_macro2::TokenStream::new();
    token_stream.extend(input.exprs[0].clone().into_token_stream());

    // a.join(b).join(c)...
    for item in input.exprs.iter().skip(1) {
        token_stream.extend(quote::quote! {.join(#item)}.into_iter());
    }

    let num_expressions = input.exprs.len();

    // __join_0, __join_1, __join_2, ...
    let output_expression = (0..num_expressions).map(|i| quote::format_ident!("__join_{}", i));

    let input_expression = {
        use proc_macro2::{Delimiter, Group, Ident, Punct, Spacing, Span, TokenStream, TokenTree};
        let mut t = Some(TokenTree::Group(Group::new(Delimiter::Parenthesis, {
            let mut t = TokenStream::new();
            t.extend(Some(TokenTree::Ident(Ident::new(
                "__join_0",
                Span::call_site(),
            ))));
            t.extend(Some(TokenTree::Punct(Punct::new(',', Spacing::Alone))));
            t.extend(Some(TokenTree::Ident(Ident::new(
                "__join_1",
                Span::call_site(),
            ))));
            t
        })));
        (2..num_expressions).for_each(|i| {
            let prev = t.take().unwrap().into_token_stream();
            t = Some(TokenTree::Group(Group::new(Delimiter::Parenthesis, {
                let mut a = TokenStream::new();
                a.extend(Some(prev));
                a.extend(Some(TokenTree::Punct(Punct::new(',', Spacing::Alone))));
                a.extend(Some(TokenTree::Ident(quote::format_ident!("__join_{}", i))));
                a
            })));
        });
        t
    };
    token_stream
        .extend(quote::quote! {.map(|#input_expression| (#(#output_expression),*))}.into_iter());
    token_stream
}
