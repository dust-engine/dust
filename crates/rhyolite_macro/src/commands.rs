use crate::transformer::CommandsTransformer;

use syn::{
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    spanned::Spanned,
};

struct CommandsTransformState {
    retain_bindings: syn::punctuated::Punctuated<syn::Expr, syn::Token![,]>,

    recycled_state_count: usize,
}
impl Default for CommandsTransformState {
    fn default() -> Self {
        Self {
            retain_bindings: Punctuated::new(),

            recycled_state_count: 0,
        }
    }
}
impl CommandsTransformer for CommandsTransformState {
    fn async_transform(&mut self, input: &syn::ExprAwait, is_inloop: bool) -> syn::Expr {
        let index = syn::Index::from(self.recycled_state_count);
        let global_future_variable_name = quote::format_ident!("__future_retain");
        self.recycled_state_count += 1;

        let id = self.retain_bindings.len();
        let id = syn::Index::from(id);
        if is_inloop {
            self.retain_bindings
                .push(syn::Expr::Verbatim(quote::quote!(Vec::new())));
        } else {
            self.retain_bindings
                .push(syn::Expr::Verbatim(quote::quote!(None)));
        }
        let dispose_replace_stmt = if is_inloop {
            quote::quote! {
                #global_future_variable_name.#id.push(retain);
            }
        } else {
            quote::quote! {
                #global_future_variable_name.#id = Some(retain);
            }
        };
        let base = input.base.clone();

        // For each future, we first call init on it. This is a no-op for most futures, but for
        // (nested) block futures, this is going to move the future to its first yield point.
        // At that point it should yield the context of the first future.
        let tokens = quote::quote_spanned! {input.span()=>
            {
                let mut fut = #base;
                unsafe {
                    let mut fut_pinned = std::pin::Pin::new_unchecked(&mut fut);
                    if let Some((out, retain)) = ::rhyolite::future::GPUCommandFuture::init(fut_pinned.as_mut(), &mut *(__fut_global_ctx_ptr as *mut _), &mut {&mut *__recycled_states}.#index) {
                        #dispose_replace_stmt
                        out
                    } else {
                        (__fut_global_ctx_ptr, __recycled_states) = yield ::rhyolite::future::GPUCommandGeneratorContextFetchPtr::new(fut_pinned.as_mut());
                        loop {
                            match ::rhyolite::future::GPUCommandFuture::record(fut_pinned.as_mut(),  &mut *(__fut_global_ctx_ptr as *mut _), &mut {&mut *__recycled_states}.#index) {
                                ::std::task::Poll::Ready((output, retain)) => {
                                    #dispose_replace_stmt
                                    break output
                                },
                                ::std::task::Poll::Pending => {
                                    (__fut_global_ctx_ptr, __recycled_states) = yield ::rhyolite::future::GPUCommandGeneratorContextFetchPtr::new(fut_pinned.as_mut());
                                },
                            };
                        }
                    }
                }
            }
        };
        syn::Expr::Verbatim(tokens)
    }
    fn macro_transform_stmt(&mut self, mac: &syn::StmtMacro, in_loop: bool) -> syn::Stmt {
        let path = &mac.mac.path;
        if path.segments.len() != 1 {
            return syn::Stmt::Macro(mac.clone());
        }
        let expr = match path.segments[0].ident.to_string().as_str() {
            "retain" => syn::Expr::Verbatim(self.retain(&mac.mac.tokens, in_loop)),
            "fork" => syn::Expr::Verbatim(self.fork_transform(&mac.mac.tokens)),
            "using" => syn::Expr::Verbatim(self.using_transform(&mac.mac)),
            _ => return syn::Stmt::Macro(mac.clone()),
        };
        syn::Stmt::Expr(expr, mac.semi_token.clone())
    }
    fn macro_transform_expr(&mut self, mac: &syn::ExprMacro, in_loop: bool) -> syn::Expr {
        let path = &mac.mac.path;
        if path.segments.len() != 1 {
            return syn::Expr::Macro(mac.clone());
        }
        match path.segments[0].ident.to_string().as_str() {
            "retain" => syn::Expr::Verbatim(self.retain(&mac.mac.tokens, in_loop)),
            "fork" => syn::Expr::Verbatim(self.fork_transform(&mac.mac.tokens)),
            "using" => syn::Expr::Verbatim(self.using_transform(&mac.mac)),
            _ => return syn::Expr::Macro(mac.clone()),
        }
    }
    fn return_transform(&mut self, ret: &syn::ExprReturn) -> Option<syn::Expr> {
        let global_res_variable_name = quote::format_ident!("__future_retain");
        // Transform each return statement into a yield, drops, and return.
        // We use RefCell on awaited_future_drops and import_drops so that they can be read while being modified.
        // This ensures that we won't drop uninitialized values.'
        // The executor will stop the execution as soon as it reaches the first `yield Complete` statement.
        // Drops are written between the yield and return, so these values are retained inside the generator
        // until the generator itself was dropped.
        // We drop the generator only after semaphore was signaled from within the queue.
        let returned_item = ret
            .expr
            .as_ref()
            .map(|a| *a.clone())
            .unwrap_or(syn::Expr::Verbatim(quote::quote!(())));
        let token_stream = quote::quote!(
            {
                return (#returned_item, #global_res_variable_name);
            }
        );
        let block = syn::parse2::<syn::ExprBlock>(token_stream).unwrap();
        Some(syn::Expr::Block(block))
    }
}
impl CommandsTransformState {
    fn retain(
        &mut self,
        input_tokens: &proc_macro2::TokenStream,
        in_loop: bool,
    ) -> proc_macro2::TokenStream {
        let id = self.retain_bindings.len();
        let id = syn::Index::from(id);
        let global_res_variable_name = quote::format_ident!("__future_retain");
        if in_loop {
            self.retain_bindings
                .push(syn::Expr::Verbatim(quote::quote!(Vec::new())));

            quote::quote! {unsafe {
                #global_res_variable_name.#id.push(#input_tokens)
            }}
        } else {
            self.retain_bindings
                .push(syn::Expr::Verbatim(quote::quote!(None)));

            quote::quote! {unsafe {
                #global_res_variable_name.#id = Some(#input_tokens)
            }}
        }
    }
    fn using_transform(&mut self, input: &syn::Macro) -> proc_macro2::TokenStream {
        // Transform the use! macros. Input should be an expression that implements Default.
        // Returns a mutable reference to the value.

        let index = syn::Index::from(self.recycled_state_count);
        self.recycled_state_count += 1;
        quote::quote_spanned! {input.span()=>
            &mut unsafe{&mut *__recycled_states}.#index
        }
    }
    fn fork_transform(&mut self, input: &proc_macro2::TokenStream) -> proc_macro2::TokenStream {
        let ForkInput {
            forked_future,
            number_of_forks,
            _comma: _,
            scope,
        } = match syn::parse2::<ForkInput>(input.clone()) {
            Ok(input) => input,
            Err(err) => return err.to_compile_error(),
        };
        let number_of_forks = number_of_forks.map(|(_, number)| number).unwrap_or(2);
        let ret = syn::Expr::Tuple(syn::ExprTuple {
            attrs: Vec::new(),
            paren_token: Default::default(),
            elems: (0..number_of_forks)
                .map(|i| {
                    syn::Expr::Verbatim(quote::quote! {
                        GPUCommandForked::new(&forked_future_inner, #i)
                    })
                })
                .collect(),
        });
        let scope = syn::Block {
            brace_token: scope.brace_token.clone(),
            stmts: scope
                .stmts
                .iter()
                .map(|stmt| self.transform_stmt(stmt, false))
                .collect(),
        };
        quote::quote! {{
            let mut forked_future = ::rhyolite::future::GPUCommandForkedStateInner::Some(#forked_future);
            let mut pinned = unsafe{std::pin::Pin::new_unchecked(&mut forked_future)};
            pinned.as_mut().unwrap_pinned().init( &mut *(__fut_global_ctx_ptr as *mut _), __recycled_states);
            let forked_future_inner = GPUCommandForkedInner::<_, #number_of_forks>::new(pinned);
            let #forked_future = #ret;
            #scope
        }}
    }
}

struct ForkInput {
    forked_future: syn::Expr,
    number_of_forks: Option<(syn::Token![,], usize)>,
    _comma: syn::Token![,],
    scope: syn::Block,
}
impl Parse for ForkInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let forked_future = input.parse()?;
        let comma: syn::Token![,] = input.parse()?;
        let number_of_forks = {
            let number: Option<syn::LitInt> = input.parse().ok();
            let number = number.and_then(|a| a.base10_parse::<usize>().ok());
            number
        };
        if let Some(number_of_forks) = number_of_forks {
            Ok(ForkInput {
                forked_future,
                number_of_forks: Some((comma, number_of_forks)),
                _comma: input.parse()?,
                scope: input.parse()?,
            })
        } else {
            Ok(ForkInput {
                forked_future,
                number_of_forks: None,
                _comma: comma,
                scope: input.parse()?,
            })
        }
    }
}

pub fn proc_macro_commands(input: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    let global_res_variable_name = quote::format_ident!("__future_retain");
    let input = match syn::parse2::<crate::ExprGpuAsync>(input) {
        Ok(input) => input,
        Err(err) => return err.to_compile_error(),
    };
    let mut state = CommandsTransformState::default();

    let mut inner_closure_stmts: Vec<_> = input
        .stmts
        .iter()
        .map(|stmt| state.transform_stmt(stmt, false))
        .collect();

    // Transform the final return
    let append_unit_return = if let Some(last) = inner_closure_stmts.last_mut() {
        match last {
            syn::Stmt::Local(_) => true,
            syn::Stmt::Macro(_) => true,
            syn::Stmt::Item(_) => todo!(),
            syn::Stmt::Expr(expr, semi) => {
                if let Some(_semi) = semi {
                    true
                } else {
                    let token_stream = quote::quote! {
                            return (#expr, #global_res_variable_name);
                    };
                    *expr = syn::Expr::Verbatim(token_stream);
                    false
                }
            }
        }
    } else {
        true
    };
    if append_unit_return {
        let token_stream = quote::quote!(
                return ((), #global_res_variable_name)
        );
        inner_closure_stmts.push(syn::Stmt::Expr(
            syn::Expr::Verbatim(token_stream),
            Some(Default::default()),
        ))
    }

    let retain_bindings = state.retain_bindings;
    let recycled_states_type = syn::Type::Tuple(syn::TypeTuple {
        paren_token: Default::default(),
        elems: {
            let mut elems = syn::punctuated::Punctuated::from_iter(
                std::iter::repeat(syn::Type::Infer(syn::TypeInfer {
                    underscore_token: Default::default(),
                }))
                .take(state.recycled_state_count),
            );
            if state.recycled_state_count == 1 {
                elems.push_punct(Default::default());
            }
            elems
        },
    });

    let mv = input.mv;
    quote::quote! {
        ::rhyolite::future::GPUCommandBlock::new(static #mv |(mut __fut_global_ctx_ptr, mut __recycled_states): (*mut (), *mut #recycled_states_type)| {
            let mut #global_res_variable_name = (#retain_bindings, );
            #(#inner_closure_stmts)*
        })
    }
}
