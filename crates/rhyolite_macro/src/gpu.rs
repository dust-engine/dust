use syn::{punctuated::Punctuated, spanned::Spanned};

use crate::transformer::CommandsTransformer;

struct State {
    current_dispose_index: u32,
    dispose_bindings: Punctuated<syn::Expr, syn::Token![,]>,

    recycled_state_count: usize,
}
impl State {
    fn retain(
        &mut self,
        input_tokens: &proc_macro2::TokenStream,
        is_inloop: bool,
    ) -> proc_macro2::TokenStream {
        let res_token_name = quote::format_ident!("__future_res");
        let id = syn::Index::from(self.current_dispose_index as usize);
        self.current_dispose_index += 1;

        if is_inloop {
            self.dispose_bindings
                .push(syn::Expr::Verbatim(quote::quote! {
                    Vec::new()
                }));
            quote::quote! {{
                #res_token_name.#id.push(#input_tokens)
            }}
        } else {
            self.dispose_bindings
                .push(syn::Expr::Verbatim(quote::quote! {
                    None
                }));
            quote::quote! {{
                #res_token_name.#id = Some(#input_tokens)
            }}
        }
    }

    fn using(&mut self, input_tokens: &proc_macro2::TokenStream) -> proc_macro2::TokenStream {
        let index = syn::Index::from(self.recycled_state_count);
        self.recycled_state_count += 1;
        quote::quote_spanned! {input_tokens.span()=>
            &mut unsafe{&mut *__recycled_states}.#index
        }
    }
}
impl Default for State {
    fn default() -> Self {
        Self {
            dispose_bindings: Default::default(),
            current_dispose_index: 0,
            recycled_state_count: 0,
        }
    }
}

impl CommandsTransformer for State {
    fn async_transform(&mut self, input: &syn::ExprAwait, is_inloop: bool) -> syn::Expr {
        let base = input.base.clone();

        let dispose_token_name = quote::format_ident!("__future_res");
        let id = syn::Index::from(self.current_dispose_index as usize);
        self.current_dispose_index += 1;

        // Creates a location to store the dispose future. Dispose futures should be invoked
        // when the future returned by the parent QueueFutureBlock was invoked.
        // When the corresponding QueueFuture was awaited, the macro writes the return value of
        // its dispose method into this location.
        // This needs to be a cell because __dispose_fn_future "pre-mutably-borrows" the value.
        // This value also needs to be written by the .await statement, creating a double borrow.
        if is_inloop {
            self.dispose_bindings
                .push(syn::Expr::Verbatim(quote::quote! {
                    Vec::new()
                }));
        } else {
            self.dispose_bindings
                .push(syn::Expr::Verbatim(quote::quote! {
                    None
                }));
        }
        let dispose_replace_stmt = if is_inloop {
            quote::quote! {
                #dispose_token_name.#id.push(::rhyolite::QueueFuture::dispose(fut));
            }
        } else {
            quote::quote! {
                #dispose_token_name.#id.replace(::rhyolite::QueueFuture::dispose(fut));
            }
        };

        let index = syn::Index::from(self.recycled_state_count);
        self.recycled_state_count += 1;
        syn::Expr::Verbatim(quote::quote! {{
            let mut fut = #base;
            let mut fut_pinned = unsafe{std::pin::Pin::new_unchecked(&mut fut)};
            ::rhyolite::QueueFuture::setup(
                fut_pinned.as_mut(),
                unsafe{&mut *(__ctx as *mut ::rhyolite::queue::SubmissionContext)},
                &mut unsafe{&mut *__recycled_states}.#index,
                __current_queue
            );
            let output = loop {
                match ::rhyolite::QueueFuture::record(
                    fut_pinned.as_mut(),
                    unsafe{&mut *(__ctx as *mut ::rhyolite::queue::SubmissionContext)},
                    &mut unsafe{&mut *__recycled_states}.#index
                ) {
                    ::rhyolite::queue::QueueFuturePoll::Ready { next_queue, output } => {
                        __current_queue = next_queue;
                        break output;
                    },
                    ::rhyolite::queue::QueueFuturePoll::Semaphore(s) => (__initial_queue, __ctx, __recycled_states) = yield Some(s),
                    ::rhyolite::queue::QueueFuturePoll::Barrier => (__initial_queue, __ctx, __recycled_states) = yield None,
                };
            };
            #dispose_replace_stmt
            output
        }})
    }
    //               queue
    // Leaf  nodes   the actual queue
    // Join  nodes   None, or the actual queue if all the same.
    // Block nodes   None
    // for blocks, who yields initially?
    // inner block yields. outer block needs to give inner block the current queue, and the inner block choose to yield or not.

    fn macro_transform_expr(&mut self, mac: &syn::ExprMacro, is_inloop: bool) -> syn::Expr {
        let path = &mac.mac.path;
        if path.segments.len() != 1 {
            return syn::Expr::Macro(mac.clone());
        }
        match path.segments[0].ident.to_string().as_str() {
            "retain" => syn::Expr::Verbatim(self.retain(&mac.mac.tokens, is_inloop)),
            "using" => syn::Expr::Verbatim(self.using(&mac.mac.tokens)),
            _ => syn::Expr::Macro(mac.clone()),
        }
    }
    fn macro_transform_stmt(&mut self, mac: &syn::StmtMacro, is_inloop: bool) -> syn::Stmt {
        let path = &mac.mac.path;
        if path.segments.len() != 1 {
            return syn::Stmt::Macro(mac.clone());
        }
        let expr = match path.segments[0].ident.to_string().as_str() {
            "retain" => syn::Expr::Verbatim(self.retain(&mac.mac.tokens, is_inloop)),
            "using" => syn::Expr::Verbatim(self.using(&mac.mac.tokens)),
            _ => return syn::Stmt::Macro(mac.clone()),
        };
        syn::Stmt::Expr(expr, mac.semi_token.clone())
    }
    fn return_transform(&mut self, ret: &syn::ExprReturn) -> Option<syn::Expr> {
        let returned_item = ret
            .expr
            .as_ref()
            .map(|a| *a.clone())
            .unwrap_or(syn::Expr::Verbatim(quote::quote!(())));

        let dispose_token_name = quote::format_ident!("__future_res");

        let token_stream = quote::quote!(
            {
                return (__current_queue, #dispose_token_name, #returned_item);
            }
        );
        Some(syn::Expr::Verbatim(token_stream))
    }
}

pub fn proc_macro_gpu(input: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    let input = match syn::parse2::<crate::ExprGpuAsync>(input) {
        Ok(input) => input,
        Err(err) => return err.to_compile_error(),
    };
    let mut state = State::default();

    let mut inner_closure_stmts: Vec<_> = input
        .stmts
        .iter()
        .map(|stmt| state.transform_stmt(stmt, false))
        .collect();

    let dispose_token_name = quote::format_ident!("__future_res");

    // Transform the final stmt
    let append_unit_return = if let Some(last) = inner_closure_stmts.last_mut() {
        match last {
            syn::Stmt::Local(_) => true,
            syn::Stmt::Macro(_) => true,
            syn::Stmt::Item(_) => todo!(),
            syn::Stmt::Expr(expr, semi) => {
                if let Some(_semi) = semi {
                    true
                } else {
                    let token_stream = quote::quote!({
                        return (__current_queue, #dispose_token_name, #expr);
                    });
                    *expr = syn::Expr::Verbatim(token_stream);
                    false
                }
            }
        }
    } else {
        true
    };
    if append_unit_return {
        let token_stream = quote::quote!({
            return (__current_queue,  #dispose_token_name, ())
        });
        inner_closure_stmts.push(syn::Stmt::Expr(
            syn::Expr::Verbatim(token_stream),
            Some(Default::default()),
        ))
    }
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
    let dispose_bindings = state.dispose_bindings;
    let dispose_token_name = quote::format_ident!("__future_res");
    quote::quote! {
        rhyolite::queue::QueueFutureBlock::new(static #mv |(mut __initial_queue, mut __ctx, mut __recycled_states):(_,_,*mut #recycled_states_type)| {
            let mut #dispose_token_name = (#dispose_bindings, );
            let mut __current_queue: ::rhyolite::queue::QueueMask = __initial_queue;
            #(#inner_closure_stmts)*
        })
    }
}
