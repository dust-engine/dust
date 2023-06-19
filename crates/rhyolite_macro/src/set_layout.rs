use quote::spanned::Spanned;
use syn::{
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    Attribute,
};

pub struct SetLayoutBinding {
    attrs: Vec<syn::Attribute>,
    name: syn::Ident,
    colon_token: syn::Token![:],
    descriptor_type: SetLayoutBindingDescriptorType,
    // ALso needs: binding id, immutable sampler, shader stage flags
}

pub enum SetLayoutBindingDescriptorType {
    Single(syn::Expr),
    Multi {
        bracket_token: syn::token::Bracket,
        ty: syn::Expr,
        semi_token: syn::Token![;],
        len: syn::Expr,
    },
}

pub struct SetLayout {
    attrs: Vec<Attribute>,
    bindings: Punctuated<SetLayoutBinding, syn::Token![,]>,
}

impl Parse for SetLayoutBindingDescriptorType {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        if input.peek(syn::token::Bracket) {
            let content;
            Ok(Self::Multi {
                bracket_token: syn::bracketed!(content in input),
                ty: content.parse()?,
                semi_token: content.parse()?,
                len: content.parse()?,
            })
        } else {
            Ok(Self::Single(input.parse()?))
        }
    }
}

impl Parse for SetLayoutBinding {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(SetLayoutBinding {
            attrs: input.call(syn::Attribute::parse_outer)?,
            name: input.parse()?,
            colon_token: input.parse()?,
            descriptor_type: input.parse()?,
        })
    }
}

impl Parse for SetLayout {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(SetLayout {
            attrs: input.call(syn::Attribute::parse_inner)?,
            bindings: input.parse_terminated(SetLayoutBinding::parse, syn::Token![,])?,
        })
    }
}

pub fn set_layout(input: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    let input = match syn::parse2::<SetLayout>(input) {
        Ok(input) => input,
        Err(err) => return err.to_compile_error(),
    };

    let mut requires_specify_binding = false;
    let flags: Vec<_> = input
        .attrs
        .iter()
        .filter(|attr| attr.meta.path().is_ident("flag"))
        .map(|attr| attr.meta.require_list().as_ref().unwrap().tokens.clone())
        .collect();
    let binding_infos = input
        .bindings
        .into_iter()
        .enumerate()
        .map(|(i, input_binding)| {
            let (descriptor_type, descriptor_count) = match input_binding.descriptor_type {
                SetLayoutBindingDescriptorType::Single(ty) => {
                    let lit = syn::Expr::Lit(syn::ExprLit {
                        lit: syn::Lit::Int(syn::LitInt::new("1u32", ty.__span())),
                        attrs: Vec::new(),
                    });
                    (ty, lit)
                }
                SetLayoutBindingDescriptorType::Multi {
                    bracket_token: _,
                    ty,
                    semi_token: _,
                    len,
                } => (ty, len),
            };
            let binding = input_binding
                .attrs
                .iter()
                .find(|attr| attr.meta.path().is_ident("binding"))
                .map(|attr| {
                    let token_stream: proc_macro2::TokenStream =
                        attr.meta.require_list().unwrap().tokens.clone().into();
                    token_stream
                });
            let binding = if let Some(binding) = binding {
                requires_specify_binding = true;
                binding
            } else {
                if requires_specify_binding {
                    // Throw error
                    todo!()
                } else {
                    // Default binding number
                    let i = i as u32;
                    quote::quote!(#i)
                }
            };
            let shader_flags = input_binding
                .attrs
                .iter()
                .find(|attr| attr.meta.path().is_ident("shader"))
                .map(|attr| {
                    let token_stream: proc_macro2::TokenStream =
                        attr.meta.require_list().unwrap().tokens.clone().into();
                    token_stream
                })
                .unwrap_or_else(|| {
                    quote::quote! {
                        Default::default()
                    }
                });
            quote::quote! {
                ::rhyolite::descriptor::DescriptorSetLayoutBindingInfo {
                    binding: #binding,
                    descriptor_type: #descriptor_type,
                    descriptor_count: #descriptor_count,
                    stage_flags: #shader_flags,
                    immutable_samplers: Vec::new(),
                }
            }
        });

    let flags = if flags.len() == 0 {
        quote::quote! {
            Default::default()
        }
    } else {
        quote::quote!(#(#flags)|*)
    };
    quote::quote! {
        ::rhyolite::descriptor::DescriptorSetLayoutCacheKey {
            bindings: vec![
                #(#binding_infos),*
            ],
            flags: #flags
        }
    }
}
