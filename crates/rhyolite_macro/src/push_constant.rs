use std::collections::{HashMap, HashSet};

use quote::ToTokens;
use syn::{parse::Parse, punctuated::Punctuated, spanned::Spanned, Token};

struct PushConstantRange {
    stage: syn::Path,
    starting_field: syn::Ident,
    ending_field: Option<syn::Ident>,
}

struct PushConstantFieldAttr {
    stages: Punctuated<syn::Path, Token![,]>,
}
impl Parse for PushConstantFieldAttr {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        Ok(Self {
            stages: input.parse_terminated(syn::Path::parse, syn::Token![,])?,
        })
    }
}

pub fn push_constant(input_token_stream: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    let mut input = match syn::parse2::<syn::ItemStruct>(input_token_stream) {
        Ok(input) => input,
        Err(err) => return err.to_compile_error(),
    };
    let fields = match &mut input.fields {
        syn::Fields::Named(field) => &mut field.named,
        syn::Fields::Unnamed(field) => &mut field.unnamed,
        syn::Fields::Unit => todo!(),
    };

    let mut current_active_stages: HashMap<String, PushConstantRange> = HashMap::new();
    for field in fields.iter_mut() {
        let Some(stage) = field.attrs.drain_filter(|a| a.meta.path().is_ident("stage")).next() else {
            return syn::Error::new(field.span(), "Field is missing the `stage` attribute").to_compile_error();
        };
        let stage = match stage.meta.require_list() {
            Ok(stage) => stage,
            Err(err) => return err.to_compile_error(),
        };

        let tokens: PushConstantFieldAttr = match stage.parse_args() {
            Ok(stage) => stage,
            Err(err) => return err.to_compile_error(),
        };
        let stages: HashSet<String> = tokens
            .stages
            .iter()
            .map(|a| a.to_token_stream().to_string())
            .collect();

        for stage in tokens.stages {
            let entry = current_active_stages
                .entry(stage.to_token_stream().to_string())
                .or_insert_with(|| PushConstantRange {
                    stage,
                    starting_field: field.ident.clone().unwrap(),
                    ending_field: None,
                });
            if entry.ending_field.is_some() {
                return syn::Error::new(field.span(), "Non-contiguous stage").to_compile_error();
            }
        }

        for (key, value) in current_active_stages.iter_mut() {
            if !stages.contains(key) {
                assert!(value.ending_field.is_none());
                value.ending_field = Some(field.ident.clone().unwrap())
            }
        }
    }

    let name = input.ident.clone();

    let ranges = current_active_stages.values().map(|range| {
        let stage = range.stage.clone();
        let field_start = range.starting_field.clone();
        let size = if let Some(field_end) = range.ending_field.clone() {
            quote::quote!(::rhyolite::offset_of!(#name, #field_end) - ::rhyolite::offset_of!(#name, #field_start))
        } else {
            quote::quote!(::std::mem::size_of::<#name>() - ::rhyolite::offset_of!(#name, #field_start))
        };
        quote::quote! {
            ::rhyolite::ash::vk::PushConstantRange {
                stage_flags: #stage,
                offset: ::rhyolite::offset_of!(#name, #field_start) as u32,
                size: (#size) as u32
            }
        }
    });
    quote::quote!(

        impl ::rhyolite::descriptor::PushConstants for #name {
            fn ranges() -> Vec<vk::PushConstantRange> {
                vec![
                    #(#ranges),*
                ]
            }
        }
    )
}
