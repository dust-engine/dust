//! The derive for `dust_ecs::Component`. Use it through `dust_ecs`, which re-exports it.

use proc_macro::TokenStream;
use quote::quote;
use syn::punctuated::Punctuated;
use syn::{DeriveInput, Path, Token, parse_macro_input};

/// Implements `Component`, giving the type a `static` of its own. `#[component(...)]` lists
/// the type's traits from `dust_ecs::traits` by path, and each is implemented for the type.
#[proc_macro_derive(Component, attributes(component))]
pub fn derive_component(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    // A `static` in a generic impl is one item shared by every instantiation, so
    // `Foo<A>` and `Foo<B>` would use the same flecs component.
    if !input.generics.params.is_empty() {
        return syn::Error::new_spanned(
            &input.generics,
            "`Component` can't be derived for a generic type",
        )
        .to_compile_error()
        .into();
    }
    let mut traits = Vec::new();
    for attr in input.attrs.iter().filter(|attr| attr.path().is_ident("component")) {
        match attr.parse_args_with(Punctuated::<Path, Token![,]>::parse_terminated) {
            Ok(list) => traits.extend(list),
            Err(error) => return error.to_compile_error().into(),
        }
    }
    let name = input.ident;
    // Only `traits` keep the spans they were written with, which is what lets go-to-definition
    // reach a trait from the attribute. Don't use `quote_spanned!` here: go-to-definition
    // would then also resolve the other tokens given that span, such as `DECLARATION`.
    quote! {
        unsafe impl ::dust_ecs::Component for #name {
            const TRAITS: &'static [::dust_ecs::traits::Declaration] =
                &::dust_ecs::traits::singleton_last([#(<Self as #traits>::DECLARATION),*]);

            unsafe fn slot() -> &'static ::std::sync::atomic::AtomicI32 {
                static SLOT: ::std::sync::atomic::AtomicI32 =
                    ::std::sync::atomic::AtomicI32::new(0);
                &SLOT
            }
        }
        #(unsafe impl #traits for #name {})*
    }
    .into()
}
