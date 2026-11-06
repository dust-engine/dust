//! The derive for `dust_ecs::Component`. Use it through `dust_ecs`, which re-exports it.

use proc_macro::TokenStream;
use quote::quote;
use syn::{DeriveInput, parse_macro_input};

/// Implements `Component`, giving the type a `static` of its own.
#[proc_macro_derive(Component)]
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
    let name = input.ident;
    quote! {
        unsafe impl ::dust_ecs::Component for #name {
            unsafe fn slot() -> &'static ::std::sync::atomic::AtomicI32 {
                static SLOT: ::std::sync::atomic::AtomicI32 =
                    ::std::sync::atomic::AtomicI32::new(0);
                &SLOT
            }
        }
    }
    .into()
}
