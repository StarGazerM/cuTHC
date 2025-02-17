use proc_macro::TokenStream;
use proc_macro::{Ident, TokenTree};
use quote::quote;

// a proc macro that concat 2 identifiers
#[proc_macro]
pub fn concat_idents(input: TokenStream) -> TokenStream {
    let mut iter = input.into_iter();
    let first = iter.next().unwrap();
    let second = iter.next().unwrap();
    let first = match first {
        TokenTree::Ident(ident) => ident,
        _ => panic!("expected an identifier"),
    };
    let second = match second {
        TokenTree::Ident(ident) => ident,
        _ => panic!("expected an identifier"),
    };
    let new_ident = Ident::new(&format!("{}{}", first, second), first.span());
    TokenStream::from(TokenTree::Ident(new_ident))
}

mod syntax;

#[proc_macro]
pub fn gen_cuthc_nvec_n(input: TokenStream) -> TokenStream {
    let fn_decls = syn::parse_macro_input!(input as syntax::CuthcNDVecApiN);

    let rep_num = fn_decls.n;
    // rename the function name
    let new_fns = fn_decls.fns.into_iter().flat_map(|fn_decl| {
        let fn_name = fn_decl.fn_name;
        let rep_num = rep_num.base10_parse::<usize>().unwrap();
        let ret_ty = fn_decl.ret_ty.clone();
        (1..rep_num+1).map(move |i| {
            let new_fn_name = syn::Ident::new(&format!("{}_{}", fn_name, i), fn_name.span());
            let args = fn_decl.args_decl.iter().map(|(arg_name, arg_ty)| {
                quote! {
                    #arg_name: #arg_ty
                }
            });
            let ret_ty = ret_ty.clone();
            if let Some(ret_ty) = ret_ty {
                quote! {
                    pub fn #new_fn_name(
                        #(#args),*
                    ) -> #ret_ty;
                }
            } else {
                quote! {
                    pub fn #new_fn_name(
                        #(#args),*
                    );
                }
            }
        })
    });

    quote! {
        #(#new_fns)*
    }.into()

}

#[proc_macro]
pub fn gen_cuthc_nvec_call_n(input: TokenStream) -> TokenStream {
    let call = syn::parse_macro_input!(input as syntax::CuthcNDVecApiCallN);

    let rep_num = call.n;
    let fn_name = call.fn_name;
    let ret_var = call.ret_var;
    let args = call.args;
    let other_code = call.other_code;

    let rep_num = rep_num.base10_parse::<usize>().unwrap();
    let call_handle_code = (1..rep_num+1).map(|i| {
        let new_fn_name = syn::Ident::new(&format!("{}_{}", fn_name, i), fn_name.span());
        let i_lit = syn::LitInt::new(&i.to_string(), fn_name.span());
        let semi_col = {
            if let Some(semicolon) = call.semicolon {
                quote! {
                    #semicolon
                }
            } else {
                quote! {}
            }
        };
        if let Some(ret_var) = &ret_var {
            quote! {
                #i_lit => {
                    let #ret_var = unsafe { #new_fn_name(#args) } #semi_col
                    #other_code
                }
            }
        } else {
            quote! {
                #i_lit => {
                    unsafe { #new_fn_name(#args) } #semi_col
                    #other_code
                }
            }
        }
    });

    quote! {
        match DIM {
            #(#call_handle_code)*
            _ => panic!("Unsupported dimension {}!", DIM),
        }
    }.into()
}
