use proc_macro2::TokenStream;
use syn::{
    Expr, Ident, Token, Type,
    parse::{Parse, ParseStream, Result},
    punctuated::Punctuated,
};

pub struct CuthcNDVecApiFn {
    pub fn_name: Ident,
    pub args_decl: Vec<(Ident, Type)>,
    pub ret_ty: Option<Type>,
}

impl Parse for CuthcNDVecApiFn {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let _: Token![fn] = input.parse()?;
        let fn_name: Ident = input.parse()?;
        let content;
        let _parentheses = syn::parenthesized!(content in input);
        let args_decl: Punctuated<(Ident, Type), Token![,]> =
            Punctuated::parse_terminated_with(&content, |input| {
                let ident: Ident = input.parse()?;
                let _: Token![:] = input.parse()?;
                let ty: Type = input.parse()?;
                Ok((ident, ty))
            })?;
        if input.peek(Token![->]) {
            let _: Token![->] = input.parse()?;
            let ret_ty: Type = input.parse()?;
            Ok(Self {
                fn_name,
                args_decl: args_decl.into_iter().collect(),
                ret_ty: Some(ret_ty),
            })
        } else {
            Ok(Self {
                fn_name,
                args_decl: args_decl.into_iter().collect(),
                ret_ty: None,
            })
        }
    }
}

pub struct CuthcNDVecApiN {
    pub n: syn::LitInt,
    pub fns: Vec<CuthcNDVecApiFn>,
}

impl Parse for CuthcNDVecApiN {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let n: syn::LitInt = input.parse()?;
        let _: Token![=>] = input.parse()?;
        // parse in curley braces
        let fns: Punctuated<CuthcNDVecApiFn, Token![;]> = Punctuated::parse_terminated(&input)?;
        Ok(Self {
            n,
            fns: fns.into_iter().collect(),
        })
    }
}

pub struct CuthcNDVecApiCallN {
    pub n: syn::LitInt,
    pub ret_var: Option<Ident>,
    pub fn_name: Ident,
    pub args: Punctuated<Expr, Token![,]>,
    pub semicolon: Option<Token![;]>,
    pub other_code: TokenStream,
}

impl Parse for CuthcNDVecApiCallN {
    fn parse(input: ParseStream) -> Result<Self> {
        let n: syn::LitInt = input.parse()?;
        let _: Token![=>] = input.parse()?;
        let ret_var;
        if input.peek(Token![let]) {
            let _: Token![let] = input.parse()?;
            ret_var = Some(input.parse()?);
            let _ : Token![=] = input.parse()?;
        } else {
            ret_var = None;
        }
        let fn_name: Ident = input.parse()?;
        let args_content;
        let _parentheses = syn::parenthesized!(args_content in input);
        let args: Punctuated<Expr, Token![,]> = Punctuated::parse_terminated(&args_content)?;
        let semicolon = if input.peek(Token![;]) {
            Some(input.parse()?)
        } else {
            None
        };
        let other_code = input.parse()?;

        Ok(Self {
            n,
            ret_var,
            fn_name,
            args,
            semicolon,
            other_code,
        })
    }
}
