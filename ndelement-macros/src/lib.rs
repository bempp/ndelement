use itertools::Itertools;
use proc_macro::TokenStream;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::{parse_macro_input, Ident, Lit, LitStr, Token};
use syn::{Expr, ExprArray};

struct DowncastWithConcreteTypes {
    var: Ident,
    op: String,
    types: Vec<Vec<String>>,
    keys: Vec<String>,
}

impl Parse for DowncastWithConcreteTypes {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        if input.is_empty() {
            panic!("Empty argument list.");
        }

        let mut type_entries = Vec::<Vec<String>>::new();
        let mut keys = Vec::<String>::new();

        // The initial var identifier.
        let var = input.parse::<Ident>()?;

        // Comma after identifier.
        input.parse::<Token![,]>()?;

        // Next need to past operation as string literal.
        let op = input.parse::<LitStr>()?.value();

        // Parse another comma.
        input.parse::<Token![,]>()?;

        // Now parse the key value pairs.
        while !input.is_empty() {
            let mut string_values = Vec::<String>::new();
            let key = input.parse::<Ident>()?.to_string();
            input.parse::<Token![=]>()?;
            let values = input.parse::<ExprArray>()?;
            for expr in values.elems {
                match expr {
                    Expr::Lit(content) => {
                        match content.lit {
                            Lit::Str(lit) => string_values.push(lit.value()),
                            _ => panic!("Only string literals are allowed in this array."),
                        };
                    }
                    _ => panic!("Only literals are allowed in the array."),
                }
            }

            type_entries.push(string_values);
            keys.push(key);

            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }

        Ok(DowncastWithConcreteTypes {
            var,
            op,
            types: type_entries,
            keys,
        })
    }
}

#[proc_macro]
pub fn downcast_with_concrete_types(item: TokenStream) -> TokenStream {
    let DowncastWithConcreteTypes {
        var,
        op,
        types,
        keys,
    } = parse_macro_input!(item as DowncastWithConcreteTypes);

    let op = op.replace("{{inner}}", &var.to_string());
    let mut output = quote!();

    for current_types in types.iter().map(|x| x.iter()).multi_cartesian_product() {
        // First do the template substitutions into concrete types. We assume
        // that later types can only depend on earlier types in the `types` array.

        let ntypes = types.len();

        let mut complete_types = current_types.iter().map(|x| (**x).clone()).collect_vec();

        for index in 0..ntypes - 1 {
            let replace_type = complete_types[index].clone();
            for ty in complete_types[index + 1..ntypes].iter_mut() {
                *ty = ty.replace(&("{{".to_owned() + &keys[index] + "}}"), &replace_type);
            }
        }

        // Now we have the complete types. We can now generate the if statements.

        // output = quote! {
        //    #output
        //    if let Some(op) = #var.downcast_ref::<
        //         #(
        //              #complete_types ,
        //         ),*
        //         >() {
        //             #op
        //             }
        // };
    }

    println!("{}", output);

    output.into()
}

// How to use this macro:
// downcast_with_concrete_types!(var, op = "{{inner}}.call(&my_op)" ,dtype = ["f32, f64"], elem = ["Element1<{{dtype}}>", "Element2<{{dtype}}>"]);
// This expands to the token stream:
//
// {
//  if let Some(inner) =
// }
