use proc_macro::TokenStream;
use syn::__private::ToTokens;
use syn::{Abi, ItemFn, LitStr, parse_macro_input};

/// Attribute macro applied to a function in order to specify its calling
/// convention. The macro argument is a [ABI] string literal, and it takes
/// priority over the [`extern`] qualifier (if any).
///
/// # Example
///
/// Define a function that follows the `"C"` calling convention in all
/// platforms except for Windows on the x86-64 architecture, where it
/// follows the `"sysv64"` calling convention.
///
/// ```
/// use coro_macros::abi;
///
/// #[cfg_attr(all(windows, target_arch = "x86_64"), abi("sysv64"))]
/// extern "C" fn foo(val: u64) {
///   println!("the `val` argument is {val}");
/// }
/// ```
///
/// [ABI]: https://doc.rust-lang.org/reference/glossary.html#r-glossary.abi
/// [`extern`]: https://doc.rust-lang.org/reference/items/functions.html#extern-function-qualifier
#[proc_macro_attribute]
pub fn abi(args: TokenStream, item: TokenStream) -> TokenStream {
  let abi_lit = parse_macro_input!(item as LitStr);
  let mut fun = parse_macro_input!(args as ItemFn);
  fun.sig.abi = Some(Abi {
    extern_token: Default::default(),
    name: Some(abi_lit),
  });
  fun.into_token_stream().into()
}
