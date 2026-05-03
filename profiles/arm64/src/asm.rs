//! Low-level AArch64 assembly building blocks.
//!
//! Helpers that emit common instruction sequences and label/symbol formatting.
//! No UIR knowledge here — pure asm-string assembly.

use crate::FnSig;

/// Mach-O symbol prefix. Apple's `as` prepends `_` to C symbol names.
pub const MACHO_SYM_PREFIX: &str = "_";

/// Format the function header: directives + globl + alignment + label.
pub fn format_function_header(sig: &FnSig) -> String {
    let mut out = String::new();
    out.push_str(&format!(".globl {}{}\n", MACHO_SYM_PREFIX, sig.name));
    out.push_str(".p2align 2\n");
    out.push_str(&format!("{}{}:\n", MACHO_SYM_PREFIX, sig.name));
    out
}

/// Format the function epilogue: `ret`.
pub fn format_function_footer() -> String {
    "    ret\n".to_string()
}
