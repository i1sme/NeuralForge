// SPDX-License-Identifier: Apache-2.0

//! `nflc` CLI binary.
//!
//! Subcommands:
//! - `nflc`                     → print usage to stdout, exit 0
//! - `nflc parse <file>`        → pretty-print AST to stdout, exit 0 (or err to stderr, exit 1)
//! - `nflc parse <file> --tokens` → pretty-print token stream to stdout
//! - `nflc parse <file> --uir`    → build and pretty-print the UIR
//! - `nflc parse <file> --uir-verbose` → build and pretty-print the UIR with annotated metadata
//! - `nflc compile <file> --profile <arm64|x86_64>` → lower UIR to assembly (arm64 Mach-O or x86_64 Linux ELF)
//! - `nflc compile <file> --profile <name> -o <file.s>` → lower UIR to assembly, write to file
//! - `nflc compile <file> --profile <name> [--no-passes]` → skip optimisation passes
//! - `nflc compile <file> --profile <name> [--passes <list>]` → run only listed passes
//! - `nflc inspect <file> --profile <arm64|x86_64>` → inspect post-pass UIR with profile annotations
//! - `nflc inspect <file> --profile <name> [--no-passes]` → skip passes during inspection
//! - `nflc inspect <file> --profile <name> [--passes <list>]` → filter passes during inspection

use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [] => {
            print_usage();
            ExitCode::SUCCESS
        }
        [cmd] if cmd == "parse" => {
            eprintln!("error: 'parse' requires a file path");
            print_usage();
            ExitCode::FAILURE
        }
        [cmd, path] if cmd == "parse" => run_parse(PathBuf::from(path), false),
        [cmd, path, flag] if cmd == "parse" && flag == "--tokens" => {
            run_parse(PathBuf::from(path), true)
        }
        [cmd, path, flag] if cmd == "parse" && flag == "--uir" => {
            run_build_uir(PathBuf::from(path))
        }
        [cmd, path, flag] if cmd == "parse" && flag == "--uir-verbose" => {
            run_build_uir_verbose(PathBuf::from(path))
        }
        [cmd, path, f1, f2]
            if cmd == "parse"
                && ((f1 == "--uir" && f2 == "--uir-verbose")
                    || (f1 == "--uir-verbose" && f2 == "--uir")) =>
        {
            eprintln!("error: --uir and --uir-verbose are mutually exclusive");
            ExitCode::FAILURE
        }
        [cmd, rest @ ..] if cmd == "inspect" => match parse_inspect_args(rest) {
            Ok(parsed) => run_inspect(parsed),
            Err(msg) => {
                eprintln!("error: {}", msg);
                print_usage();
                ExitCode::FAILURE
            }
        },
        [cmd, rest @ ..] if cmd == "compile" => match parse_compile_args(rest) {
            Ok(parsed) => run_compile(parsed),
            Err(msg) => {
                eprintln!("error: {}", msg);
                print_usage();
                ExitCode::FAILURE
            }
        },
        _ => {
            eprintln!("error: unknown invocation");
            print_usage();
            ExitCode::FAILURE
        }
    }
}

fn print_usage() {
    println!("nflc — NFL Compiler");
    println!();
    println!("USAGE:");
    println!("  nflc parse   <file.nfl>                    Parse and pretty-print the AST");
    println!("  nflc parse   <file.nfl> --tokens           Print the lexer's token stream");
    println!("  nflc parse   <file.nfl> --uir              Build and pretty-print the UIR");
    println!("  nflc parse   <file.nfl> --uir-verbose      Print UIR with annotated metadata");
    println!("  nflc compile <file.nfl> --profile <arm64|x86_64>   Lower UIR to assembly");
    println!("                          [-o <file.s>]      Output path (default: stdout)");
    println!("                          [--no-passes]      Skip optimisation passes (debugging)");
    println!(
        "                          [--passes <list>]  Run only listed passes (comma-separated)"
    );
    println!("  nflc inspect <file.nfl> --profile <arm64|x86_64>   Inspect post-pass UIR with profile annotations");
    println!("                          [--no-passes]      Skip optimisation passes");
    println!(
        "                          [--passes <list>]  Run only listed passes (comma-separated)"
    );
}

// ---------------------------------------------------------------------------
// Shared pass-flag helpers (used by both `compile` and `inspect`).
// ---------------------------------------------------------------------------

/// Parse the `--no-passes` / `--passes <list>` flag pair from a flag
/// iterator. Returns `Ok(true)` if the arg was consumed, `Ok(false)` if
/// the caller should handle it, or `Err` on a malformed value.
/// Shared by `nflc compile` and `nflc inspect`.
fn parse_pass_flag(
    arg: &str,
    iter: &mut std::slice::Iter<'_, String>,
    no_passes: &mut bool,
    passes: &mut Option<Vec<String>>,
) -> Result<bool, String> {
    match arg {
        "--no-passes" => {
            *no_passes = true;
            Ok(true)
        }
        "--passes" => {
            let v = iter
                .next()
                .ok_or_else(|| "--passes requires a value".to_string())?;
            if v.is_empty() {
                return Err(
                    "--passes value cannot be empty (use --no-passes to skip the pipeline)"
                        .to_string(),
                );
            }
            let names: Vec<String> = v.split(',').map(str::to_owned).collect();
            if names.iter().any(|n| n.is_empty()) {
                return Err(format!(
                    "--passes value '{v}' contains an empty token (use --no-passes for empty)"
                ));
            }
            *passes = Some(names);
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn validate_pass_args(no_passes: bool, passes: &Option<Vec<String>>) -> Result<(), String> {
    if no_passes && passes.is_some() {
        return Err("--no-passes and --passes are mutually exclusive".to_string());
    }
    if let Some(names) = passes {
        let available_names: Vec<String> = compiler::passes::default_pipeline()
            .iter()
            .map(|p| p.name().to_owned())
            .collect();
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for n in names {
            if !seen.insert(n.as_str()) {
                return Err(format!("pass '{n}' specified more than once in --passes"));
            }
        }
        for n in names {
            if !available_names.iter().any(|c| c == n) {
                return Err(format!(
                    "unknown pass '{n}' (available: {})",
                    available_names.join(", ")
                ));
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// compile subcommand
// ---------------------------------------------------------------------------

struct CompileArgs {
    path: PathBuf,
    profile: String,
    output: Option<PathBuf>,
    no_passes: bool,
    /// `None` = run `default_pipeline()`; `Some(list)` = filter to listed
    /// names (canonical order preserved regardless of user order).
    passes: Option<Vec<String>>,
}

fn parse_compile_args(args: &[String]) -> Result<CompileArgs, String> {
    let mut iter = args.iter();
    let path = iter
        .next()
        .ok_or_else(|| "compile: missing <file.nfl>".to_string())?
        .clone();
    if path.starts_with('-') {
        return Err(format!(
            "compile: expected <file.nfl> as first argument, got flag '{path}'"
        ));
    }

    let mut profile: Option<String> = None;
    let mut output: Option<PathBuf> = None;
    let mut no_passes = false;
    let mut passes: Option<Vec<String>> = None;

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--profile" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--profile requires a value".to_string())?;
                profile = Some(v.clone());
            }
            "-o" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "-o requires a value".to_string())?;
                output = Some(PathBuf::from(v));
            }
            other => {
                if !parse_pass_flag(other, &mut iter, &mut no_passes, &mut passes)? {
                    return Err(format!("unknown flag: {other}"));
                }
            }
        }
    }

    let profile = profile.ok_or_else(|| "compile: missing --profile <name>".to_string())?;
    validate_pass_args(no_passes, &passes)?;

    Ok(CompileArgs {
        path: PathBuf::from(path),
        profile,
        output,
        no_passes,
        passes,
    })
}

// ---------------------------------------------------------------------------
// inspect subcommand
// ---------------------------------------------------------------------------

struct InspectArgs {
    path: PathBuf,
    profile: String,
    no_passes: bool,
    passes: Option<Vec<String>>,
}

fn parse_inspect_args(args: &[String]) -> Result<InspectArgs, String> {
    let mut iter = args.iter();
    let path = iter
        .next()
        .ok_or_else(|| "inspect: missing <file.nfl>".to_string())?
        .clone();
    if path.starts_with('-') {
        return Err(format!(
            "inspect: expected <file.nfl> as first argument, got flag '{path}'"
        ));
    }

    let mut profile: Option<String> = None;
    let mut no_passes = false;
    let mut passes: Option<Vec<String>> = None;

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--profile" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--profile requires a value".to_string())?;
                profile = Some(v.clone());
            }
            other => {
                if !parse_pass_flag(other, &mut iter, &mut no_passes, &mut passes)? {
                    return Err(format!("unknown flag: {other}"));
                }
            }
        }
    }

    let profile = profile.ok_or_else(|| "inspect: missing --profile <name>".to_string())?;
    validate_pass_args(no_passes, &passes)?;

    Ok(InspectArgs {
        path: PathBuf::from(path),
        profile,
        no_passes,
        passes,
    })
}

// ---------------------------------------------------------------------------
// Shared error rendering helper
// ---------------------------------------------------------------------------

/// Render an error with a source-snippet pointer. Output format mirrors
/// rustc / cargo:
///
/// ```text
/// error: <message>
///   --> <path>:<line>:<col>
///    |
/// 12 |     x -> dropout[rate=1.5] -> softmax
///    |                       ^
/// ```
fn render_error_with_snippet(
    source: &str,
    path: &Path,
    line: u32,
    col: u32,
    message: &str,
    first_span: Option<(u32, u32)>,
) {
    eprintln!("error: {}", message);
    eprintln!("  --> {}:{}:{}", path.display(), line, col);
    let line_idx = line.saturating_sub(1) as usize;
    if let Some(src_line) = source.lines().nth(line_idx) {
        let prefix = line.to_string();
        let pad = " ".repeat(prefix.len());
        eprintln!("{}  |", pad);
        eprintln!("{} | {}", prefix, src_line);
        let mut underline = String::with_capacity(col as usize);
        for _ in 1..col {
            underline.push(' ');
        }
        underline.push('^');
        eprintln!("{}  | {}", pad, underline);
    }
    if let Some((fl, fc)) = first_span {
        eprintln!(
            "note: previously defined at {}:{}:{}",
            path.display(),
            fl,
            fc
        );
    }
}

// ---------------------------------------------------------------------------
// Shared pass-pipeline runner (used by both run_compile and run_inspect)
// ---------------------------------------------------------------------------

/// Run (or skip) the UIR pass pipeline according to `no_passes` / `passes`
/// flags. Returns `(post_pass_uir, applied_pass_names)` where
/// `applied_pass_names` is `None` when passes were skipped.
///
/// Side effect: emits `note:` lines to stderr.
fn run_pass_pipeline(
    uir: compiler::Uir,
    source: &str,
    path: &Path,
    no_passes: bool,
    passes: Option<Vec<String>>,
) -> Result<(compiler::Uir, Option<Vec<String>>), ExitCode> {
    if no_passes {
        eprintln!("note: passes skipped (--no-passes)");
        return Ok((uir, None));
    }

    let canonical = compiler::passes::default_pipeline();
    let canonical_names: Vec<String> = canonical.iter().map(|p| p.name().to_owned()).collect();

    let (pipeline, divergent) = match passes {
        None => (canonical, false),
        Some(user_names) => {
            let user_set: std::collections::HashSet<&str> =
                user_names.iter().map(String::as_str).collect();
            let filtered: Vec<Box<dyn compiler::passes::UirPass>> = canonical
                .into_iter()
                .filter(|p| user_set.contains(p.name()))
                .collect();
            let canonical_filtered_names: Vec<&str> = filtered.iter().map(|p| p.name()).collect();
            let div = user_names.len() >= 2
                && user_names.iter().map(String::as_str).collect::<Vec<_>>()
                    != canonical_filtered_names;
            (filtered, div)
        }
    };

    match compiler::passes::run_pipeline(&uir, &pipeline) {
        Ok(u) => {
            let names: Vec<String> = pipeline.iter().map(|p| p.name().to_owned()).collect();
            eprintln!("note: applied passes: {}", names.join(", "));
            if divergent {
                eprintln!(
                    "note: pass order is canonical ({}); user-specified order ignored",
                    canonical_names.join(", ")
                );
            }
            Ok((u, Some(names)))
        }
        Err(e) => {
            let span = e.span();
            render_error_with_snippet(source, path, span.line, span.col, &format!("{}", e), None);
            Err(ExitCode::FAILURE)
        }
    }
}

// ---------------------------------------------------------------------------
// parse subcommand
// ---------------------------------------------------------------------------

fn run_parse(path: PathBuf, tokens_only: bool) -> ExitCode {
    let source = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read {}: {}", path.display(), e);
            return ExitCode::FAILURE;
        }
    };

    if tokens_only {
        match compiler::lexer::lex(&source) {
            Ok(tokens) => {
                for t in tokens {
                    println!("{:>3}:{:<3}  {:?}", t.line, t.col, t.kind);
                }
                ExitCode::SUCCESS
            }
            Err(e) => {
                let (line, col) = e.position();
                render_error_with_snippet(&source, &path, line, col, &format!("{}", e), None);
                ExitCode::FAILURE
            }
        }
    } else {
        match compiler::parse(&source) {
            Ok(ast) => {
                print_ast(&ast);
                ExitCode::SUCCESS
            }
            Err(e) => {
                render_error_with_snippet(&source, &path, e.line, e.col, &e.to_string(), None);
                ExitCode::FAILURE
            }
        }
    }
}

fn print_ast(nfl: &compiler::NflSource) {
    for m in &nfl.models {
        println!("model {} [", m.name);
        for p in &m.params {
            println!("  {} = {}", p.name, p.value);
        }
        println!("]:");
        for stmt in &m.body {
            print_stmt(stmt, 1);
        }
        println!();
    }
}

fn print_stmt(s: &compiler::ModelStmt, depth: usize) {
    let pad = "  ".repeat(depth);
    match s {
        compiler::ModelStmt::VariableDecl(v) => {
            println!("{pad}var {} : Tensor[{}]", v.name, format_dims(&v.ty.dims));
        }
        compiler::ModelStmt::Pipeline(ps) => {
            print!("{pad}pipeline {}", ps.source);
            for op in &ps.steps {
                print!(" -> {}", op.name);
                if !op.args.is_empty() {
                    print!("[");
                    for (i, a) in op.args.iter().enumerate() {
                        if i > 0 {
                            print!(", ");
                        }
                        match a {
                            compiler::OpArg::Positional(v) => print!("{}", format_arg(v)),
                            compiler::OpArg::Named { name, value } => {
                                print!("{name}={}", format_arg(value))
                            }
                        }
                    }
                    print!("]");
                }
            }
            println!();
        }
        compiler::ModelStmt::NamedPipeline(np) => {
            print!(
                "{pad}named_pipeline {} : Tensor[{}] = {}",
                np.binding_name,
                format_dims(&np.declared_ty.dims),
                np.source
            );
            for op in &np.steps {
                print!(" -> {}", op.name);
                if !op.args.is_empty() {
                    print!("[");
                    for (i, a) in op.args.iter().enumerate() {
                        if i > 0 {
                            print!(", ");
                        }
                        match a {
                            compiler::OpArg::Positional(v) => print!("{}", format_arg(v)),
                            compiler::OpArg::Named { name, value } => {
                                print!("{name}={}", format_arg(value))
                            }
                        }
                    }
                    print!("]");
                }
            }
            println!();
        }
    }
}

fn format_dims(dims: &[compiler::Dim]) -> String {
    dims.iter()
        .map(|d| match d {
            compiler::Dim::Integer(n) => n.to_string(),
            compiler::Dim::Symbol(s) => s.clone(),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn format_arg(a: &compiler::ArgValue) -> String {
    match a {
        compiler::ArgValue::Integer(n) => n.to_string(),
        compiler::ArgValue::Float(f) => format!("{f}"),
        compiler::ArgValue::Symbol(s) => s.clone(),
    }
}

fn run_build_uir(path: PathBuf) -> ExitCode {
    let source = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read {}: {}", path.display(), e);
            return ExitCode::FAILURE;
        }
    };
    match compiler::parse(&source) {
        Ok(ast) => match compiler::ir::build(&ast) {
            Ok(uir) => {
                print!("{}", uir);
                ExitCode::SUCCESS
            }
            Err(e) => {
                render_error_with_snippet(&source, &path, e.line, e.col, &e.to_string(), None);
                ExitCode::FAILURE
            }
        },
        Err(e) => {
            render_error_with_snippet(&source, &path, e.line, e.col, &e.to_string(), None);
            ExitCode::FAILURE
        }
    }
}

fn run_build_uir_verbose(path: PathBuf) -> ExitCode {
    let source = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read {}: {}", path.display(), e);
            return ExitCode::FAILURE;
        }
    };
    match compiler::parse(&source) {
        Ok(ast) => match compiler::ir::build(&ast) {
            Ok(uir) => {
                use compiler::ir::types::VerboseUir;
                print!("{}", VerboseUir(&uir));
                ExitCode::SUCCESS
            }
            Err(e) => {
                render_error_with_snippet(&source, &path, e.line, e.col, &e.to_string(), None);
                ExitCode::FAILURE
            }
        },
        Err(e) => {
            render_error_with_snippet(&source, &path, e.line, e.col, &e.to_string(), None);
            ExitCode::FAILURE
        }
    }
}

// ---------------------------------------------------------------------------
// compile subcommand runner
// ---------------------------------------------------------------------------

fn run_compile(args: CompileArgs) -> ExitCode {
    let CompileArgs {
        path,
        profile,
        output: out_path,
        no_passes,
        passes,
    } = args;

    let source = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read {}: {}", path.display(), e);
            return ExitCode::FAILURE;
        }
    };

    let ast = match compiler::parse(&source) {
        Ok(a) => a,
        Err(e) => {
            render_error_with_snippet(&source, &path, e.line, e.col, &e.to_string(), None);
            return ExitCode::FAILURE;
        }
    };

    let uir = match compiler::ir::build(&ast) {
        Ok(u) => u,
        Err(e) => {
            let first = match &e.kind {
                compiler::BuildErrorKind::DuplicateModelName { first_span, .. } => {
                    Some((first_span.line, first_span.col))
                }
                _ => None,
            };
            let msg = e.to_string();
            render_error_with_snippet(&source, &path, e.line, e.col, &msg, first);
            return ExitCode::FAILURE;
        }
    };

    let profile_impl: Box<dyn profile_api::Profile> = match profile.as_str() {
        "arm64" => Box::new(profiles_arm64::Arm64Profile),
        "x86_64" => Box::new(profiles_x86_64::X86_64Profile),
        other => {
            eprintln!(
                "error: unknown profile '{}' (supported: arm64, x86_64)",
                other
            );
            return ExitCode::FAILURE;
        }
    };

    // M5b: run UIR-passes pipeline with optional filter, or skip
    // entirely if --no-passes. See spec §9.3.
    let post_pass_uir = match run_pass_pipeline(uir, &source, &path, no_passes, passes) {
        Ok((u, _names)) => u,
        Err(code) => return code,
    };

    match profile_impl.lower(&post_pass_uir) {
        Ok(asm) => match out_path {
            Some(p) => match std::fs::write(&p, &asm.source) {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("error: cannot write {}: {}", p.display(), e);
                    ExitCode::FAILURE
                }
            },
            None => {
                print!("{}", asm.source);
                ExitCode::SUCCESS
            }
        },
        Err(e) => {
            let span = e.span();
            render_error_with_snippet(&source, &path, span.line, span.col, &format!("{}", e), None);
            ExitCode::FAILURE
        }
    }
}

// ---------------------------------------------------------------------------
// inspect subcommand runner
// ---------------------------------------------------------------------------

fn run_inspect(args: InspectArgs) -> ExitCode {
    let InspectArgs {
        path,
        profile,
        no_passes,
        passes,
    } = args;

    let source = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read {}: {}", path.display(), e);
            return ExitCode::FAILURE;
        }
    };

    let ast = match compiler::parse(&source) {
        Ok(a) => a,
        Err(e) => {
            render_error_with_snippet(&source, &path, e.line, e.col, &e.to_string(), None);
            return ExitCode::FAILURE;
        }
    };

    let uir = match compiler::ir::build(&ast) {
        Ok(u) => u,
        Err(e) => {
            let first = match &e.kind {
                compiler::BuildErrorKind::DuplicateModelName { first_span, .. } => {
                    Some((first_span.line, first_span.col))
                }
                _ => None,
            };
            let msg = e.to_string();
            render_error_with_snippet(&source, &path, e.line, e.col, &msg, first);
            return ExitCode::FAILURE;
        }
    };

    let profile_impl: Box<dyn profile_api::Profile> = match profile.as_str() {
        "arm64" => Box::new(profiles_arm64::Arm64Profile),
        "x86_64" => Box::new(profiles_x86_64::X86_64Profile),
        other => {
            eprintln!(
                "error: unknown profile '{}' (supported: arm64, x86_64)",
                other
            );
            return ExitCode::FAILURE;
        }
    };

    let (post_pass_uir, applied_pass_names) =
        match run_pass_pipeline(uir, &source, &path, no_passes, passes) {
            Ok(pair) => pair,
            Err(code) => return code,
        };

    let inspection = match profile_impl.inspect(&post_pass_uir) {
        Ok(i) => i,
        Err(e) => {
            let span = e.span();
            render_error_with_snippet(&source, &path, span.line, span.col, &format!("{}", e), None);
            return ExitCode::FAILURE;
        }
    };

    // Render. Convert applied_pass_names to &[&str] for the renderer.
    let applied_refs: Option<Vec<&str>> = applied_pass_names
        .as_ref()
        .map(|v| v.iter().map(String::as_str).collect());
    let header = inspect_render::RenderHeader {
        source_path: &path,
        profile: &profile,
        applied_passes: applied_refs.as_deref(),
    };
    print!("{}", inspect_render::render_inspection(&inspection, header));
    ExitCode::SUCCESS
}
