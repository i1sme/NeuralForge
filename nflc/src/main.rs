//! `nflc` CLI binary.
//!
//! Subcommands:
//! - `nflc`                     → print usage to stdout, exit 0
//! - `nflc parse <file>`        → pretty-print AST to stdout, exit 0 (or err to stderr, exit 1)
//! - `nflc parse <file> --tokens` → pretty-print token stream to stdout
//! - `nflc parse <file> --uir`    → build and pretty-print the UIR
//! - `nflc compile <file> --profile <name>` → lower UIR to assembly
//! - `nflc compile <file> --profile <name> -o <file.s>` → lower UIR to assembly, write to file
//! - `nflc compile <file> --profile <name> [--no-passes]` → skip optimisation passes
//! - `nflc compile <file> --profile <name> [--passes <list>]` → run only listed passes

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
    println!("  nflc compile <file.nfl> --profile <name>   Lower UIR to assembly");
    println!("                          [-o <file.s>]      Output path (default: stdout)");
    println!("                          [--no-passes]      Skip optimisation passes (debugging)");
    println!(
        "                          [--passes <list>]  Run only listed passes (comma-separated)"
    );
}

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
            "--no-passes" => {
                no_passes = true;
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
                // Strict split on `,` — no whitespace trimming. Users invoke
                // as --passes a,b or --passes "a,b" (no spaces inside).
                let names: Vec<String> = v.split(',').map(str::to_owned).collect();
                if names.iter().any(|n| n.is_empty()) {
                    return Err(format!(
                        "--passes value '{v}' contains an empty token (use --no-passes for empty)"
                    ));
                }
                passes = Some(names);
            }
            other => {
                return Err(format!("unknown flag: {other}"));
            }
        }
    }

    let profile = profile.ok_or_else(|| "compile: missing --profile <name>".to_string())?;

    // Mutually exclusive: --no-passes and --passes can't coexist.
    if no_passes && passes.is_some() {
        return Err("--no-passes and --passes are mutually exclusive".to_string());
    }

    // Validate --passes content against the canonical pass registry.
    // The list is for *validation only* here — `run_compile` builds the
    // actual filtered pipeline from `default_pipeline()` independently.
    if let Some(ref names) = passes {
        let available_names: Vec<String> = compiler::passes::default_pipeline()
            .iter()
            .map(|p| p.name().to_owned())
            .collect();

        // Duplicate check.
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for n in names {
            if !seen.insert(n.as_str()) {
                return Err(format!("pass '{n}' specified more than once in --passes"));
            }
        }

        // Unknown-name check (dynamic available list).
        for n in names {
            if !available_names.iter().any(|c| c == n) {
                return Err(format!(
                    "unknown pass '{n}' (available: {})",
                    available_names.join(", ")
                ));
            }
        }
    }

    Ok(CompileArgs {
        path: PathBuf::from(path),
        profile,
        output,
        no_passes,
        passes,
    })
}

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
                render_error_with_snippet(&source, &path, e.line, e.col, &e.message, None);
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
                render_error_with_snippet(&source, &path, e.line, e.col, &e.message, None);
                ExitCode::FAILURE
            }
        },
        Err(e) => {
            render_error_with_snippet(&source, &path, e.line, e.col, &e.message, None);
            ExitCode::FAILURE
        }
    }
}

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
            render_error_with_snippet(&source, &path, e.line, e.col, &e.message, None);
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

    if profile != "arm64" {
        eprintln!("error: unknown profile '{}' (supported: arm64)", profile);
        return ExitCode::FAILURE;
    }

    // M5b: run UIR-passes pipeline with optional filter, or skip
    // entirely if --no-passes. See spec §9.3.
    let post_pass_uir = if no_passes {
        eprintln!("note: passes skipped (--no-passes)");
        uir
    } else {
        let canonical = compiler::passes::default_pipeline();
        // Own the names (Vec<String>, not Vec<&str>) so the borrow on
        // `canonical` doesn't outlive the move into either match arm —
        // E0505 if Vec<&str> were used here (see spec §9.3).
        let canonical_names: Vec<String> = canonical.iter().map(|p| p.name().to_owned()).collect();

        let (pipeline, divergent) = match passes {
            None => (canonical, false),
            Some(user_names) => {
                // Filter canonical to retain only user-named passes,
                // preserving canonical order.
                let user_set: std::collections::HashSet<&str> =
                    user_names.iter().map(String::as_str).collect();
                let filtered: Vec<Box<dyn compiler::passes::UirPass>> = canonical
                    .into_iter()
                    .filter(|p| user_set.contains(p.name()))
                    .collect();
                let canonical_filtered_names: Vec<&str> =
                    filtered.iter().map(|p| p.name()).collect();
                // Order divergence: only meaningful when len >= 2.
                // user_names is Vec<String> (owned), canonical_filtered_names
                // is Vec<&str> (borrowed). Project user_names through
                // String::as_str into a Vec<&str> for type-aligned `!=`.
                let div = user_names.len() >= 2
                    && user_names.iter().map(String::as_str).collect::<Vec<_>>()
                        != canonical_filtered_names;
                (filtered, div)
            }
        };

        match compiler::passes::run_pipeline(&uir, &pipeline) {
            Ok(u) => {
                // Applied-note emitted only on success (M5a polish kept).
                let names: Vec<&str> = pipeline.iter().map(|p| p.name()).collect();
                eprintln!("note: applied passes: {}", names.join(", "));
                if divergent {
                    eprintln!(
                        "note: pass order is canonical ({}); user-specified order ignored",
                        canonical_names.join(", ")
                    );
                }
                u
            }
            Err(e) => {
                let span = e.span();
                render_error_with_snippet(
                    &source,
                    &path,
                    span.line,
                    span.col,
                    &format!("{}", e),
                    None,
                );
                return ExitCode::FAILURE;
            }
        }
    };

    match profiles_arm64::lower(&post_pass_uir) {
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
