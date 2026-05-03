//! `nflc` CLI binary.
//!
//! Subcommands:
//! - `nflc`                     → print usage to stdout, exit 0
//! - `nflc parse <file>`        → pretty-print AST to stdout, exit 0 (or err to stderr, exit 1)
//! - `nflc parse <file> --tokens` → pretty-print token stream to stdout
//! - `nflc parse <file> --uir`    → build and pretty-print the UIR

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
        _ => {
            eprintln!("error: unknown invocation");
            print_usage();
            ExitCode::FAILURE
        }
    }
}

fn print_usage() {
    println!("nflc — NFL Compiler (Milestone 3c)");
    println!();
    println!("USAGE:");
    println!("  nflc parse <file.nfl>            Parse and pretty-print the AST");
    println!("  nflc parse <file.nfl> --tokens   Print the lexer's token stream");
    println!("  nflc parse <file.nfl> --uir      Build and pretty-print the UIR");
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
        match nflc::lexer::lex(&source) {
            Ok(tokens) => {
                for t in tokens {
                    println!("{:>3}:{:<3}  {:?}", t.line, t.col, t.kind);
                }
                ExitCode::SUCCESS
            }
            Err(e) => {
                let (line, col) = e.position();
                render_error_with_snippet(&source, &path, line, col, &format!("{}", e));
                ExitCode::FAILURE
            }
        }
    } else {
        match nflc::parse(&source) {
            Ok(ast) => {
                print_ast(&ast);
                ExitCode::SUCCESS
            }
            Err(e) => {
                render_error_with_snippet(&source, &path, e.line, e.col, &e.message);
                ExitCode::FAILURE
            }
        }
    }
}

fn print_ast(nfl: &nflc::NflSource) {
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

fn print_stmt(s: &nflc::ModelStmt, depth: usize) {
    let pad = "  ".repeat(depth);
    match s {
        nflc::ModelStmt::VariableDecl(v) => {
            println!("{pad}var {} : Tensor[{}]", v.name, format_dims(&v.ty.dims));
        }
        nflc::ModelStmt::Pipeline(ps) => {
            print!("{pad}pipeline {}", ps.source);
            for op in &ps.steps {
                print!(" -> {}", op.name);
                if !op.args.is_empty() {
                    print!("[");
                    for (i, a) in op.args.iter().enumerate() {
                        if i > 0 { print!(", "); }
                        match a {
                            nflc::OpArg::Positional(v) => print!("{}", format_arg(v)),
                            nflc::OpArg::Named { name, value } => print!("{name}={}", format_arg(value)),
                        }
                    }
                    print!("]");
                }
            }
            println!();
        }
    }
}

fn format_dims(dims: &[nflc::Dim]) -> String {
    dims.iter()
        .map(|d| match d {
            nflc::Dim::Integer(n) => n.to_string(),
            nflc::Dim::Symbol(s) => s.clone(),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn format_arg(a: &nflc::ArgValue) -> String {
    match a {
        nflc::ArgValue::Integer(n) => n.to_string(),
        nflc::ArgValue::Float(f) => format!("{f}"),
        nflc::ArgValue::Symbol(s) => s.clone(),
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
    match nflc::parse(&source) {
        Ok(ast) => match nflc::ir::build(&ast) {
            Ok(uir) => {
                print!("{}", uir);
                ExitCode::SUCCESS
            }
            Err(e) => {
                render_error_with_snippet(&source, &path, e.line, e.col, &e.message);
                ExitCode::FAILURE
            }
        },
        Err(e) => {
            render_error_with_snippet(&source, &path, e.line, e.col, &e.message);
            ExitCode::FAILURE
        }
    }
}
