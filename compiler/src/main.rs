//! `nflc` CLI binary.
//!
//! Subcommands:
//! - `nflc`                     → print usage to stdout, exit 0
//! - `nflc parse <file>`        → pretty-print AST to stdout, exit 0 (or err to stderr, exit 1)
//! - `nflc parse <file> --tokens` → pretty-print token stream to stdout

use std::path::PathBuf;
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
        _ => {
            eprintln!("error: unknown invocation");
            print_usage();
            ExitCode::FAILURE
        }
    }
}

fn print_usage() {
    println!("nflc — NFL Compiler (Milestone 2)");
    println!();
    println!("USAGE:");
    println!("  nflc parse <file.nfl>            Parse and pretty-print the AST");
    println!("  nflc parse <file.nfl> --tokens   Print the lexer's token stream");
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
                eprintln!("error: {} at {}:{}:{}", e, path.display(), line, col);
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
                eprintln!("error: {} at {}:{}:{}", e.message, path.display(), e.line, e.col);
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
