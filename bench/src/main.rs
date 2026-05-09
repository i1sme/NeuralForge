// SPDX-License-Identifier: Apache-2.0

//! NeuralForge bench harness (OQ-BENCH closure, M11).
//!
//! Compiles fixed NFL fixtures through the host-native profile,
//! times warmup × 10 + measurement × 100 FFI calls, emits markdown
//! to stdout (intended for `$GITHUB_STEP_SUMMARY` in CI).

use rand::distributions::Standard;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use std::path::PathBuf;
use std::time::Instant;

use compiler::Uir;
use profile_api::{Asm, Profile};

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// One fixture's measurement outcome. Owned form passed to `render_report`.
struct FixtureResult {
    name: &'static str,
    purpose: &'static str,
    median_ns: u64,
    p95_ns: u64,
}

/// Parsed CLI arguments. All fields owned to outlive `args`.
#[derive(Debug)]
struct BenchArgs {
    profile: String,
    format: String,
    seed: u64,
}

// ---------------------------------------------------------------------------
// Rendering helpers
// ---------------------------------------------------------------------------

/// Format a nanosecond duration as a microsecond string. Below 1000 µs
/// uses one decimal place (`48.7`); at or above 1000 µs uses integer
/// rounded half-up (`2845`). Per spec §7.3 — small fixtures benefit
/// from decimal precision, large ones don't.
fn format_us(ns: u64) -> String {
    let us_x10 = (ns + 50) / 100; // round to one decimal of µs
    if us_x10 < 10_000 {
        // < 1000.0 µs
        format!("{}.{}", us_x10 / 10, us_x10 % 10)
    } else {
        // >= 1000 µs: integer (round half-up)
        format!("{}", (ns + 500) / 1_000)
    }
}

/// Render a per-profile markdown report (spec §7.1 layout).
fn render_report(
    profile: &str,
    host: &str,
    seed: u64,
    git_short_sha: &str,
    results: &[FixtureResult],
) -> String {
    let mut s = String::new();
    s.push_str(&format!("## NeuralForge bench — {profile} profile\n\n"));
    s.push_str(&format!("**Host:** {host}\n"));
    s.push_str("**Build:** release, single-thread\n");
    s.push_str("**Methodology:** 10 warmup + 100 measurement, median + p95\n");
    s.push_str(&format!("**Seed:** {seed} (StdRng::seed_from_u64)\n"));
    s.push_str(&format!("**Compiled at:** {git_short_sha}\n\n"));
    s.push_str(
        "| Fixture              | Median (µs) | p95 (µs) | Purpose                          |\n",
    );
    s.push_str(
        "|----------------------|-------------|----------|----------------------------------|\n",
    );
    for r in results {
        s.push_str(&format!(
            "| {:<20} | {:>11} | {:>8} | {:<32} |\n",
            r.name,
            format_us(r.median_ns),
            format_us(r.p95_ns),
            r.purpose,
        ));
    }
    s
}

// ---------------------------------------------------------------------------
// Statistics helpers
// ---------------------------------------------------------------------------

/// Returns the strict median of `samples` in nanoseconds.
/// Sorts `samples` in-place. Caller is responsible for length contract:
/// requires `samples.len() >= 2` and even N for the canonical bench
/// case (N = 100 measurements).
fn median_ns(samples: &mut [u64]) -> u64 {
    samples.sort_unstable();
    let n = samples.len();
    // Strict median for even N: average of the two central elements.
    // Spec §6.2: for N=100, indices 49 and 50.
    let lo = samples[n / 2 - 1];
    let hi = samples[n / 2];
    (lo + hi) / 2
}

/// Returns the 95th percentile of `samples`. Sorts `samples` in-place.
/// Index convention: for N=100, returns `samples[95]`. Caller is
/// responsible for length contract: requires `samples.len() >= 96`
/// for the canonical bench case.
fn p95_ns(samples: &mut [u64]) -> u64 {
    samples.sort_unstable();
    samples[(samples.len() * 95) / 100]
}

// ---------------------------------------------------------------------------
// Deterministic data fill
// ---------------------------------------------------------------------------

/// Fill `buffer` with deterministic-random `f32` values from
/// `Standard` distribution (range `[0.0, 1.0)`). Spec §6.5.
fn fill_random(buffer: &mut [f32], seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    for v in buffer.iter_mut() {
        *v = rng.sample(Standard);
    }
}

// ---------------------------------------------------------------------------
// CLI flag parsing
// ---------------------------------------------------------------------------

/// Hand-rolled flag parser (no `clap` dependency — three flags doesn't
/// justify the dep). Recognises:
/// - `--profile {arm64|x86_64}` (required)
/// - `--format {markdown|github-summary}` (required)
/// - `--seed <u64>` (optional, default 42)
fn parse_args(args: &[String]) -> Result<BenchArgs, String> {
    let mut profile: Option<String> = None;
    let mut format: Option<String> = None;
    let mut seed: u64 = 42;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--profile" => {
                i += 1;
                let v = args.get(i).ok_or("missing value after --profile")?;
                match v.as_str() {
                    "arm64" | "x86_64" => profile = Some(v.clone()),
                    other => return Err(format!("unknown --profile {other}")),
                }
                i += 1;
            }
            "--format" => {
                i += 1;
                let v = args.get(i).ok_or("missing value after --format")?;
                match v.as_str() {
                    "markdown" | "github-summary" => format = Some(v.clone()),
                    other => return Err(format!("unknown --format {other}")),
                }
                i += 1;
            }
            "--seed" => {
                i += 1;
                let v = args.get(i).ok_or("missing value after --seed")?;
                seed = v
                    .parse()
                    .map_err(|e| format!("bad --seed value '{v}': {e}"))?;
                i += 1;
            }
            other => return Err(format!("unknown flag {other}")),
        }
    }

    Ok(BenchArgs {
        profile: profile.ok_or("--profile is required")?,
        format: format.ok_or("--format is required")?,
        seed,
    })
}

// ---------------------------------------------------------------------------
// Compile-and-load helper
// ---------------------------------------------------------------------------

/// Assemble + link `asm_source` to a host-native shared library and
/// return its path. Tempdir under `std::env::temp_dir()/nflc-bench-<pid>/`.
///
/// Spec §5.4: macOS arm64 → `cc -shared -arch arm64 -o lib<name>.dylib`.
/// Linux x86_64 → `cc -shared -fPIC -o lib<name>.so`.
/// Other host/profile combinations error out: bench does not
/// cross-execute (Rosetta/qemu would skew the SSE2 baseline).
fn compile_to_dylib_for_host(
    asm_source: &str,
    name: &str,
    requested_profile: &str,
) -> Result<PathBuf, String> {
    let host_arch = std::env::consts::ARCH;
    let host_os = std::env::consts::OS;

    // Validate host can natively run the requested profile (spec §5.4).
    match (requested_profile, host_arch, host_os) {
        ("arm64", "aarch64", "macos") => {}
        ("x86_64", "x86_64", "linux") => {}
        (p, a, o) => {
            return Err(format!(
                "host ({o}/{a}) cannot natively run --profile {p}; \
                 cross-execution unsupported"
            ));
        }
    }

    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("nflc-bench-{pid}"));
    std::fs::create_dir_all(&dir)
        .map_err(|e| format!("cannot create tempdir {}: {e}", dir.display()))?;

    let s_path = dir.join(format!("{name}.s"));
    std::fs::write(&s_path, asm_source)
        .map_err(|e| format!("cannot write {}: {e}", s_path.display()))?;

    let (lib_path, mut cc_args) = match requested_profile {
        "arm64" => (
            dir.join(format!("lib{name}.dylib")),
            vec!["-shared", "-arch", "arm64"],
        ),
        "x86_64" => (dir.join(format!("lib{name}.so")), vec!["-shared", "-fPIC"]),
        // Already validated above.
        _ => unreachable!(),
    };
    cc_args.push("-o");

    let mut cmd = std::process::Command::new("cc");
    cmd.args(&cc_args).arg(&lib_path).arg(&s_path);
    if requested_profile == "x86_64" {
        // Linux needs explicit libm linkage for `expf` (softmax codegen
        // emits `call expf@PLT`). macOS arm64 gets it implicitly via
        // libsystem. Order matters: `-lm` must follow `<src>.s` because
        // Linux ld resolves symbols left-to-right (library after the
        // object that uses it).
        cmd.arg("-lm");
    }
    let status = cmd
        .status()
        .map_err(|e| format!("cc invocation failed to spawn: {e}"))?;
    if !status.success() {
        return Err(format!(
            "cc failed to assemble {}: {status}",
            s_path.display()
        ));
    }

    Ok(lib_path)
}

// ---------------------------------------------------------------------------
// Timing loop
// ---------------------------------------------------------------------------

/// Run warmup × 10 + measurement × 100 calls of `call`, return
/// per-iteration measurement nanoseconds. Spec §6.4 warmup rationale
/// (dyld/PLT lazy binding, cache warm-up).
///
/// `call` is a closure that performs one forward pass. The `_i` argument
/// is the iteration index (unused by callers; present to satisfy the
/// closure signature uniformly).
fn time_forward_closure<F: FnMut(usize)>(mut call: F) -> Vec<u64> {
    const WARMUP: usize = 10;
    const MEASURE: usize = 100;

    for i in 0..WARMUP {
        call(i);
    }

    let mut samples = Vec::with_capacity(MEASURE);
    for i in 0..MEASURE {
        let t0 = Instant::now();
        call(i);
        let dt = t0.elapsed().as_nanos() as u64;
        samples.push(dt);
    }
    samples
}

// ---------------------------------------------------------------------------
// Per-fixture bench function
// ---------------------------------------------------------------------------

/// Build per-input buffers and params buffer for a fixture using the seed
/// cascade (spec §9.6). Returns `(input_bufs, params, output)`.
///
/// Each input buffer `i` is filled with `fill_random(buf, seed + i)`.
/// The params buffer is filled with `fill_random(params, seed + n_inputs)`.
/// For N=1 this produces `inputs[0]` filled with `seed` and `params` filled
/// with `seed + 1` — bit-identical to M11 behaviour.
fn build_buffers_for_sig(
    sig: &profile_api::FnSig,
    seed: u64,
) -> (Vec<Vec<f32>>, Vec<f32>, Vec<f32>) {
    let n_inputs = sig.inputs_floats.len();
    let mut input_bufs: Vec<Vec<f32>> = Vec::with_capacity(n_inputs);
    for (i, &n_floats) in sig.inputs_floats.iter().enumerate() {
        let mut buf = vec![0f32; n_floats];
        fill_random(&mut buf, seed.wrapping_add(i as u64));
        input_bufs.push(buf);
    }
    let mut params = vec![0f32; sig.params_floats];
    fill_random(&mut params, seed.wrapping_add(n_inputs as u64));
    let output = vec![0f32; sig.output_floats];
    (input_bufs, params, output)
}

/// Compile + load + time one fixture, return its measurement summary.
///
/// Mutates `samples` Vec internally; sorts in-place inside `median_ns` /
/// `p95_ns` calls before extraction. Spec §5.5 step list.
///
/// # Safety
/// Calls `dlopen`-loaded foreign code with bench-allocated buffers
/// sized from `FnSig`. The contract is: profile codegen must obey
/// the FFI ABI it advertised in `Asm.functions[*]`, established and
/// validated by M3-M10 integration tests.
fn bench_one_fixture(
    fixture_path: &str,
    fixture_name: &'static str,
    purpose: &'static str,
    profile: &dyn Profile,
    profile_label: &str,
    seed: u64,
) -> Result<FixtureResult, String> {
    // 1. Read NFL source.
    let src =
        std::fs::read_to_string(fixture_path).map_err(|e| format!("read {fixture_path}: {e}"))?;

    // 2. Parse → AST.
    let ast = compiler::parse(&src).map_err(|e| format!("parse {fixture_name}: {e:?}"))?;

    // 3. Build → UIR.
    let uir = compiler::ir::build(&ast).map_err(|e| format!("build {fixture_name}: {e:?}"))?;

    // 4. Default pipeline (passes ON, spec §8.3).
    let passes = compiler::default_pipeline();
    let uir: Uir = compiler::run_pipeline(&uir, &passes)
        .map_err(|e| format!("pipeline {fixture_name}: {e:?}"))?;

    // 5. Lower to Asm.
    let asm: Asm = profile
        .lower(&uir)
        .map_err(|e| format!("lower {fixture_name}: {e:?}"))?;

    // Single-model fixtures (all M3-M10): take the first FnSig.
    let sig = asm
        .functions
        .first()
        .ok_or_else(|| format!("no functions in Asm for {fixture_name}"))?;

    // 6. Allocate and fill FFI buffers via seed cascade (spec §9.6).
    // For N=1 fixtures this is bit-identical to M11: inputs[0] gets `seed`,
    // params get `seed + 1`.
    let (input_bufs, params, mut output) = build_buffers_for_sig(sig, seed);

    // 7. Compile to dylib.
    let lib_path = compile_to_dylib_for_host(&asm.source, fixture_name, profile_label)?;

    // 8. dlopen + dlsym. Note: `libloading` + Mach-O `dlsym` strip
    // the leading `_` automatically, so we pass `sig.name` verbatim
    // (verified at plan synthesis time against the M3-M10 integration
    // test pattern: `lib.get(b"nfl_forward_M4Demo")` works on macos-14).
    //
    // Note: `params.as_ptr()` is non-null aligned for `params_floats == 0`
    // (Vec::as_ptr returns NonNull::dangling for empty vecs, which is
    // f32-aligned; the codegen never dereferences a zero-param buffer —
    // established by M10 self_attention contract).
    let lib = unsafe { libloading::Library::new(&lib_path) }
        .map_err(|e| format!("dlopen {}: {e}", lib_path.display()))?;

    // 9. Per-arity FFI dispatch + timing (spec §9.6). Each arm loads the
    // concrete extern "C" fn type, then runs warmup + measurement.
    // The Symbol borrow on `lib` ends when `mut samples` is bound (NLL).
    let n_inputs = input_bufs.len();
    let mut samples = match n_inputs {
        1 => {
            type Fn1 = unsafe extern "C" fn(*const f32, *const f32, *mut f32);
            let f: libloading::Symbol<Fn1> = unsafe {
                lib.get(sig.name.as_bytes())
                    .map_err(|e| format!("dlsym {}: {e}", sig.name))?
            };
            let f = *f;
            let in0 = input_bufs[0].as_ptr();
            let par = params.as_ptr();
            let out = output.as_mut_ptr();
            // SAFETY: buffers correctly sized per FnSig; f valid for lifetime of lib.
            time_forward_closure(|_| unsafe { f(in0, par, out) })
        }
        2 => {
            type Fn2 = unsafe extern "C" fn(*const f32, *const f32, *const f32, *mut f32);
            let f: libloading::Symbol<Fn2> = unsafe {
                lib.get(sig.name.as_bytes())
                    .map_err(|e| format!("dlsym {}: {e}", sig.name))?
            };
            let f = *f;
            let in0 = input_bufs[0].as_ptr();
            let in1 = input_bufs[1].as_ptr();
            let par = params.as_ptr();
            let out = output.as_mut_ptr();
            // SAFETY: buffers correctly sized per FnSig; f valid for lifetime of lib.
            time_forward_closure(|_| unsafe { f(in0, in1, par, out) })
        }
        3 => {
            type Fn3 =
                unsafe extern "C" fn(*const f32, *const f32, *const f32, *const f32, *mut f32);
            let f: libloading::Symbol<Fn3> = unsafe {
                lib.get(sig.name.as_bytes())
                    .map_err(|e| format!("dlsym {}: {e}", sig.name))?
            };
            let f = *f;
            let in0 = input_bufs[0].as_ptr();
            let in1 = input_bufs[1].as_ptr();
            let in2 = input_bufs[2].as_ptr();
            let par = params.as_ptr();
            let out = output.as_mut_ptr();
            // SAFETY: buffers correctly sized per FnSig; f valid for lifetime of lib.
            time_forward_closure(|_| unsafe { f(in0, in1, in2, par, out) })
        }
        n => unimplemented!(
            "bench: arity {n} not supported (M12 caps at N=4; current bench fixtures all N=1)"
        ),
    };
    // lib is explicitly dropped after all Symbol uses end (NLL).
    drop(lib);

    Ok(FixtureResult {
        name: fixture_name,
        purpose,
        median_ns: median_ns(&mut samples),
        p95_ns: p95_ns(&mut samples),
    })
}

// ---------------------------------------------------------------------------
// Main wiring + fixtures registry
// ---------------------------------------------------------------------------

/// Three fixtures, three orthogonal signals (spec §8.1). Hardcoded.
const FIXTURES: &[(&str, &str, &str)] = &[
    (
        "tests/fixtures/classifier.nfl",
        "classifier",
        "matmul-mass (3-layer MLP)",
    ),
    (
        "tests/fixtures/large_classifier_k.nfl",
        "large_classifier_k",
        "large-K inner-loop accumulator",
    ),
    (
        "tests/fixtures/self_attention.nfl",
        "self_attention",
        "expf/softmax dispatch overhead",
    ),
];

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let args = match parse_args(&argv) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("bench: {e}");
            eprintln!(
                "usage: bench --profile {{arm64|x86_64}} --format {{markdown|github-summary}} [--seed N]"
            );
            std::process::exit(2);
        }
    };

    // Profile dispatch.
    let profile_label = args.profile.clone();
    let profile: Box<dyn Profile> = match args.profile.as_str() {
        "arm64" => Box::new(profiles_arm64::Arm64Profile),
        "x86_64" => Box::new(profiles_x86_64::X86_64Profile),
        other => {
            eprintln!("bench: unreachable — parse_args validated profile, got {other}");
            std::process::exit(2);
        }
    };

    // Host metadata for the report header.
    let host = host_label(&args.format);
    let git_sha = git_short_sha().unwrap_or_else(|| "unknown".to_string());

    // Run all fixtures.
    let mut results = Vec::with_capacity(FIXTURES.len());
    for (path, name, purpose) in FIXTURES {
        eprintln!("bench: running {name}...");
        match bench_one_fixture(
            path,
            name,
            purpose,
            profile.as_ref(),
            &profile_label,
            args.seed,
        ) {
            Ok(r) => results.push(r),
            Err(e) => {
                eprintln!("bench: fixture {name} failed: {e}");
                std::process::exit(1);
            }
        }
    }

    // Render and print to stdout.
    let report = render_report(&profile_label, &host, args.seed, &git_sha, &results);
    print!("{report}");
}

/// Best-effort host label (spec §7.3).
/// In CI: prefers `RUNNER_OS` + `RUNNER_ARCH` env vars when set.
/// Locally: falls back to `std::env::consts::{OS,ARCH}`.
fn host_label(format: &str) -> String {
    if format == "github-summary" {
        if let (Ok(os), Ok(arch)) = (std::env::var("RUNNER_OS"), std::env::var("RUNNER_ARCH")) {
            return format!("{os}, {arch}");
        }
    }
    format!("{}, {}", std::env::consts::OS, std::env::consts::ARCH)
}

/// Best-effort `git rev-parse --short HEAD`. Returns `None` if `git`
/// isn't on PATH or fails.
fn git_short_sha() -> Option<String> {
    let out = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8(out.stdout).ok()?.trim().to_string())
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- B1: Stats helpers ---

    #[test]
    fn median_of_even_length_returns_average_of_two_middles() {
        // Spec §6.2: strict median for even N = (samples[N/2-1] + samples[N/2]) / 2
        // For N=100, indices 49 and 50.
        let mut samples = vec![10u64, 20, 30, 40, 50, 60, 70, 80, 90, 100];
        assert_eq!(median_ns(&mut samples), 55);
    }

    #[test]
    fn median_sorts_input_in_place() {
        let mut samples = vec![100u64, 50, 25, 75];
        let _ = median_ns(&mut samples);
        assert_eq!(samples, vec![25, 50, 75, 100]);
    }

    #[test]
    fn p95_of_100_samples_is_index_95() {
        // After sort, samples[95] is the p95.
        let mut samples: Vec<u64> = (0..100).collect();
        assert_eq!(p95_ns(&mut samples), 95);
    }

    #[test]
    fn p95_handles_repeated_values() {
        let mut samples = vec![10u64; 100];
        assert_eq!(p95_ns(&mut samples), 10);
    }

    // --- B2: Markdown rendering ---

    #[test]
    fn render_report_produces_stable_markdown_header() {
        let rows = vec![FixtureResult {
            name: "classifier",
            purpose: "matmul-mass (3-layer MLP)",
            median_ns: 2_845_300,
            p95_ns: 3_104_700,
        }];
        let report = render_report(
            "arm64",
            "macos-14, Apple M1 @ 3.2 GHz",
            42,
            "abc1234",
            &rows,
        );
        assert!(report.starts_with("## NeuralForge bench — arm64 profile\n"));
        assert!(report.contains("**Host:** macos-14, Apple M1 @ 3.2 GHz"));
        assert!(report.contains("**Methodology:** 10 warmup + 100 measurement, median + p95"));
        assert!(report.contains("**Seed:** 42"));
        assert!(report.contains("**Compiled at:** abc1234"));
        assert!(report.contains("| Fixture"));
        assert!(report.contains("| classifier"));
        assert!(report.contains("matmul-mass (3-layer MLP)"));
    }

    #[test]
    fn render_report_formats_microseconds_with_one_decimal_when_below_1000() {
        let rows = vec![FixtureResult {
            name: "tiny",
            purpose: "tiny",
            median_ns: 48_700, // 48.7 µs
            p95_ns: 52_100,    // 52.1 µs
        }];
        let report = render_report("arm64", "h", 42, "c", &rows);
        assert!(report.contains("48.7"), "expected 48.7 µs in: {report}");
        assert!(report.contains("52.1"), "expected 52.1 µs in: {report}");
    }

    #[test]
    fn render_report_formats_microseconds_as_integer_when_at_or_above_1000() {
        let rows = vec![FixtureResult {
            name: "large",
            purpose: "large",
            median_ns: 2_845_300, // 2845.3 µs → 2845 (rounds down, < 0.5)
            p95_ns: 3_104_700,    // 3104.7 µs → 3105 (rounds up, >= 0.5)
        }];
        let report = render_report("arm64", "h", 42, "c", &rows);
        assert!(
            report.contains(" 2845 "),
            "expected integer µs in: {report}"
        );
        assert!(
            report.contains(" 3105 "),
            "expected integer µs in: {report}"
        );
    }

    // --- B3: Deterministic data fill ---

    #[test]
    fn fill_random_is_deterministic_for_fixed_seed() {
        let mut a = vec![0f32; 100];
        let mut b = vec![0f32; 100];
        fill_random(&mut a, 42);
        fill_random(&mut b, 42);
        assert_eq!(a, b, "same seed must produce identical buffers");
    }

    #[test]
    fn fill_random_differs_for_different_seeds() {
        let mut a = vec![0f32; 100];
        let mut b = vec![0f32; 100];
        fill_random(&mut a, 42);
        fill_random(&mut b, 43);
        assert_ne!(a, b, "different seeds must produce different buffers");
    }

    // --- B4: CLI flag parsing ---

    #[test]
    fn parse_args_defaults_seed_to_42() {
        let args: Vec<String> = vec![
            "bench".into(),
            "--profile".into(),
            "arm64".into(),
            "--format".into(),
            "markdown".into(),
        ];
        let parsed = parse_args(&args).expect("parse");
        assert_eq!(parsed.profile, "arm64");
        assert_eq!(parsed.format, "markdown");
        assert_eq!(parsed.seed, 42);
    }

    #[test]
    fn parse_args_accepts_seed_override() {
        let args: Vec<String> = vec![
            "bench".into(),
            "--profile".into(),
            "x86_64".into(),
            "--format".into(),
            "github-summary".into(),
            "--seed".into(),
            "7".into(),
        ];
        let parsed = parse_args(&args).expect("parse");
        assert_eq!(parsed.profile, "x86_64");
        assert_eq!(parsed.format, "github-summary");
        assert_eq!(parsed.seed, 7);
    }

    #[test]
    fn parse_args_rejects_unknown_profile() {
        let args: Vec<String> = vec![
            "bench".into(),
            "--profile".into(),
            "riscv64".into(),
            "--format".into(),
            "markdown".into(),
        ];
        let err = parse_args(&args).unwrap_err();
        assert!(err.contains("unknown --profile"), "got: {err}");
    }

    #[test]
    fn parse_args_rejects_unknown_format() {
        let args: Vec<String> = vec![
            "bench".into(),
            "--profile".into(),
            "arm64".into(),
            "--format".into(),
            "json".into(),
        ];
        let err = parse_args(&args).unwrap_err();
        assert!(err.contains("unknown --format"), "got: {err}");
    }

    // --- E1: Seed cascade ---

    #[test]
    fn seed_cascade_three_inputs_is_deterministic_and_independent() {
        // Synthetic FnSig with 3 inputs (spec §9.6 verification).
        let sig = profile_api::FnSig {
            name: "test_cascade".to_string(),
            model: "TestCascade".to_string(),
            inputs_floats: vec![10, 20, 30],
            params_floats: 5,
            output_floats: 4,
            params_layout: vec![],
        };
        let seed: u64 = 42;

        // Build buffers via cascade.
        let (input_bufs, params, _output) = build_buffers_for_sig(&sig, seed);

        // Three input buffers produced.
        assert_eq!(input_bufs.len(), 3);
        assert_eq!(input_bufs[0].len(), 10);
        assert_eq!(input_bufs[1].len(), 20);
        assert_eq!(input_bufs[2].len(), 30);
        assert_eq!(params.len(), 5);

        // Recompute independently with the same seeds — must match exactly.
        let mut ref0 = vec![0f32; 10];
        fill_random(&mut ref0, 42);
        let mut ref1 = vec![0f32; 20];
        fill_random(&mut ref1, 43);
        let mut ref2 = vec![0f32; 30];
        fill_random(&mut ref2, 44);
        let mut refp = vec![0f32; 5];
        fill_random(&mut refp, 45);

        assert_eq!(input_bufs[0], ref0, "input[0] must match seed 42");
        assert_eq!(input_bufs[1], ref1, "input[1] must match seed 43");
        assert_eq!(input_bufs[2], ref2, "input[2] must match seed 44");
        assert_eq!(params, refp, "params must match seed 45 (n_inputs=3)");
    }
}
