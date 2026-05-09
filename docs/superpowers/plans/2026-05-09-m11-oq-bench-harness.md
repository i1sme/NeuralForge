# M11 — OQ-BENCH Harness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land a `bench/` workspace crate that compiles three NFL fixtures through one profile (arm64 OR x86_64 — host's native arch) and writes per-profile latency reports to GitHub Actions Job Summary, closing the OQ-BENCH trigger that fired on M9 merge.

**Architecture:** New workspace member `bench/` with a single-file `main.rs` (~300 lines): CLI flag parsing → fixture registry (3 hardcoded fixtures) → for each fixture: parse → build UIR → `default_pipeline` → `profile.lower` → `cc -shared` → `libloading::Library::new` → warmup × 10 + measurement × 100 → median + p95 → markdown row. New CI workflow `.github/workflows/bench.yml` runs the harness on a 2-leg matrix (`macos-14` + `ubuntu-latest`), each leg writing to `$GITHUB_STEP_SUMMARY`. No artifact sharing between legs.

**Tech Stack:** Rust workspace, `compiler` crate (parse + UIR build + pass pipeline), `profile-api` (`Asm`, `FnSig`, `Profile` trait), `profiles-arm64` (`Arm64Profile`), `profiles-x86_64` (`X86_64Profile`), `libloading 0.8` (dlopen), `rand 0.8` (deterministic seed), system `cc` (assembling `.s` to `.dylib`/`.so`).

**Spec:** `docs/superpowers/specs/2026-05-09-m11-oq-bench-harness-design.md`. Plan terminology and ordering match spec §s 1–14.

---

## File Structure

### Files to create
- `bench/Cargo.toml` — new workspace member manifest.
- `bench/src/main.rs` — single-file harness, ~300 lines.
- `.github/workflows/bench.yml` — new CI workflow (two-leg matrix).

### Files to modify
- `Cargo.toml` (workspace root) — add `"bench"` to `[workspace] members`, **first alphabetically** (per spec §5.3).
- `PROJECT_SPEC.md` — close OQ-BENCH under "Trigger-driven cleanup" (matching OQ-NEW closure pattern); bump "First Milestones" with M11 row; bump "Current Status" reference.
- `CLAUDE.md` — add `bench/` member to Repository Structure; bump "Current Status" to M11.
- `DEVLOG.md` — milestone closure entry at the top.

### Files NOT to touch
- `.github/workflows/ci.yml` — spec §12 forbids; bench.yml is independent.
- `compiler/src/`, `profile-api/src/`, `profiles/*/src/` — bench is consumer-only; spec §12 forbids profile/compiler edits.
- `tests/fixtures/*.nfl` — no new fixtures; spec §4.2 rejects.
- `nflc/` — bench is its own bin, doesn't extend `nflc`.

---

## Pre-Flight Verification

Run from worktree root before Task A1:

- [ ] **PF-1: Confirm clean working tree.**

  ```bash
  git status --short
  ```

  Expected: empty output (clean).

- [ ] **PF-2: Confirm baseline workspace gates.**

  ```bash
  cargo fmt --all -- --check
  cargo clippy --workspace --all-targets -- -D warnings
  cargo build --workspace
  cargo test --workspace
  ```

  Expected: all 4 commands exit 0. Test count baseline = 331 on macos-14 arm64 host (`~347` on linux x86_64 with FFI included).

  **Action if any fail:** stop, surface failure to user. Do not proceed.

- [ ] **PF-3: Confirm host-arch matches an existing profile.**

  ```bash
  uname -m
  ```

  Expected: `arm64` (Apple Silicon — bench will smoke `--profile arm64`) OR `x86_64` (Linux — bench will smoke `--profile x86_64`). Plan tasks B9 and C3 assume host-native smoke.

---

## Group A — Workspace Skeleton (1 commit)

**Goal:** New `bench/` workspace member that builds and lints clean as an empty bin.

**Files:** `bench/Cargo.toml`, `bench/src/main.rs`, `Cargo.toml` (workspace root).

### Task A1: Create bench crate skeleton

- [ ] **A1.1: Create directory.**

  ```bash
  mkdir -p bench/src
  ```

- [ ] **A1.2: Write `bench/Cargo.toml`.**

  Create `bench/Cargo.toml`:

  ```toml
  [package]
  name = "bench"
  version = "0.1.0"
  edition = "2021"
  description = "NeuralForge per-profile bench harness (OQ-BENCH closure, M11)"
  license.workspace = true

  [dependencies]
  compiler        = { path = "../compiler" }
  profile-api     = { path = "../profile-api" }
  profiles-arm64  = { path = "../profiles/arm64" }
  profiles-x86_64 = { path = "../profiles/x86_64" }
  libloading      = "0.8"
  rand            = "0.8"
  ```

- [ ] **A1.3: Write `bench/src/main.rs` stub.**

  Create `bench/src/main.rs`:

  ```rust
  // SPDX-License-Identifier: Apache-2.0

  //! NeuralForge bench harness (OQ-BENCH closure, M11).
  //!
  //! Compiles fixed NFL fixtures through the host-native profile,
  //! times warmup × 10 + measurement × 100 FFI calls, emits markdown
  //! to stdout (intended for `$GITHUB_STEP_SUMMARY` in CI).

  fn main() {
      eprintln!("bench: skeleton — wiring lands in M11 group B");
      std::process::exit(0);
  }
  ```

- [ ] **A1.4: Add `bench` to workspace members, first alphabetically.**

  Edit `Cargo.toml` (workspace root). Old:

  ```toml
  members = [
      "compiler",
      "nflc",
      "profile-api",
      "profiles/arm64",
      "profiles/x86_64",
  ]
  ```

  New:

  ```toml
  members = [
      "bench",
      "compiler",
      "nflc",
      "profile-api",
      "profiles/arm64",
      "profiles/x86_64",
  ]
  ```

- [ ] **A1.5: Build the workspace.**

  ```bash
  cargo build --workspace
  ```

  Expected: passes; `bench` builds.

- [ ] **A1.6: Lint the new crate.**

  ```bash
  cargo clippy --workspace --all-targets -- -D warnings
  ```

  Expected: passes.

- [ ] **A1.7: Format check.**

  ```bash
  cargo fmt --all -- --check
  ```

  Expected: passes.

- [ ] **A1.8: Smoke run.**

  ```bash
  cargo run -p bench --quiet
  ```

  Expected: prints "bench: skeleton — wiring lands in M11 group B" to stderr; exits 0.

- [ ] **A1.9: Commit Group A.**

  ```bash
  git add bench/Cargo.toml bench/src/main.rs Cargo.toml
  git commit -m "$(cat <<'EOF'
  feat(m11): bench crate skeleton

  New `bench/` workspace member (first alphabetically). Stub `main.rs`
  exits 0 with a placeholder message. Group B wires the harness.

  Closes part of OQ-BENCH (M11 group A of 4).

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Group B — Harness Implementation (1 commit)

**Goal:** A working `cargo run -p bench --release -- --profile <host-arch>` that emits a valid markdown report to stdout, with 2 unit tests gating the spec §11 #2 acceptance criteria.

**Files:** `bench/src/main.rs` (full rewrite, ~300 lines).

**TDD discipline:** for every pure function (stats, render, fill, CLI parse), write the test first, run it failing, then write the implementation. For the wiring (compile_and_load, main), use a manual smoke run in Task B9 instead of unit tests — these touch system `cc`, the filesystem, and `libloading`, which are integration concerns.

**Commit cadence:** workspace gates (`cargo fmt`, `cargo clippy`, `cargo test`) run after every task in this group; the actual `git commit` only at Task B11. This matches the M9/M10 group-commit cadence.

### Task B1: Stats helpers (`median`, `p95`)

**Files:** `bench/src/main.rs` (new section).

- [ ] **B1.1: Write the failing tests.**

  Append to `bench/src/main.rs`:

  ```rust
  #[cfg(test)]
  mod tests {
      use super::*;

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
  }
  ```

- [ ] **B1.2: Run tests, expect failure.**

  ```bash
  cargo test -p bench
  ```

  Expected: 4 failures with "cannot find function `median_ns` in this scope" / "cannot find function `p95_ns`".

- [ ] **B1.3: Implement `median_ns` and `p95_ns`.**

  Insert above the `#[cfg(test)] mod tests` block in `bench/src/main.rs`:

  ```rust
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
  ```

- [ ] **B1.4: Run tests, expect pass.**

  ```bash
  cargo test -p bench
  ```

  Expected: 4 passing.

- [ ] **B1.5: Workspace gates.**

  ```bash
  cargo fmt --all
  cargo clippy --workspace --all-targets -- -D warnings
  ```

  Expected: passes.

### Task B2: Markdown rendering (`render_report`)

**Files:** `bench/src/main.rs` (new section).

- [ ] **B2.1: Write the failing test.**

  Append to the `#[cfg(test)] mod tests` block in `bench/src/main.rs`:

  ```rust
      #[test]
      fn render_report_produces_stable_markdown_header() {
          let rows = vec![
              FixtureResult {
                  name: "classifier",
                  purpose: "matmul-mass (3-layer MLP)",
                  median_ns: 2_845_300,
                  p95_ns: 3_104_700,
              },
          ];
          let report = render_report("arm64", "macos-14, Apple M1 @ 3.2 GHz", 42, "abc1234", &rows);
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
          let rows = vec![
              FixtureResult {
                  name: "tiny",
                  purpose: "tiny",
                  median_ns: 48_700,   // 48.7 µs
                  p95_ns: 52_100,      // 52.1 µs
              },
          ];
          let report = render_report("arm64", "h", 42, "c", &rows);
          assert!(report.contains("48.7"), "expected 48.7 µs in: {report}");
          assert!(report.contains("52.1"), "expected 52.1 µs in: {report}");
      }

      #[test]
      fn render_report_formats_microseconds_as_integer_when_at_or_above_1000() {
          let rows = vec![
              FixtureResult {
                  name: "large",
                  purpose: "large",
                  median_ns: 2_845_300, // 2845.3 µs → 2845
                  p95_ns: 3_104_700,    // 3104.7 µs → 3104 (truncating)
              },
          ];
          let report = render_report("arm64", "h", 42, "c", &rows);
          assert!(report.contains(" 2845 "), "expected integer µs in: {report}");
          assert!(report.contains(" 3104 "), "expected integer µs in: {report}");
      }
  ```

- [ ] **B2.2: Run tests, expect failure.**

  ```bash
  cargo test -p bench
  ```

  Expected: 3 failures with "cannot find type `FixtureResult`" / "cannot find function `render_report`".

- [ ] **B2.3: Implement `FixtureResult` + `render_report`.**

  Insert above the existing `median_ns` definition:

  ```rust
  /// One fixture's measurement outcome. Owned form passed to `render_report`.
  struct FixtureResult {
      name: &'static str,
      purpose: &'static str,
      median_ns: u64,
      p95_ns: u64,
  }

  /// Format a nanosecond duration as a microsecond string. Below 1000 µs
  /// uses one decimal place (`48.7`); at or above 1000 µs uses integer
  /// (`2845`). Per spec §7.3 — small fixtures benefit from decimal
  /// precision, large ones don't.
  fn format_us(ns: u64) -> String {
      let us_x10 = (ns + 50) / 100; // round to one decimal of µs
      if us_x10 < 10_000 {
          // < 1000.0 µs
          format!("{}.{}", us_x10 / 10, us_x10 % 10)
      } else {
          // >= 1000 µs: integer
          format!("{}", ns / 1_000)
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
      s.push_str("| Fixture              | Median (µs) | p95 (µs) | Purpose                          |\n");
      s.push_str("|----------------------|-------------|----------|----------------------------------|\n");
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
  ```

- [ ] **B2.4: Run tests, expect pass.**

  ```bash
  cargo test -p bench
  ```

  Expected: 7 passing total (4 from B1 + 3 from B2).

- [ ] **B2.5: Workspace gates.**

  ```bash
  cargo fmt --all
  cargo clippy --workspace --all-targets -- -D warnings
  ```

### Task B3: Deterministic data fill (`fill_random`)

**Files:** `bench/src/main.rs` (new section).

- [ ] **B3.1: Write the failing test.**

  Append to the `#[cfg(test)] mod tests` block:

  ```rust
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
  ```

- [ ] **B3.2: Run tests, expect failure.**

  ```bash
  cargo test -p bench
  ```

  Expected: 2 failures with "cannot find function `fill_random`".

- [ ] **B3.3: Implement `fill_random`.**

  Insert above the existing `format_us` definition:

  ```rust
  use rand::distributions::Standard;
  use rand::rngs::StdRng;
  use rand::{Rng, SeedableRng};

  /// Fill `buffer` with deterministic-random `f32` values from
  /// `Standard` distribution (range `[0.0, 1.0)`). Spec §6.5.
  fn fill_random(buffer: &mut [f32], seed: u64) {
      let mut rng = StdRng::seed_from_u64(seed);
      for v in buffer.iter_mut() {
          *v = rng.sample(Standard);
      }
  }
  ```

- [ ] **B3.4: Run tests, expect pass.**

  ```bash
  cargo test -p bench
  ```

  Expected: 9 passing.

- [ ] **B3.5: Workspace gates.**

  ```bash
  cargo fmt --all
  cargo clippy --workspace --all-targets -- -D warnings
  ```

### Task B4: CLI flag parsing (`parse_args`)

**Files:** `bench/src/main.rs` (new section).

- [ ] **B4.1: Write the failing tests.**

  Append to the `#[cfg(test)] mod tests` block:

  ```rust
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
  ```

- [ ] **B4.2: Run tests, expect failure.**

  ```bash
  cargo test -p bench
  ```

  Expected: 4 failures with "cannot find function `parse_args`" / "cannot find type `BenchArgs`".

- [ ] **B4.3: Implement `BenchArgs` + `parse_args`.**

  Insert above the existing `fill_random` definition:

  ```rust
  /// Parsed CLI arguments. All fields owned to outlive `args`.
  struct BenchArgs {
      profile: String,
      format: String,
      seed: u64,
  }

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
                  seed = v.parse().map_err(|e| format!("bad --seed value '{v}': {e}"))?;
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
  ```

- [ ] **B4.4: Run tests, expect pass.**

  ```bash
  cargo test -p bench
  ```

  Expected: 13 passing.

- [ ] **B4.5: Workspace gates.**

  ```bash
  cargo fmt --all
  cargo clippy --workspace --all-targets -- -D warnings
  ```

### Task B5: Compile-and-load helper (`compile_to_dylib_for_host`)

**Files:** `bench/src/main.rs` (new section).

- [ ] **B5.1: Add wiring imports + helper. No unit test — this shells out to `cc` and touches the filesystem; it's covered by Task B9's manual smoke run.**

  Insert above the existing `BenchArgs` definition (top of the section, just below the `use rand::...` lines):

  ```rust
  use std::path::PathBuf;

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
          "x86_64" => (
              dir.join(format!("lib{name}.so")),
              vec!["-shared", "-fPIC"],
          ),
          // Already validated above.
          _ => unreachable!(),
      };
      cc_args.push("-o");

      let status = std::process::Command::new("cc")
          .args(&cc_args)
          .arg(&lib_path)
          .arg(&s_path)
          .status()
          .map_err(|e| format!("cc invocation failed to spawn: {e}"))?;
      if !status.success() {
          return Err(format!("cc failed to assemble {}: {status}", s_path.display()));
      }

      Ok(lib_path)
  }
  ```

- [ ] **B5.2: Workspace gates.**

  ```bash
  cargo fmt --all
  cargo clippy --workspace --all-targets -- -D warnings
  ```

  Expected: passes (the new function is unused; clippy will warn `dead_code`. Tolerate transiently — Task B7 wires it. If clippy `-D warnings` rejects, add `#[allow(dead_code)]` on this function temporarily; Task B7 removes the allow.)

  **Action if clippy rejects:** add `#[allow(dead_code)]` on `compile_to_dylib_for_host`. Plan to remove in Task B7.

### Task B6: Timing loop (`time_forward`)

**Files:** `bench/src/main.rs` (new section).

- [ ] **B6.1: Write helper. Unit-testing the timing loop is impractical (it consumes a real `extern "C" fn` pointer); covered by Task B9's manual smoke.**

  Insert below the `compile_to_dylib_for_host` definition:

  ```rust
  use std::time::Instant;

  /// Type alias for the bench's FFI calling convention. Matches
  /// `nfl_forward_<Model>(input, params, output)` exported by both
  /// profiles since M3.
  type ForwardFn = unsafe extern "C" fn(*const f32, *const f32, *mut f32);

  /// Run warmup × 10 + measurement × 100 calls of `forward`, return
  /// per-iteration measurement nanoseconds. Spec §6.4 warmup rationale
  /// (dyld/PLT lazy binding, cache warm-up).
  ///
  /// # Safety
  /// Caller must guarantee `input` and `params` point to valid buffers
  /// of the FFI-required lengths and that `output` is writable for its
  /// length.
  unsafe fn time_forward(
      forward: ForwardFn,
      input: *const f32,
      params: *const f32,
      output: *mut f32,
  ) -> Vec<u64> {
      const WARMUP: usize = 10;
      const MEASURE: usize = 100;

      for _ in 0..WARMUP {
          forward(input, params, output);
      }

      let mut samples = Vec::with_capacity(MEASURE);
      for _ in 0..MEASURE {
          let t0 = Instant::now();
          forward(input, params, output);
          let dt = t0.elapsed().as_nanos() as u64;
          samples.push(dt);
      }
      samples
  }
  ```

- [ ] **B6.2: Workspace gates.**

  ```bash
  cargo fmt --all
  cargo clippy --workspace --all-targets -- -D warnings
  ```

  Expected: passes (also unused — see B5.2 same `#[allow(dead_code)]` note if needed).

### Task B7: Per-fixture bench function (`bench_one_fixture`)

**Files:** `bench/src/main.rs` (new section).

- [ ] **B7.1: Implement `bench_one_fixture`.**

  Insert below the `time_forward` definition:

  ```rust
  use compiler::Uir;
  use profile_api::{Asm, Profile};

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
      let src = std::fs::read_to_string(fixture_path)
          .map_err(|e| format!("read {fixture_path}: {e}"))?;

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

      // 6. Allocate FFI buffers from FnSig (spec §5.5 step 4).
      let mut input: Vec<f32> = vec![0.0; sig.input_floats];
      let mut params: Vec<f32> = vec![0.0; sig.params_floats];
      let mut output: Vec<f32> = vec![0.0; sig.output_floats];

      // 7. Fill input then params from seed (spec §6.5).
      // Order matters for cross-host determinism: input first, then params,
      // each from a fresh PRNG re-seed (so params doesn't depend on input
      // length).
      fill_random(&mut input, seed);
      fill_random(&mut params, seed.wrapping_add(1));

      // 8. Compile to dylib.
      let lib_path = compile_to_dylib_for_host(&asm.source, fixture_name, profile_label)?;

      // 9. dlopen + dlsym. Note: `libloading` + Mach-O `dlsym` strip
      // the leading `_` automatically, so we pass `sig.name` verbatim
      // (verified at plan synthesis time against the M3-M10 integration
      // test pattern: `lib.get(b"nfl_forward_M4Demo")` works on macos-14).
      let lib = unsafe { libloading::Library::new(&lib_path) }
          .map_err(|e| format!("dlopen {}: {e}", lib_path.display()))?;
      let forward: libloading::Symbol<ForwardFn> = unsafe {
          lib.get(sig.name.as_bytes())
              .map_err(|e| format!("dlsym {}: {e}", sig.name))?
      };

      // 10. Time. Note: `params.as_ptr()` is non-null aligned for
      // `params_floats == 0` (Vec::as_ptr returns NonNull::dangling for
      // empty vecs, which is f32-aligned; the codegen never dereferences
      // a zero-param buffer — established by M10 self_attention contract).
      let mut samples = unsafe {
          time_forward(*forward, input.as_ptr(), params.as_ptr(), output.as_mut_ptr())
      };

      // 11. Drop FFI handle before returning so the dylib is unloaded.
      drop(forward);
      drop(lib);

      Ok(FixtureResult {
          name: fixture_name,
          purpose,
          median_ns: median_ns(&mut samples),
          p95_ns: p95_ns(&mut samples),
      })
  }
  ```

- [ ] **B7.2: Remove any `#[allow(dead_code)]` from B5/B6 (now wired).**

  If B5.2 or B6.2 added `#[allow(dead_code)]`, remove now.

- [ ] **B7.3: Workspace gates.**

  ```bash
  cargo fmt --all
  cargo clippy --workspace --all-targets -- -D warnings
  ```

  Expected: passes.

### Task B8: Main wiring + fixtures registry

**Files:** `bench/src/main.rs` (replace stub `main`).

- [ ] **B8.1: Replace the stub `main` with the full driver.**

  Replace the existing stub `fn main()` (the one inserted in Task A1.3):

  ```rust
  fn main() {
      eprintln!("bench: skeleton — wiring lands in M11 group B");
      std::process::exit(0);
  }
  ```

  With:

  ```rust
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
          eprintln!("bench: running {name}…");
          match bench_one_fixture(path, name, purpose, profile.as_ref(), &profile_label, args.seed) {
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
      format!(
          "{}, {}",
          std::env::consts::OS,
          std::env::consts::ARCH
      )
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
  ```

- [ ] **B8.2: Workspace gates.**

  ```bash
  cargo fmt --all
  cargo clippy --workspace --all-targets -- -D warnings
  cargo test -p bench
  ```

  Expected: passes; 13 unit tests still passing (no regressions).

### Task B9: Local smoke run

- [ ] **B9.1: Build release.**

  ```bash
  cargo build -p bench --release
  ```

  Expected: passes.

- [ ] **B9.2: Run on host's native profile.**

  Determine host arch:

  ```bash
  uname -m
  ```

  - If `arm64`: run `cargo run -p bench --release -- --profile arm64 --format markdown`
  - If `x86_64`: run `cargo run -p bench --release -- --profile x86_64 --format markdown`

  Expected: prints a markdown report to stdout matching spec §7.1 layout. Three fixture rows. No panics, no `cc` failures, no dlopen failures.

  **Sample expected output structure:**

  ```
  ## NeuralForge bench — arm64 profile

  **Host:** macos, aarch64
  **Build:** release, single-thread
  **Methodology:** 10 warmup + 100 measurement, median + p95
  **Seed:** 42 (StdRng::seed_from_u64)
  **Compiled at:** <git-sha>

  | Fixture              | Median (µs) | p95 (µs) | Purpose                          |
  |----------------------|-------------|----------|----------------------------------|
  | classifier           |        2845 |     3104 | matmul-mass (3-layer MLP)        |
  | large_classifier_k   |        48.7 |     52.1 | large-K inner-loop accumulator   |
  | self_attention       |       108.4 |    119.5 | expf/softmax dispatch overhead   |
  ```

  **Action if any fixture fails:** debug per fixture. Likely culprits:
  - Wrong fixture path → check `cargo run` is invoked from worktree root.
  - Symbol-name lookup fails on macOS → `libloading` should strip Mach-O `_` automatically; if it doesn't, manually prepend `_` to `sig.name` in B7.1 step 9 and re-test. Document the deviation as a plan-time correction.
  - `cc` fails on Linux x86_64 → ensure `cc` and `libc6-dev` are present on the test host (default on ubuntu-latest).

- [ ] **B9.3: Sanity-check numbers.**

  Median values should land in the expected ranges per spec §8.1:
  - `classifier`: ~2-5 ms (range 2000-5000 µs)
  - `large_classifier_k`: ~30-100 µs
  - `self_attention`: ~50-200 µs

  If any fixture's median is < 1 µs (suspect `time_forward` is broken) or > 100 ms (suspect FFI buffer corruption causing infinite loop), stop and debug.

- [ ] **B9.4: Verify wrong-host error path.**

  On macOS arm64 host, attempt the unsupported profile:

  ```bash
  cargo run -p bench --release -- --profile x86_64 --format markdown
  ```

  Expected: exits 1, stderr includes "host (macos/aarch64) cannot natively run --profile x86_64; cross-execution unsupported".

  (On Linux x86_64 host, equivalently use `--profile arm64`.)

### Task B10: Final workspace gates for Group B

- [ ] **B10.1: Full workspace gates.**

  ```bash
  cargo fmt --all -- --check
  cargo clippy --workspace --all-targets -- -D warnings
  cargo build --workspace
  cargo test --workspace
  ```

  Expected: all pass. Test count = 331 (baseline) + ≥ 9 new bench unit tests = ≥ 340 on macos-14 (≥ 356 on linux). Spec §11 #2 wants ≥ 333; we exceed.

### Task B11: Commit Group B

- [ ] **B11.1: Stage and commit.**

  ```bash
  git add bench/src/main.rs
  git commit -m "$(cat <<'EOF'
  feat(m11): bench harness — compile, dlopen, time, render

  Single-file `bench/src/main.rs` (~300 lines) implementing the OQ-BENCH
  harness:

  - Pure-function helpers (median, p95, format_us, fill_random, parse_args,
    render_report) with 13 unit tests.
  - `compile_to_dylib_for_host`: shells to system `cc`, host-arch-aware
    (macOS arm64 → .dylib; Linux x86_64 → .so); refuses cross-execution.
  - `time_forward`: warmup × 10 + measurement × 100, `Instant::now()`-based
    nanosecond timing.
  - `bench_one_fixture`: end-to-end (parse → build → default_pipeline →
    profile.lower → cc → dlopen → time).
  - `main`: hardcoded fixture registry (classifier, large_classifier_k,
    self_attention) + profile dispatch (`Arm64Profile` / `X86_64Profile`)
    + markdown render to stdout.

  Symbol lookup: `lib.get(sig.name.as_bytes())` directly — `libloading`
  + Mach-O `dlsym` strip the leading `_` automatically (verified against
  the M3-M10 integration test pattern). Spec §5.5 step 7 corrected at
  plan time per spec §13 Q4.

  Hand-smoke on macos-14 arm64 confirms expected µs ranges:
  classifier ~2-5 ms, large_classifier_k ~30-100 µs,
  self_attention ~50-200 µs.

  Closes part of OQ-BENCH (M11 group B of 4).

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Group C — CI Workflow (1 commit)

**Goal:** `.github/workflows/bench.yml` runs successfully on the feature branch via `workflow_dispatch`, and each leg writes to `$GITHUB_STEP_SUMMARY`.

**Files:** `.github/workflows/bench.yml`.

### Task C1: Write the workflow

- [ ] **C1.1: Write `.github/workflows/bench.yml`.**

  Create `.github/workflows/bench.yml`:

  ```yaml
  name: Bench

  on:
    workflow_dispatch:
    push:
      branches: [main]
      paths-ignore:
        - 'bench/results/**'
        - 'docs/**'
        - '**.md'

  concurrency:
    group: bench
    cancel-in-progress: false

  env:
    CARGO_TERM_COLOR: always
    RUST_BACKTRACE: 1

  jobs:
    bench:
      strategy:
        fail-fast: false
        matrix:
          include:
            - { os: macos-14,      profile: arm64  }
            - { os: ubuntu-latest, profile: x86_64 }
      runs-on: ${{ matrix.os }}
      name: Bench ${{ matrix.profile }} (${{ matrix.os }})
      steps:
        - name: Checkout
          uses: actions/checkout@v4

        - name: Install Rust stable
          uses: dtolnay/rust-toolchain@stable

        - name: Cache cargo registry + target
          uses: Swatinem/rust-cache@v2

        - name: Build bench (release)
          run: cargo build -p bench --release

        - name: Run bench
          run: |
            cargo run -p bench --release -- \
              --profile ${{ matrix.profile }} \
              --format github-summary \
              --seed 42 \
              >> "$GITHUB_STEP_SUMMARY"
  ```

- [ ] **C1.2: YAML lint (best-effort).**

  ```bash
  python3 -c "import yaml; yaml.safe_load(open('.github/workflows/bench.yml'))"
  ```

  Expected: no error.

  If `actionlint` is on PATH, also:

  ```bash
  actionlint .github/workflows/bench.yml
  ```

  Expected: no error. (If `actionlint` is not on PATH, skip — hand-review per Task C2.)

### Task C2: Hand-review workflow

- [ ] **C2.1: Read the workflow against spec §9.**

  Re-read `.github/workflows/bench.yml` and verify:
  - `on:` includes `workflow_dispatch` AND `push: branches: [main]` with `paths-ignore` covering `bench/results/**`, `docs/**`, `**.md`.
  - `concurrency: { group: bench, cancel-in-progress: false }`.
  - Matrix has exactly two legs: `macos-14`/`arm64` and `ubuntu-latest`/`x86_64`.
  - `fail-fast: false` (spec §9.3).
  - The `Run bench` step pipes `>> "$GITHUB_STEP_SUMMARY"` (note quotes for path safety).
  - **No `actions/upload-artifact` / `actions/download-artifact` anywhere** (spec §11 #8).
  - **No threshold check / regression gate** (spec §11 #9). The job exits 0 unless the bench binary panics.

  If any fail, fix inline.

### Task C3: Smoke via `gh workflow run`

- [ ] **C3.1: Push the current branch.**

  Pre-condition: `bench.yml` must be committed to the feature branch first (per spec §13 Q5).

  ```bash
  git add .github/workflows/bench.yml
  # Commit happens at C4.1; for now, stage only.
  ```

  **Action:** make a temporary commit ONLY for the smoke step, then amend it into the proper commit at C4.1:

  ```bash
  git commit -m "chore(m11): bench.yml (smoke commit, will amend)"
  git push origin claude/sweet-perlman-60d78b
  ```

- [ ] **C3.2: Trigger the workflow on the feature branch.**

  ```bash
  gh workflow run bench.yml --ref claude/sweet-perlman-60d78b
  ```

  Expected: `gh` reports run created. (If error "workflow not found" → branch push didn't reach origin or workflow file isn't on the ref. Re-push and retry.)

- [ ] **C3.3: Watch the run.**

  ```bash
  gh run list --workflow=bench.yml --limit 1
  gh run watch <run-id>
  ```

  Expected: both legs complete. Each leg's Job Summary contains the §7.1 markdown table for its profile.

  **Action if a leg fails:**
  - `arm64` leg fails on macos-14: most likely `cc` arch flag or symbol lookup. Re-check Task B5.1 / B7.1.
  - `x86_64` leg fails on ubuntu-latest: most likely `-fPIC` missing or PLT lazy-binding not warmed. Re-check Task B5.1 — `-shared -fPIC` should be enough.

- [ ] **C3.4: Inspect both Job Summaries.**

  ```bash
  gh run view <run-id> --log
  ```

  Or via web UI: navigate to the run, expand each leg, click "Summary".

  Expected: each Job Summary shows the §7.1 layout. Three fixtures, four columns (Fixture, Median, p95, Purpose). No raw stdout leaking past the table (the redirect should consume the entire bench stdout into the summary).

### Task C4: Finalise the workflow commit

- [ ] **C4.1: Amend the temporary smoke commit into the proper Group C commit.**

  ```bash
  git commit --amend -m "$(cat <<'EOF'
  feat(m11): bench.yml workflow — Job Summary per matrix leg

  New `.github/workflows/bench.yml`. Two-leg matrix (`macos-14` for arm64,
  `ubuntu-latest` for x86_64). Each leg runs `cargo run -p bench
  --release` and pipes stdout into `\$GITHUB_STEP_SUMMARY`.

  No artifact upload/download — matrix legs are independent (preserves
  M10 §11.2 rule). No threshold check — informational only.

  Trigger: `workflow_dispatch` + `push: branches: [main]` with
  `paths-ignore` for `bench/results/**`, `docs/**`, `**.md` (prevents
  self-trigger on combined-report commits and on doc-only changes).

  `concurrency: bench, cancel-in-progress: false` so overlapping pushes
  don't race the same runner mid-bench.

  Smoke-tested via \`gh workflow run bench.yml --ref <branch>\` on the
  feature branch before opening the PR. Both legs produced the expected
  §7.1 Job Summary layout.

  Closes part of OQ-BENCH (M11 group C of 4).

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

- [ ] **C4.2: Force-push the amended commit (only allowed because we amended a commit we ourselves authored within minutes, on a feature branch only we are using).**

  ```bash
  git push --force-with-lease origin claude/sweet-perlman-60d78b
  ```

  Expected: push succeeds.

  **Action if `--force-with-lease` is rejected:** investigate. Probably a concurrent push from the user; ask the user before recovering.

---

## Group D — Documentation Closure (1 commit)

**Goal:** `PROJECT_SPEC.md`, `CLAUDE.md`, `DEVLOG.md` reflect M11 closure.

**Files:** `PROJECT_SPEC.md`, `CLAUDE.md`, `DEVLOG.md`.

### Task D1: Update `PROJECT_SPEC.md`

- [ ] **D1.1: Add M11 row to the "First Milestones" table.**

  Edit `PROJECT_SPEC.md`. Find the row for M10 (the longest one in the table). Insert a new row directly after it:

  ```markdown
  | 11 | OQ-BENCH harness — close the M9-merge trigger (complete) | New `bench/` workspace crate (`cargo run -p bench --release -- --profile {arm64|x86_64} --format {markdown|github-summary} [--seed N]`) compiling 3 fixtures (`classifier`, `large_classifier_k`, `self_attention`) through host-native profile, timing 10 warmup + 100 measurement FFI calls, reporting median + p95 µs to stdout / Job Summary. New `.github/workflows/bench.yml` with 2-leg matrix (`macos-14` arm64 + `ubuntu-latest` x86_64); each leg writes to `$GITHUB_STEP_SUMMARY` (no artifact sharing, no aggregator). Cross-profile combined report composed manually post-CI to `bench/results/<date>.md`. Test count: 331 → ≥ 340 (host-dependent). |
  ```

- [ ] **D1.2: Mark OQ-BENCH closed under "Trigger-driven cleanup".**

  Find the OQ-BENCH bullet. Old:

  ```markdown
  - **OQ-BENCH** (opened by M9 spec, fires on M9 merge) — Build a benchmark harness that compiles a single NFL source through both `arm64` and `x86_64` profiles, runs both binaries with the same input/params, and reports timing side-by-side. Goal: quantify the cost of "scalar-only" vs the eventual SIMD profile, and lay groundwork for performance claims. *Trigger: M9 merged. Scope: stretch enough to handle multiple fixtures; output a markdown report. No regression-gate yet — informational only.*
  ```

  New:

  ```markdown
  - **OQ-BENCH** — **Closed in M11 (commit `<TBD>`).** Bench harness shipped as the `bench/` workspace crate; CI workflow `.github/workflows/bench.yml` writes per-profile Job Summaries on `macos-14` (arm64) and `ubuntu-latest` (x86_64). Cross-profile comparison is composed manually into `bench/results/<date>.md` after each run (no aggregator job — preserves M10 §11.2 rule). Three fixtures (`classifier` / `large_classifier_k` / `self_attention`) chosen for orthogonal signals (matmul-mass / large-K inner-loop accumulator / expf-dominated dispatch overhead). Methodology: 10 warmup + 100 measurement, median + p95.
  ```

  **Note:** the `<TBD>` is the closing PR's squash-merge commit hash. Filled in by the implementer post-merge or by the user before the PR is squashed (the merge commit hash is known to the user before squash via GitHub UI).

- [ ] **D1.3: Update "Current Status".**

  Edit `PROJECT_SPEC.md` Current Status section. Look for "Milestone 10 complete" — bump to "Milestone 11 complete" with the new test count. (CLAUDE.md "Current Status" gets the same change in Task D2.)

  Specifically: find any sentence like "M10 closed the first leg of Axis 2…". Append: "M11 closed OQ-BENCH (informational scalar-baseline bench harness) as a separate trigger-driven cleanup milestone." Test count line updates to "≥ 340 on macOS arm64 (≥ 356 on Linux x86_64 CI with x86_64 FFI tests included)" — exact numbers come from `cargo test --workspace` output post-Group B.

### Task D2: Update `CLAUDE.md`

- [ ] **D2.1: Add `bench/` to the Repository Structure tree.**

  Edit `CLAUDE.md`. Find the tree fragment that lists workspace members (look for `compiler/`, `nflc/`, `profile-api/`). Add `bench/` block in the right alphabetical position (first):

  ```text
  ├── bench/                  ← `bench` crate (bin only) — OQ-BENCH harness
  │   ├── Cargo.toml
  │   ├── src/main.rs         ← single-file harness (~300 lines)
  │   └── results/            ← committed cross-profile reports
  │       └── <YYYY-MM-DD>.md ← lands as a post-merge follow-up commit
  ```

  Also update workspace members list in the same file: the `Cargo.toml` workspace manifest line "members = [..." should now show `["bench", "compiler", "nflc", "profile-api", "profiles/arm64", "profiles/x86_64"]`.

- [ ] **D2.2: Update "Current Status" in CLAUDE.md.**

  Find the line "Milestone 10 complete. 331 tests passing on macOS arm64 (~347 on Linux x86_64 CI with x86_64 FFI tests included)." Change to:

  ```markdown
  **Milestone 11 complete. ≥ 340 tests passing on macOS arm64 (≥ 356 on Linux x86_64 CI).**
  ```

  Replace exact counts with the actual `cargo test --workspace` count from after Group B.

  Update the paragraph below explaining what M11 added (analogous to existing M10 paragraph): one sentence summary of OQ-BENCH closure.

  Update "Strategic direction" paragraph: keep most of it but note OQ-BENCH closure: "OQ-BENCH closed in M11 (commit `<TBD>`)" added near the existing "OQ-NEW closed in M9" line.

### Task D3: Update `DEVLOG.md`

- [ ] **D3.1: Insert the M11 closure entry at the top, just below the format header.**

  Open `DEVLOG.md`. Find the line:

  ```markdown
  ---

  ## 2026-05-09 — Milestone 10 closed: NFL v0.2 self-attention + 4D codegen
  ```

  Insert ABOVE it (after the format `---`):

  ```markdown
  ## 2026-05-09 — Milestone 11 closed: OQ-BENCH harness — closes M9-merge trigger

  ### What was done
  - **`bench/` workspace crate** (new). Single-file `bench/src/main.rs` (~300
    lines) implementing the harness: hand-rolled CLI parser
    (`--profile {arm64|x86_64}`, `--format {markdown|github-summary}`,
    `--seed N` default 42), pure-function helpers (`median_ns`, `p95_ns`,
    `fill_random`, `format_us`, `render_report`) covered by 13 unit
    tests, and the wiring (`compile_to_dylib_for_host`, `time_forward`,
    `bench_one_fixture`, `main`).
  - **Three fixtures, three orthogonal signals** (per spec §8.1): hardcoded
    in `FIXTURES`. `classifier` (matmul-mass, ~3 ms), `large_classifier_k`
    (large-K inner-loop accumulator, ~50 µs), `self_attention`
    (expf/softmax dispatch overhead, ~100 µs). No new `.nfl` files.
  - **Buffer plumbing from `FnSig` only.** `sig.input_floats` /
    `sig.params_floats` / `sig.output_floats` are the source of truth;
    bench does not duplicate `walk_model`'s param-layout logic.
    `params_floats == 0` (e.g. `self_attention`) handled by the standard
    `vec![0f32; 0].as_ptr()` non-null-aligned dangling-pointer pattern.
  - **CI workflow** `.github/workflows/bench.yml`. Two-leg matrix
    (`macos-14` arm64 + `ubuntu-latest` x86_64). Each leg pipes
    `cargo run -p bench --release -- --profile <leg> --format
    github-summary --seed 42` into `\$GITHUB_STEP_SUMMARY`. No artifact
    upload/download anywhere. `concurrency: bench, cancel-in-progress:
    false`. Triggered on `workflow_dispatch` + `push: branches: [main]`
    with `paths-ignore: ['bench/results/**', 'docs/**', '**.md']`
    (prevents self-triggering on the combined-report commits and on
    doc-only changes).
  - **First combined report** `bench/results/<merge-date>.md` lands as a
    post-merge follow-up commit (sequenced per spec §11 #5 — inaugural
    CI run cannot precede merge).
  - **Documentation**: PROJECT_SPEC.md (M11 row in milestones table +
    OQ-BENCH closed under Trigger-driven cleanup + Current Status
    bumped), CLAUDE.md (Repository Structure tree gains `bench/`,
    Current Status to M11), this entry.

  ### Decisions made

  **Median is strict (`(samples[49] + samples[50]) / 2`), not upper-median.**
  Spec review caught this in the §6.2 wording. For even N (=100) the
  upper-median index 50 introduces a < 1-sample upward bias; the strict
  formula matches the "median" label exactly. One extra add + division —
  free.

  **Symbol lookup uses `sig.name` directly.** Spec §5.5 step 7 said
  "prepend `_` for Mach-O" and spec §13 Q4 deferred verification to plan
  synthesis. Plan synthesis confirmed against the M3-M10 integration test
  pattern (`lib.get(b"nfl_forward_M4Demo")` works on macos-14): `libloading`
  + `dlsym` strip the leading `_` automatically. Bench passes
  `sig.name.as_bytes()` verbatim; `profile.sym_prefix()` is unused at the
  bench layer.

  **`compile_to_dylib_for_host` not extracted to a shared crate.** The
  bench's host-arch-aware variant of `compile_to_dylib` lives inline in
  `bench/src/main.rs`. The arm64 integration tests' helper is hard-coded
  to `-arch arm64`; lifting both to a shared crate to share ~30 lines of
  `Command::new("cc")` wrapping is not worth a new crate-on-crate
  dependency at M11 scale. If a third caller appears, that's the
  trigger for extraction (spec §5.4).

  **No artifact sharing, no aggregator job.** Reaffirmed M10 §11.2
  rule. The "side-by-side" deliverable is composed manually offline
  into `bench/results/<date>.md` from the two Job Summary outputs of
  the inaugural CI run; the matrix legs themselves stay strictly
  independent.

  **`mean ± stddev` not reported.** Spec §6.2 picked median + p95
  because inference latency on shared CI runners is right-skewed (host
  co-tenancy → occasional long outliers). `mean ± stddev` overstates
  noise and hides the typical case.

  **Default pipeline ON.** Per spec §8.3 the bench runs the default
  pipeline (`EliminateDropout + FuseLinearRelu + FuseLinearSoftmax`).
  No `--no-passes` ablation column — out of scope for M11.

  ### Problems encountered
  - <Filled in during execution. Likely candidates: cc tool-chain on
    Linux/macos-14 missing flag; symbol-lookup edge case if libloading
    behaviour differs by version; tempdir cleanup if PIDs reuse.>

  ### Next step
  Push branch + open PR titled `feat(m11): OQ-BENCH harness — close
  M9-merge trigger`. Once merged, run the workflow once on `main` (it
  fires automatically on the merge push), copy both Job Summaries
  into `bench/results/<merge-date>.md`, push the follow-up commit
  directly to `main`. M11 is then fully closed.

  After M11, the next milestone selection runs over the post-M10
  Strategic Roadmap (Axis 2 follow-ups: A1 multi-input grammar with
  ABI-scope disclosure, A2 transformer block, A3 viewer annotations;
  Axis 3 bare-metal `expf`; Axis 1 follow-ups: SIMD / macOS x86_64).
  M11's first numbers feed into that decision: if matmul dominates
  classifier as expected, B1 (SIMD) becomes the highest-leverage
  next milestone.

  ---
  ```

### Task D4: Final workspace gates

- [ ] **D4.1: Run all gates.**

  ```bash
  cargo fmt --all -- --check
  cargo clippy --workspace --all-targets -- -D warnings
  cargo build --workspace
  cargo test --workspace
  ```

  Expected: all pass.

  Capture exact test count from `cargo test --workspace`. Update PROJECT_SPEC and CLAUDE.md exact numbers if Task D1.3 / D2.2 used placeholders.

### Task D5: Commit Group D

- [ ] **D5.1: Stage and commit.**

  ```bash
  git add PROJECT_SPEC.md CLAUDE.md DEVLOG.md
  git commit -m "$(cat <<'EOF'
  docs(m11): close M11 — PROJECT_SPEC, CLAUDE, DEVLOG

  - PROJECT_SPEC.md: M11 row in First Milestones; OQ-BENCH bullet under
    Trigger-driven cleanup transitions to "Closed in M11 (commit
    \`<TBD>\`)" matching the OQ-NEW closure pattern; Current Status
    bumped.
  - CLAUDE.md: Repository Structure tree gains `bench/` member;
    Current Status to M11; workspace members list updated.
  - DEVLOG.md: standard milestone-closure entry.

  Closes M11 (commit graph: groups A/B/C/D — this is D, plus one
  post-merge follow-up commit landing `bench/results/<merge-date>.md`
  per spec §11 #5).

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Branch Finalisation

### Task F1: Push and open PR

- [ ] **F1.1: Push the branch.**

  ```bash
  git push origin claude/sweet-perlman-60d78b
  ```

  Expected: push succeeds; the four group commits (A, B, C, D) are on origin.

- [ ] **F1.2: Open the PR.**

  ```bash
  gh pr create --title "feat(m11): OQ-BENCH harness — close M9-merge trigger" --body "$(cat <<'EOF'
  ## Summary

  Closes the OQ-BENCH trigger that fired on M9 merge. New `bench/` workspace
  crate, `.github/workflows/bench.yml`, and milestone closure in PROJECT_SPEC,
  CLAUDE.md, and DEVLOG.

  - **Bench**: `cargo run -p bench --release -- --profile {arm64|x86_64} --format {markdown|github-summary} [--seed N]`. Three fixtures (classifier, large_classifier_k, self_attention) chosen for orthogonal signals. 10 warmup + 100 measurement, median + p95.
  - **CI**: 2-leg matrix (`macos-14` arm64 + `ubuntu-latest` x86_64). Each leg writes to `\$GITHUB_STEP_SUMMARY`. **No artifact sharing, no aggregator job** (preserves M10 §11.2 rule).
  - **Reports**: combined cross-profile markdown lands at `bench/results/<date>.md` as a **post-merge follow-up commit** (inaugural CI run cannot precede merge — spec §11 #5).

  Spec: \`docs/superpowers/specs/2026-05-09-m11-oq-bench-harness-design.md\`.
  Plan: \`docs/superpowers/plans/2026-05-09-m11-oq-bench-harness.md\`.

  ## Test plan

  - [x] `cargo build --workspace`
  - [x] `cargo clippy --workspace --all-targets -- -D warnings`
  - [x] `cargo fmt --all -- --check`
  - [x] `cargo test --workspace` (test count: 331 → ≥ 340 macos-14 / ≥ 356 ubuntu)
  - [x] Local smoke: `cargo run -p bench --release -- --profile <host-arch> --format markdown` produces a 3-row table with µs values in the spec §8.1 expected ranges
  - [x] Workflow smoke: `gh workflow run bench.yml --ref claude/sweet-perlman-60d78b` produces 2 successful jobs, each with a §7.1-shaped Job Summary
  - [ ] **Post-merge:** workflow runs automatically on the merge push to \`main\`; both Job Summaries are copied into \`bench/results/<merge-date>.md\` as a follow-up commit

  🤖 Generated with [Claude Code](https://claude.com/claude-code)
  EOF
  )"
  ```

  Expected: PR opened; URL printed. Return URL to user.

- [ ] **F1.3: After the PR is reviewed and merged**, the branch ends. The post-merge follow-up commit (Task G1) is performed on `main` directly.

---

## Post-Merge — `bench/results/<date>.md` (1 commit on `main`)

**Goal:** First combined cross-profile report lands.

**Trigger:** PR #X is squash-merged into `main`. The push to `main` fires `bench.yml` automatically. The merging implementer (or user) performs Task G1 once that run completes.

### Task G1: Compose the first combined report

- [ ] **G1.1: Wait for the inaugural `main` run to complete.**

  ```bash
  gh run list --workflow=bench.yml --branch=main --limit=1
  gh run watch <run-id>
  ```

  Expected: both legs succeed.

- [ ] **G1.2: Copy both Job Summaries.**

  ```bash
  gh run view <run-id> --log
  ```

  Or via web UI: navigate to the run, expand each leg, click "Summary", copy markdown.

- [ ] **G1.3: Compose `bench/results/<merge-date>.md`.**

  Replace `<merge-date>` with the actual date of the merge run (YYYY-MM-DD format). Create the file with the spec §7.2 layout, filling values from both Job Summaries:

  ```markdown
  # NeuralForge bench — <merge-date>

  Source: GitHub Actions run #<NNN> ([arm64 leg](<url>), [x86_64 leg](<url>)).
  Commit: <merge-commit-sha>

  | Fixture              | arm64 median | arm64 p95 | x86_64 median | x86_64 p95 | arm64 / x86_64 |
  |----------------------|--------------|-----------|---------------|------------|----------------|
  | classifier           | <a-med>      | <a-p95>   | <x-med>       | <x-p95>    | <a-med/x-med>  |
  | large_classifier_k   | <a-med>      | <a-p95>   | <x-med>       | <x-p95>    | <a-med/x-med>  |
  | self_attention       | <a-med>      | <a-p95>   | <x-med>       | <x-p95>    | <a-med/x-med>  |

  **Hosts:**
  - arm64 = macos-14 (Apple M1 @ 3.2 GHz)
  - x86_64 = ubuntu-latest (Intel Xeon, runner SKU varies)

  Ratio is computed on **median** (not p95). Ratio < 1 means arm64 is faster.

  *First M11 combined report. Numbers are noisy on shared GitHub-hosted
  runners; treat absolute values as informational only. Cross-run
  comparisons are unreliable; comparisons within this single run are
  reliable.*
  ```

- [ ] **G1.4: Commit and push directly to `main`.**

  Pre-condition: branch protection on `main` permits direct push, OR a tiny PR is opened. If branch protection blocks direct push, open a follow-up PR titled `chore(m11): inaugural bench/results/<date>.md` with this single file change.

  ```bash
  git checkout main
  git pull
  git add bench/results/<merge-date>.md
  git commit -m "$(cat <<'EOF'
  chore(m11): inaugural bench/results/<merge-date>.md

  First combined cross-profile bench report, composed manually from the
  two Job Summary outputs of the inaugural `bench.yml` run on `main`.

  Per spec §11 #5: this commit cannot live in the M11 closing PR
  because the inaugural CI run requires the workflow to be on `main`,
  which only happens at merge.

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  git push origin main
  ```

  (If branch protection requires a PR: open it with `gh pr create`, merge it normally.)

- [ ] **G1.5: Update PROJECT_SPEC.md OQ-BENCH closure with the actual hash.**

  Edit PROJECT_SPEC.md OQ-BENCH bullet, replace `<TBD>` with the squash-merge commit hash from the M11 PR (visible in `git log main` or in the GitHub PR). Commit:

  ```bash
  git add PROJECT_SPEC.md
  git commit -m "$(cat <<'EOF'
  docs(m11): backfill OQ-BENCH closure commit hash

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  git push origin main
  ```

  (Or fold into G1.4 if doing a single follow-up PR — both files in one commit.)

- [ ] **G1.6: M11 fully closed.**

  Final state on `main`:
  - 4 commits from the closing PR (A, B, C, D as squash-merged).
  - 1-2 follow-up commits (G1.4 + optionally G1.5 separately, or G1.4 fold).
  - `bench/results/<merge-date>.md` exists.
  - PROJECT_SPEC.md OQ-BENCH bullet shows the actual closing-PR hash.

---

## Self-Review Checklist (run before considering plan done)

This is run by the plan author once the plan is written, before handing to the implementer.

**1. Spec coverage:**
- ✓ §2 goal 1 (workspace member, CLI flags) → Task A1.2 + B4.3 + B8.1.
- ✓ §2 goal 2 (single-file harness) → Group B (Tasks B1 → B11).
- ✓ §2 goal 3 (3 fixtures wired by absolute path + label) → Task B8.1 `FIXTURES` constant.
- ✓ §2 goal 4 (`bench.yml` with two-leg matrix and Job Summary writes) → Group C.
- ✓ §2 goal 5 (`bench/results/<date>.md` follow-up) → Task G1.
- ✓ §2 goal 6 (PROJECT_SPEC, CLAUDE.md, DEVLOG updates) → Group D.
- ✓ §6.2 strict-median formula → Task B1.3 inline code.
- ✓ §7.1/§7.2 markdown layout → Task B2.3 + Task G1.3.
- ✓ §9 workflow shape → Task C1.1 inline YAML.
- ✓ §11 acceptance criteria → distributed across Group B (test count), Task C2 (workflow lint), Task D4 (final gates).
- ✓ §11 #8 (no artifact sharing) → enforced by Task C1.1 (no upload/download steps) + Task C2 hand-review.
- ✓ §11 #9 (no regression gate) → enforced by Task C1.1 (no threshold step).
- ✓ §13 Q4 (libloading underscore handling) → resolved in Task B7.1 with rationale.
- ✓ §13 Q5 (gh-cli precondition) → Task C3.1 explicitly orders push-then-trigger.

**2. Placeholder scan:** searched the plan for "TBD", "TODO", "fill in", "implement later", "appropriate". Found:
- "<TBD>" in commit-hash references — these are legitimate variable substitutions, filled at merge time. Acceptable.
- "<merge-date>" in Task G1 — same: filled at merge time. Acceptable.
- "<run-id>", "<NNN>", "<a-med>" placeholders in Task G1 — filled by reading the actual CI run output. Acceptable (the file Task G1 produces is itself the manual artefact; placeholders are spelled out as such).
- One `<Filled in during execution. Likely candidates: ...>` in DEVLOG entry Task D3.1 (Problems encountered section). Plan-time we can't predict actual problems; M9/M10 DEVLOG follow the same pattern.
- No "appropriate error handling" / "edge cases" / "similar to Task N" violations.

**3. Type consistency:**
- `FixtureResult` defined in B2.3, consumed in B7.1 step 11 — fields `name`, `purpose`, `median_ns`, `p95_ns` consistent.
- `BenchArgs` defined in B4.3, consumed in B8.1 — fields `profile`, `format`, `seed` consistent.
- `ForwardFn` type alias defined in B6.1, consumed in B7.1 step 9 — same signature.
- `compile_to_dylib_for_host` defined B5.1 with 3 params, called B7.1 step 8 with the same 3.
- `time_forward` defined B6.1, called B7.1 step 10 with same arg shape.
- `bench_one_fixture` defined B7.1 with 6 params, called B8.1 with same 6.
- `parse_args` returns `Result<BenchArgs, String>` (B4.3); B8.1 uses the `Ok`/`Err` pattern matching it.

All consistent.

---

## Execution Notes for Implementer

- Work strictly in the worktree (`/Users/arseniivoloshyn/Проекты/experimental_projects/NeuralForge/.claude/worktrees/sweet-perlman-60d78b`). Do not jump to the main repository directory.
- Run workspace gates after **every** task in Group B, even if the spec doesn't explicitly say so. Catching warnings early avoids end-of-group rebase pain.
- The temporary commit + amend in Task C3.1 / C4.1 is the only `git commit --amend` in the plan and is allowed because it's done within minutes of the original commit on a branch only the implementer is touching. Do not amend any other commit.
- Group commits land in order A, B, C, D — never interleaved. If a workspace gate fails partway through Group B (Task BN), fix and continue *within* Group B; don't commit the partial state.
- `--force-with-lease` (Task C4.2) is the safe variant of `--force`; it refuses to push if origin moved unexpectedly. Don't substitute plain `--force`.
- The post-merge tasks (G1.x) are performed by whoever has push access to `main`. Sequence them after the PR merges; don't run them before.

---

*Plan synthesised: 2026-05-09. Spec reference: `docs/superpowers/specs/2026-05-09-m11-oq-bench-harness-design.md`. Brainstorm session: same date.*
