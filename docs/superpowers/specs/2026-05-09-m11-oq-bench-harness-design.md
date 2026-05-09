# Milestone 11 — OQ-BENCH Harness — Design

> Brainstormed: 2026-05-09
> Strategic axis: **Trigger-driven cleanup** — closes OQ-BENCH (`PROJECT_SPEC.md` §"Open Questions" / "Trigger-driven cleanup"; trigger fired on M9 merge, opened explicitly by M10 spec §11.2 deferral).
> Predecessor: M10 (NFL v0.2 self-attention + 4D codegen)
> Status: spec draft for plan synthesis

---

## 1. Overview

M11 closes the OQ-BENCH trigger that fired when M9 merged. It builds a
benchmark harness that compiles fixed NFL fixtures through both `arm64`
and `x86_64` profiles, runs each binary on its own native host, and
reports inference latency in a form a developer can read at a glance.

The deliverable shape is intentionally minimal: a single new workspace
crate (`bench/`), a single CI workflow (`bench.yml`) that emits per-arch
half-reports into GitHub Actions Job Summary, and a manually-maintained
`bench/results/YYYY-MM-DD.md` for cross-profile comparison. There is
**no regression gate, no artifact sharing between matrix legs, no aggregator
job** — the spec preserves the M10 §11.2 rule that matrix jobs remain
independent.

The strategic claim being validated is that **debt incurred by triggered
cleanup items closes promptly**: PROJECT_SPEC's "Trigger-driven cleanup"
list is meant to fire and resolve, not accumulate. M11 is the worked
example for OQ-BENCH; future trigger-driven items follow the same shape.

The numerical claim being established is the **scalar-only baseline** for
both profiles. Future SIMD work (Axis 1) and future ABI-changing
modelling work (Axis 2 A1 multi-input grammar) will reference these
numbers as the "before" state.

---

## 2. Goals

Ship a single PR with ~4 atomic commits (final count delegated to
`writing-plans`), plus one **post-merge follow-up commit** to land
the first combined report (rationale: §11 #5). Together they:

1. Add a new workspace member `bench/` (crate name `bench`) with a
   single-binary `cargo run -p bench --release` CLI accepting
   `--profile {arm64|x86_64}`, `--format {markdown|github-summary}`,
   and `--seed <u64>` (default `42`).
2. Implement the bench harness as a single-file `bench/src/main.rs`:
   parse fixture → `profile.lower(&uir)` → `cc -shared` → `dlopen` →
   warmup × 10 → measurement × 100 → emit median + p95 to stdout.
3. Land three bench fixtures via existing test fixtures
   (no new `.nfl` files): `classifier`, `large_classifier_k`,
   `self_attention`. Each fixture is wired by absolute path
   constant + a static `&str` purpose label.
4. Add `.github/workflows/bench.yml` with two matrix legs
   (`macos-14` for arm64, `ubuntu-latest` for x86_64). Each leg
   runs the host's native profile and writes to `$GITHUB_STEP_SUMMARY`.
   Trigger: `workflow_dispatch` + `push: branches: [main]` with
   `paths-ignore: ['bench/results/**', 'docs/**', '**.md']`.
5. Update `PROJECT_SPEC.md` (mark OQ-BENCH closed under
   "Trigger-driven cleanup" with the closing PR/commit reference,
   matching the OQ-NEW closure pattern), `CLAUDE.md` (Repository
   Structure tree gains `bench/` member), `DEVLOG.md`.
6. **(Post-merge follow-up commit on `main`.)** Manually compose
   `bench/results/<merge-date>.md` from the two Job Summary outputs
   of the inaugural CI run that fires when the PR merges, and push.

The merge of the PR closes M11. The post-merge follow-up commit is
the first artefact of the ongoing `bench/results/` series, not a
re-opening of M11.

---

## 3. Strategic Positioning

### 3.1 What M11 Proves

- **Triggered cleanup closes promptly.** OQ-BENCH fired on M9 merge,
  was deferred during M10 (M10 §11.2), and closes in M11 as the next
  milestone. The pattern is: trigger fires → next milestone closes
  it → no debt accumulation.
- **Scalar baseline is established.** Future SIMD claims ("AVX2
  doubled matmul throughput") need a baseline number to compare
  against. M11 produces that number for both profiles.
- **Light-CI integration is sufficient for informational data.**
  Job Summary is the right primitive for non-gated metrics; no
  artifact orchestration is required.

### 3.2 What M11 Does Not Prove

- **Cross-profile speedup at the byte level.** Hosts differ (Apple
  M1 vs ubuntu-latest x86_64 SKU varies); reported numbers are
  apples-to-oranges across hosts but valid within a single profile.
- **Performance competitiveness against PyTorch / ONNX Runtime / TF.**
  No comparison to industry baselines is in scope.
- **Per-op timing.** Per-op breakdown was discussed and rejected
  during brainstorm — dimensional analysis from fixture shapes
  already tells which op dominates (see §6.1).
- **Stability under noise.** GitHub-hosted runners share underlying
  hardware with other tenants; absolute numbers are noisy. Median +
  p95 mitigate this within a single run; cross-run variance is
  declared out of scope.

### 3.3 Trigger Closure

Per `PROJECT_SPEC.md` §"Trigger-driven cleanup", OQ-BENCH's exit
criteria are:
- harness compiles a single NFL source through both profiles ✓
- runs both binaries with the same input/params ✓
  (same fixtures, same seed; per-host execution, not single-host)
- reports timing side-by-side ✓
  (combined `bench/results/YYYY-MM-DD.md` is the side-by-side
  artefact)
- output format is markdown ✓
- multiple fixtures ✓ (three, see §6)
- no regression gate ✓ (informational only)

PROJECT_SPEC entry transitions to "**Closed in M11 (commit `<TBD>`)**"
with the merging-commit hash, matching the OQ-NEW closure precedent.

---

## 4. Scope

### 4.1 In scope

- New `bench/` workspace crate, `Cargo.toml` + `src/main.rs` only.
- Three bench-grade fixtures, reused as-is from `tests/fixtures/`.
- `--profile`, `--format`, `--seed` CLI flags.
- Two output formats: `markdown` (stdout, dev-friendly), `github-summary`
  (stdout, tuned for `$GITHUB_STEP_SUMMARY`; difference is minor —
  see §7.3).
- `.github/workflows/bench.yml` with the matrix and Job Summary writes.
- First `bench/results/<date>.md` cross-profile report (manually
  composed).

### 4.2 Out of scope

- Per-op breakdown (rejected — see §3.2).
- Compile-time benchmarking (orthogonal to scalar-vs-SIMD; can be
  added later as a column).
- `passes ON / passes OFF` ablation columns (orthogonal to scalar-vs-SIMD;
  separate milestone if ever needed).
- Cross-run variance tracking, regression alerts, dashboard.
- Comparison against PyTorch / TF / ONNX Runtime.
- Per-fixture `*.bench-data` files with hardcoded inputs (rejected —
  deterministic seed via `StdRng` is single source of truth).
- New NFL fixtures sized for bench (rejected — `large_classifier_k`'s
  ~50 µs median is above noise floor; new fixtures would duplicate
  signals already covered).
- AVX/SVE/NEON/SIMD codegen (Axis 1 territory, future milestone).
- Bare-metal `expf` (Axis 3 territory, future milestone).
- macOS x86_64 (Mach-O) host. The matrix leg for x86_64 is Linux-only,
  matching the existing `unit` job in `ci.yml`.

---

## 5. Architecture

### 5.1 Workspace layout

```
bench/
├── Cargo.toml              ← new workspace member
├── src/
│   └── main.rs             ← single-file ~250–350 lines
└── results/
    └── 2026-05-09.md       ← first combined cross-profile report
```

`bench/results/` is checked in. Future bench reports add new
date-stamped markdown files; never overwrite past ones. Old reports
are historical record (frozen-in-time snapshots of scalar baseline
at a specific commit).

### 5.2 Crate dependencies

| Dependency      | Why                                                              |
|-----------------|------------------------------------------------------------------|
| `compiler`      | `compiler::parse(&src)` to obtain `NflSource`, `compiler::ir::build` for `Uir`. |
| `profile-api`   | `Asm`, `FnSig`, `LowerError` types (returned by `Profile::lower`). |
| `profiles-arm64`| `Arm64Profile` (concrete profile, gated by `--profile arm64`).   |
| `profiles-x86_64`| `X8664Profile` (concrete profile, gated by `--profile x86_64`). |
| `libloading`    | `Library::new(.dylib | .so)` + `library.get::<extern "C" fn>`.   |
| `rand`          | `StdRng::seed_from_u64(seed)` + `Standard` distribution for f32. |

`criterion` and `divan` are explicitly **not** dependencies. They are
proper benchmarking frameworks with statistical machinery far beyond
what M11 needs; for "10 warmup + 100 measurement, median + p95"
hand-rolled timing via `std::time::Instant` is sufficient and
transparent.

### 5.3 Workspace Cargo.toml change

`Cargo.toml` `[workspace] members = [...]` gains `"bench"` as the
sixth member, alphabetically placed.

### 5.4 Shared compile-to-dylib helper

The arm64 integration tests already have
`profiles/arm64/tests/common/mod.rs::compile_to_dylib` (assembles via
`cc -shared -arch arm64`; returns `.dylib` path). The bench's
equivalent is host-arch-aware:

- macOS arm64 host + `--profile arm64`: `cc -shared -arch arm64 -o libbench.dylib bench.s`
- Linux x86_64 host + `--profile x86_64`: `cc -shared -fPIC -o libbench.so bench.s`
- Any other host/profile combination: bench errors out:
  `host arch <H> cannot natively run --profile <P>; cross-execution unsupported`.

The bench keeps its own `compile_to_dylib_for_host` function in
`main.rs` rather than extracting the integration-test helper. Reasons:
the test helper is hard-coded to arm64 (`-arch arm64`), and lifting it
to a shared crate adds a new crate-on-crate dependency just to share
~30 lines of `Command::new("cc")` wrapping. The cost-benefit doesn't
justify shared code at M11 scale; if a third caller appears, that's
the trigger for extraction.

### 5.5 FFI signature

The compiled bench binary exports an `extern "C" fn` per UIR model:

```rust
type ForwardFn = unsafe extern "C" fn(
    input: *const f32,
    params: *const f32,
    output: *mut f32,
);
```

Buffer sizes come from `FnSig`:
- `input` length: `sig.input_floats`
- `params` length: `sig.params_floats`
- `output` length: `sig.output_floats`

The bench:
1. Parses the fixture (`compiler::parse`).
2. Builds UIR (`compiler::ir::build`).
3. Lowers via the chosen profile (`profile.lower(&uir)?`) → `Asm`
   with `functions: Vec<FnSig>`.
4. For each `FnSig` (typically one per fixture; M3-M10 fixtures are
   single-model): allocates `Vec<f32>` of `input_floats`,
   `params_floats`, and `output_floats`. Fills `input` and `params`
   from `StdRng::seed_from_u64(seed)` with `Standard` distribution.
   `output` is zero-initialised.
5. Writes `asm.source` to a tempdir `.s` file.
6. Invokes `cc -shared` → `.dylib` (macOS) or `.so` (Linux).
7. `libloading::Library::new(path)` → `library.get::<ForwardFn>(b"<sig.name>")`.
   Symbol-prefix is handled per profile: arm64 (Mach-O) prepends `_`;
   x86_64 (ELF) does not. The bench reads `profile.sym_prefix()` and
   prepends accordingly.
8. Times warmup × 10 + measurement × 100 calls of `forward(input, params, output)`.
9. Emits the per-fixture row.

Step 4's `params_floats` may be 0 for fixtures that have only `Matmul`/
`MulScalar`-only ops with no `Linear` parameters (e.g. `self_attention`).
The bench passes a non-null aligned pointer to a zero-length buffer
(via `vec![0f32; 0].as_ptr()` — Rust guarantees `vec![]` returns a
non-null aligned `as_ptr()`); the generated assembly never
dereferences, matching the existing M10 FFI integration-test contract.

---

## 6. Methodology

### 6.1 Timing

- `std::time::Instant::now()` before the FFI call,
  `.elapsed().as_nanos() as u64` after.
- Per fixture: 10 warmup + 100 measurement iterations.
- Warmup discarded entirely.
- Measurement collected into `Vec<u64>` of nanoseconds.

### 6.2 Statistics

Two reported figures per fixture:
- **Median** (50th percentile): sort the 100 measurements,
  return `samples[50]`. Robust against right-skewed inference latency
  distributions (typical of short-running kernels with OS scheduling
  jitter).
- **p95** (95th percentile): `samples[95]`. Shows the right-tail
  behaviour; for noisy hosts the gap between median and p95 is the
  noise envelope.

`mean` and `stddev` are explicitly **not** reported. Inference latency
on shared CI runners is right-skewed (hardware sharing causes
occasional long outliers), so `mean ± stddev` overstates noise and
hides the typical case.

### 6.3 Why median over min

`min` rewards single luckiest sample, which on a shared runner is
often anomalous (cold-cache to warm-cache transition exactly synced
with a kernel quiescent moment). `median` reflects what 50%+ of runs
look like, which is closer to "what users experience".

### 6.4 Warmup rationale

10 warmup iterations cover:
- macOS arm64: dyld lazy binding for the `bl _expf` PLT-equivalent
  resolves on first `bl _expf` execution.
- Linux x86_64: PLT lazy binding for `call expf@PLT` resolves on
  first call.
- Both: I-cache and D-cache warm-up, branch predictor priming,
  TLB warm-up for buffer pages.

10 is conservative — for these specific fixtures, dyld/PLT resolves
on the first call and caches warm by ~2-3 iterations. 10 is a margin.

### 6.5 Determinism

Inputs and params are derived from a fixed seed (`--seed 42` by
default, overridable). `StdRng::seed_from_u64(seed)` is the PRNG;
`Standard` distribution over `f32` produces values in `[0.0, 1.0)`.

The same seed is used for input and params; in practice this means
draw input first (size `input_floats`), then params (size
`params_floats`), in that order. Identical across runs and across
hosts.

This makes timing comparable across a single bench invocation but does
**not** guarantee output equality with `tests/fixtures/` integration
tests (which use their own input distribution). Output equality is
out of scope for M11 — correctness is established by M3–M10 FFI
integration tests; M11 trusts that contract.

---

## 7. Output Format

### 7.1 Per-profile Job Summary (markdown)

Format identical for stdout (`--format markdown`) and Job Summary
(`--format github-summary`); the github-summary variant differs only
in metadata source (CI env vars vs `uname`/`sw_vers`):

```markdown
## NeuralForge bench — arm64 profile

**Host:** macos-14, Apple M1 @ 3.2 GHz
**Build:** release, single-thread
**Methodology:** 10 warmup + 100 measurement, median + p95
**Seed:** 42 (StdRng::seed_from_u64)
**Compiled at:** <git rev-parse --short HEAD>

| Fixture              | Median (µs) | p95 (µs) | Purpose                          |
|----------------------|-------------|----------|----------------------------------|
| classifier           |    2845.3   |  3104.7  | matmul-mass (3-layer MLP)        |
| large_classifier_k   |      48.7   |    52.1  | large-K inner-loop accumulator   |
| self_attention       |     108.4   |   119.5  | expf/softmax dispatch overhead   |
```

### 7.2 Combined cross-profile report (markdown)

Created **manually** by reading both Job Summaries from the inaugural
CI run, copying numbers, and committing to `bench/results/YYYY-MM-DD.md`.
Subsequent updates follow the same manual flow — no aggregator job
ever exists.

```markdown
# NeuralForge bench — 2026-05-09

Source: GitHub Actions run #NNN ([arm64 leg](url), [x86_64 leg](url)).
Commit: <git rev-parse HEAD>

| Fixture              | arm64 median | arm64 p95 | x86_64 median | x86_64 p95 | arm64 / x86_64 |
|----------------------|--------------|-----------|---------------|------------|----------------|
| classifier           |   2845 µs    |  3104 µs  |   4013 µs     |  4567 µs   |     0.71×      |
| large_classifier_k   |     49 µs    |    52 µs  |     71 µs     |    78 µs   |     0.69×      |
| self_attention       |    108 µs    |   120 µs  |    136 µs     |   142 µs   |     0.80×      |

**Hosts:**
- arm64 = macos-14 (Apple M1 @ 3.2 GHz)
- x86_64 = ubuntu-latest (Intel Xeon @ 2.8 GHz, runner SKU varies)

Ratio is computed on **median** (not p95). Ratio < 1 means arm64 is
faster.
```

### 7.3 Format flag mechanics

`--format markdown` writes the §7.1 template to stdout, with
metadata sourced from the host: `Host:` derived from `std::env::consts::OS`
+ `std::env::consts::ARCH` + a best-effort `sw_vers` (macOS) or
`/proc/cpuinfo` (Linux) read.

`--format github-summary` is identical, with `Host:` instead sourced
from CI env vars (`RUNNER_OS`, `RUNNER_ARCH`). When env vars are
absent (i.e. dev runs the flag locally), it falls back to the same
host-derived values.

Numbers are formatted with one decimal for µs; figures over 1000 µs
display rounded to integer (e.g. `2845`, not `2845.3`) — small
fixtures need decimal precision, large ones don't.

---

## 8. Fixtures

Three fixtures, reused as-is from `tests/fixtures/` — no new `.nfl`
files. Each chosen for an **orthogonal** signal; none duplicates
another.

| Fixture                | Path                                     | Purpose label                   | Approx. latency (scalar) |
|------------------------|------------------------------------------|---------------------------------|---------------------------|
| `classifier`           | `tests/fixtures/classifier.nfl`          | matmul-mass (3-layer MLP)       | ~3.4 ms                  |
| `large_classifier_k`   | `tests/fixtures/large_classifier_k.nfl`  | large-K inner-loop accumulator  | ~50 µs                   |
| `self_attention`       | `tests/fixtures/self_attention.nfl`      | expf/softmax dispatch overhead  | ~100 µs                  |

### 8.1 Why these three

- `classifier`: 3 linear layers across `[batch=32]` with hidden dims
  784→512→256→10; ~34M FLOPs per forward. Matmul dominates (>99% of
  flops); exercises fused `linear → relu` and `linear → softmax`
  (default pipeline ON). Primary signal for **arm64 FMA vs x86_64
  `mulss + addss`** matmul throughput.
- `large_classifier_k`: 1 linear layer `[2, 8192] @ [8192, 10]`;
  ~327K FLOPs per forward. Matmul not dominant in absolute mass
  (output dim 10 is tight), but **inner accumulator loop runs over
  K=8192 elements per output**, so this fixture stresses inner-loop
  scalar-accumulator behaviour (cache prefetch, register pressure on
  the running sum, branch prediction on the K-loop). Distinct signal
  from `classifier`; not a substitute.
- `self_attention`: M10 acceptance fixture, `[batch=2, heads=4, seq=16,
  head_dim=16]`, ~130K FLOPs in matmul plus 2048 `expf` calls in
  softmax. **expf-dominated** — both profiles route through libm
  (`bl _expf` on arm64, `call expf@PLT` on x86_64), so the SIMD
  question doesn't apply to the dominant cost. The fixture instead
  measures **dispatch overhead** — function-call boundary cost,
  loop-iteration overhead at small kernel size, register-spill
  cost around extern math calls (the M10 hazard the spec already
  validated).

### 8.2 Why not more fixtures

- `large_classifier_n`: same shape story as `large_classifier_k`
  with the long axis on N. Duplicates the inner-loop signal on a
  different axis but doesn't tell a different story about the
  arm64-vs-x86_64 question.
- `softmax_with_bias`: dominated by softmax + small linear; signal
  overlaps with `self_attention`'s expf dominance.
- `m4_linear_relu`: too small for stable timing.
- `tiny_mlp`, `pipeline_styles`, `comments`, `mixed_args`,
  `dropout_only`: parser/grammar coverage, not bench-grade.

### 8.3 Pass pipeline

All bench runs use the **default pipeline** (`EliminateDropout +
FuseLinearRelu + FuseLinearSoftmax`, ON). No `--no-passes` runs are
in scope. Fusion ROI ablation is a separate (future, optional)
milestone if ever needed.

---

## 9. CI Integration

### 9.1 New workflow file

`.github/workflows/bench.yml`:

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

jobs:
  bench:
    strategy:
      fail-fast: false
      matrix:
        include:
          - { os: macos-14,       profile: arm64  }
          - { os: ubuntu-latest,  profile: x86_64 }
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - name: Build bench (release)
        run: cargo build -p bench --release
      - name: Run bench
        run: |
          cargo run -p bench --release -- \
            --profile ${{ matrix.profile }} \
            --format github-summary \
            --seed 42 \
            >> $GITHUB_STEP_SUMMARY
```

### 9.2 Why `paths-ignore`

Without `paths-ignore`, every commit to `bench/results/*.md` (the
manually-maintained combined report) triggers another bench run,
which produces another summary, which a human eventually decides to
record into another `bench/results/*.md` file, which triggers another
bench run. Not an infinite loop (the human gates the recording), but
unnecessary CI minutes.

`paths-ignore` excludes:
- `bench/results/**`: the combined-report files themselves.
- `docs/**`: doc-only commits don't change codegen.
- `**.md`: README, DEVLOG, CLAUDE.md, PROJECT_SPEC.md, spec/plan
  docs; none affect generated assembly.

### 9.3 Why `fail-fast: false`

If one matrix leg has a transient failure (e.g. `cc` not on PATH on
ubuntu), the other leg's data is still useful. Bench is informational;
one bad leg shouldn't cancel the other.

### 9.4 Why no PR trigger

Bench runs only on `workflow_dispatch` and `push: main`. PRs do not
trigger bench. Reasons:
- Each bench run consumes ~5 min of CI runner time (release build +
  3 fixtures × 110 iterations × 2 architectures). Multiplying by the
  number of PR pushes per PR is wasteful for non-gated data.
- Bench is informational; a PR that "regresses" bench by 2% does not
  fail merge, so requiring the data per-PR adds no signal.
- `workflow_dispatch` covers the case where a developer wants to
  measure a specific PR before merging.

### 9.5 No regression gate

The workflow does not compare numbers against any baseline. It does
not fail on any threshold. It writes to Job Summary and exits 0
(unless the bench binary panics, in which case the matrix leg fails
and is visible in the run UI).

A regression-gating workflow is a future milestone if and when
needed; M11's scope is "produce the numbers", not "police the numbers".

---

## 10. Reproducibility & Noise

### 10.1 What is controlled

- **Seed.** Default `42`, overridable via `--seed`.
- **Build profile.** `--release` always (debug numbers are
  meaningless).
- **Iteration counts.** 10 warmup + 100 measurement, hard-coded in
  `bench/src/main.rs`. Not configurable in M11 — keeps the report
  format invariant.
- **Pass pipeline.** Default ON, hard-coded.
- **Fixture set.** Three fixtures, hard-coded.

### 10.2 What is not controlled

- **CPU frequency / turbo boost / governor.** GitHub-hosted runners
  do not expose CPU-management primitives without sudo, and even with
  sudo the underlying virtualisation hides much of the control.
- **Co-tenancy.** GitHub runners share underlying hardware with
  other tenants; cross-run absolute-µs values vary.
- **`RUSTFLAGS=-C target-cpu=native`.** Explicitly **not** set.
  We bench what we ship; what we ship is `--release` with
  default codegen flags. Setting `target-cpu=native` would inflate
  the scalar baseline by enabling host-specific instructions in
  the host-side timing loop (not the generated `.dylib`/`.so`,
  which is unaffected — but cleaner to keep all build flags
  identical to a normal release build).
- **Memory layout / huge pages / NUMA.** Not relevant on
  single-thread tiny-buffer fixtures.

### 10.3 Mitigations

- Median + p95 reporting absorbs single-outlier noise.
- Multiple independent measurements (100) per fixture saturate the
  noise distribution.
- Same workflow file is used for every run, so noise variability is
  consistent across reports.

The combined report (§7.2) explicitly notes "Hosts vary; absolute
numbers are noisy — comparisons within a single CI run are reliable;
cross-run comparisons are not."

---

## 11. Acceptance Criteria

The PR closing M11 must:

1. **Compile workspace clean.**
   - `cargo build --workspace` passes.
   - `cargo clippy --workspace --all-targets -- -D warnings` passes.
   - `cargo fmt --all -- --check` passes.

2. **Test count goes up monotonically.** Specifically, the bench
   crate ships at least:
   - 1 unit test in `bench/src/main.rs` confirming buffer-size
     derivation from `FnSig` matches expected for one fixture.
   - 1 unit test confirming markdown rendering produces a stable
     header structure (no whitespace drift).
   Project total goes from 331 → ≥ 333.

3. **Workflow file lints.** `actionlint` (run locally if available)
   or hand-review confirms the workflow YAML parses and the matrix
   is well-formed.

4. **CI run produces Job Summary on both legs.** Triggering the
   workflow via `workflow_dispatch` from a feature branch produces
   two Job Summary outputs containing the §7.1 table layout. This
   is verified manually by the implementer before opening the PR.

5. **First combined report is checked in as a follow-up commit on
   `main`.** The closing PR cannot contain `bench/results/<date>.md`:
   the inaugural CI run that produces the numbers requires the
   workflow file to already be on `main`, which only happens once the
   PR is merged. The sequence is:
   1. PR merges (4 atomic commits: skeleton, harness, workflow, docs).
   2. `bench.yml` runs on the merge push to `main`, producing two
      Job Summary outputs.
   3. Implementer manually composes `bench/results/<date>.md` from
      both Job Summaries and pushes a follow-up commit directly to
      `main` (or via a tiny PR if branch protection requires).
   4. M11 is now fully closed.

   The PROJECT_SPEC.md OQ-BENCH "Closed in M11 (commit `<hash>`)"
   reference points at the merging PR's squash-merge commit, not at
   the follow-up — closure is declared at merge, the follow-up is
   the first instance of an ongoing artefact.

6. **PROJECT_SPEC.md and DEVLOG.md updated.**
   - PROJECT_SPEC.md OQ-BENCH entry transitions to "Closed in M11
     (commit `<hash>`)".
   - PROJECT_SPEC.md Current Status bumped to reference M11 closure.
   - DEVLOG.md gets the standard milestone-closure entry.

7. **CLAUDE.md updated.** The Repository Structure tree adds the
   `bench/` member.

8. **No artifact sharing introduced.** `bench.yml` matrix legs are
   independent. No `actions/upload-artifact` or `actions/download-artifact`
   anywhere.

9. **No regression gate introduced.** No threshold, no comparison,
   no exit-1 on any number.

---

## 12. Non-Goals (explicit)

- New NFL fixture files: rejected (existing fixtures are bench-grade
  per §8.1).
- New language constructs: rejected (out of scope for triggered cleanup).
- ABI changes (`FnSig`, `extern "C" fn` shape): rejected (M11 is
  consumer-only of the M9 ABI contract).
- Changes to either profile's codegen: rejected (M11 measures, doesn't
  modify).
- Changes to existing CI workflows: rejected (`bench.yml` is new and
  isolated; `ci.yml` is unchanged).
- Aggregator CI job, artifact upload/download, cross-leg coordination:
  rejected per M10 §11.2 rule, reaffirmed by M11.
- Compile-time benchmarking: out of scope.
- Per-op breakdown: out of scope.
- `--no-passes` ablation: out of scope.
- `criterion` / `divan` integration: out of scope.

---

## 13. Open Questions for `writing-plans`

The plan synthesis stage will need to answer:

1. **Commit grouping.** §2 estimates ~4 atomic commits in the closing
   PR + 1 post-merge follow-up. Plan determines exact split — likely
   (a) workspace member skeleton (`bench/Cargo.toml` + empty
   `main.rs` + workspace `Cargo.toml` member-list update), (b)
   harness implementation + unit tests, (c) `bench.yml` workflow,
   (d) docs closure (PROJECT_SPEC, DEVLOG, CLAUDE.md). The
   post-merge commit (e) `bench/results/<date>.md` is sequenced
   per §11 #5 — it cannot precede merge.
2. **Local-dev story.** Does `cargo run -p bench --release -- --profile
   arm64` on a developer's macOS arm64 box produce useful output? Plan
   confirms by hand-running and including in the verification checklist.
3. **`compile_to_dylib_for_host` precise signature.** §5.4 sketches
   the function but plan settles return type (path? handle?) and error
   propagation (`Result<PathBuf, std::io::Error>` likely).
4. **Symbol-name lookup.** `library.get::<ForwardFn>(b"<name>")` —
   the symbol name is `sig.name` for ELF and `format!("_{}", sig.name)`
   for Mach-O? `libloading` typically handles the underscore on macOS
   automatically, but plan verifies by hand-test on macos-14.
5. **`workflow_dispatch` smoke**: how does the implementer exercise
   the workflow on the feature branch before opening the PR? Plan
   sketches the flow (push to `claude/<branch>`, trigger via gh-cli
   `gh workflow run bench.yml --ref <branch>`).

These are implementation-level details surface in plan synthesis; the
spec defers them.

---

## 14. Predecessor Lessons Applied

- **M10 §11.2** ("Avoid CI artifact sharing") is preserved verbatim:
  no aggregator job; matrix legs are independent.
- **M9 FFI register preservation** is irrelevant to M11 (the bench
  consumes M9/M10 codegen, doesn't modify it).
- **Trigger-driven cleanup as obligation** (this brainstorm): M11
  exists *because* OQ-BENCH triggered on M9 merge. If a future
  brainstorm is presented with multiple triggered items still open,
  this spec is the precedent for "close them as the next milestone,
  not later".

---

*Brainstorming session: 2026-05-09. Implementation plan to be
synthesised by `superpowers:writing-plans`.*
