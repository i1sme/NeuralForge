# Milestone 5b — Bias-Aware Fusion + EliminateDropout + `--passes` filter — Design

> **Status:** Brainstormed and approved 2026-05-05. To be implemented in the
> `claude/m5b-bias-aware-fusion` worktree.
> **Source:** This spec captures the M5b brainstorming conversation. If
> something here disagrees with what was decided in the conversation, the
> conversation wins — file an amendment.

## 1. Overview

M5a closed kernel fusion for the `linear (no bias=true) → relu` pattern: a
UIR-pass framework + one fusion pass + asm-level inline `fmax` + a `--no-fuse`
CLI escape hatch + bit-exact numerical equivalence proven against the
classifier fixture. M5a deliberately scoped out three things, all carried
forward here:

1. **Bias-aware fusion** — `linear[bias=true] → relu` is currently rejected by
   `FuseLinearRelu` despite the arm64 codegen already supporting the asm
   shape (matmul → bias-add → fmax → store).
2. **`EliminateDropout` pass** — Dropout is currently a no-op at inference
   only because `profiles/arm64::buffer.rs` aliases the dropout node's buffer
   to its operand's. The UIR still carries the Dropout node; it should be
   removed up-front.
3. **`--passes <names>` CLI filter** — M5a's `--no-fuse` is a binary toggle
   for the whole pipeline. With two passes registered, users need finer
   control for debugging.

M5b ships all three as one milestone: each is small, all share the same
worktree, and the three exercise complementary surfaces (pass logic, pass
infrastructure, CLI ergonomics). The combined diff is smaller than M5a in
absolute terms because no new types or new traits are introduced — only
extensions to existing structures.

## 2. Goal

1. Lift the `linear_has_bias` restriction in `FuseLinearRelu` (3 lines
   removed). `linear[bias=true] → relu` becomes a fusion candidate; the
   existing arm64 `emit_linear` correctly stacks `matmul → bias-add → fmax
   → store` for fused-bias-relu Linears with no profile-side changes.
2. Implement `EliminateDropout` — a pass that removes every `Dropout` node
   from the UIR, remapping consumers to the dropout's operand. Functional
   victim/remap pattern, structurally similar to `FuseLinearRelu` but
   simpler (no consumer-count check, no producer mutation).
3. Rename `--no-fuse` → `--no-passes` (no alias). Add `--passes <names>`
   filter: comma-separated whitelist, canonical order enforced, validation
   for unknown / empty / duplicate / mutually-exclusive-with-`--no-passes`,
   stderr `note:` when user-specified order diverges from canonical.

## 3. Non-goals

- **Profile guide doc updates** (`docs/profile_guide/arm64.md`: bias-aware
  fusion section, `--no-passes` / `--passes` documentation, EliminateDropout
  note) → **M5c**.
- **`PROJECT_SPEC.md` milestones table close-out** for M5 → **M5c**.
- **Snapshot tests via `insta`.** Substring asserts continue to suffice;
  defer until a real case demands stronger pinning.
- **Pass dependency declaration / fixed-point iteration.** Two passes with
  a known-trivial order; document the ordering as a comment in
  `default_pipeline()`. Revisit when the third pass with a non-trivial
  interaction lands (M6+).
- **Multi-pattern fusion** (e.g., `linear → softmax_max` for upcoming
  attention milestones). M6+.
- **`--passes` reorder mode** (variant B from brainstorming Q4 — user-
  specified subset *and* order). M5b implements filter-only with canonical
  order. Reorder is a superset; can be added in M6+ without breaking
  existing call sites.
- **Pass-shared helper for the victim/remap pattern.** Two passes is too
  few examples to design a clean abstraction. Wait for a third pass with
  the same structural pattern — the "three strikes then refactor" rule.
  DEVLOG to capture this rationale.
- **Numerical fuzzing / property-based bit-exact**. The integration tests
  cover the cases that matter (classifier from M5a + mixed_args from M5b).
- **Bare-metal target / second profile / x86_64.** M6+ / M7+.

## 4. Pre-decided architectural calls

This section captures decisions made during the M5b brainstorming dialogue.
Each is a deliberate choice, not the only sensible one — the rejected
alternatives are recorded so the reasoning survives.

### 4.1. Pipeline order: `[EliminateDropout, FuseLinearRelu]`

EliminateDropout MUST run before FuseLinearRelu. Without this order, a
`linear → dropout → relu` pattern is invisible to FuseLinearRelu (it sees
Linear's consumer as Dropout, not Relu) and remains unfused forever. With
this order, EliminateDropout collapses the chain to `linear → relu` first,
then FuseLinearRelu applies.

`default_pipeline()` hard-codes the order with an explanatory comment.
A fixed-point iteration scheme would be premature generalisation: at two
passes with one trivial dependency, the ordering is known and small. When
M6+ introduces a pass with non-trivial interactions, a concrete example
will inform whether to switch to fixed-point or a dependency declaration
mechanism.

### 4.2. Profile dropout handling: keep `BufferLoc::Alias`

After M5b, in default mode, EliminateDropout removes every Dropout node
from the UIR before the profile sees it. But `--no-passes` (and
`--passes <subset>` without `eliminate_dropout`) leave Dropout in the UIR
that reaches the profile. Three options were considered:

- **A. Keep `BufferLoc::Alias(operand)` for Dropout in `profiles/arm64::buffer.rs`.**
  Profile remains complete relative to its input grammar. `--no-passes`
  continues to compile dropout-containing fixtures. Chosen.
- **B. Delete the alias arm; profile errors on Dropout via
  `LowerError::UnsupportedOp`.** Forces `--no-passes` to fail on dropout
  models with a clear error. Rejected: `--no-passes` is a verification
  tool. A verification tool that fails on valid UIR isn't a verification
  tool — it's a trap.
- **C. Keep alias, mark `#[deprecated]`.** Without a concrete migration
  plan to B, deprecation marks are noise. Rejected.

Profile (`profiles/arm64`) gets **no source changes** in M5b — the same
codegen path serves both fused (default) and un-fused (`--no-passes`)
modes.

### 4.3. CLI rename: `--no-fuse` → `--no-passes`, no alias

After M5b the pipeline contains two passes, only one of which is fusion.
`--no-fuse` becomes a lie-by-omission: it disables EliminateDropout too.
`--no-passes` describes the actual behaviour.

`v0` has no external consumers; backward-compat aliases are cargo-culting.
Clean break: `--no-fuse` is fully removed (no `#[allow(dead_code)]` shim,
no deprecated alias), `--no-passes` is the only flag.

### 4.4. `--passes <names>`: filter-only, canonical order

User specifies a comma-separated subset of registered pass names. The
filtered pipeline runs in canonical order (the order from
`default_pipeline()`), regardless of the order the user typed.

Rejected: full replace mode (user-specified subset *and* order). The
reorder feature is a footgun — the user can shoot themselves with a wrong
order and get silently un-fused output, with no signal until they read
the asm. The filter-only contract makes canonical order a safety
guarantee. M6+ can extend to replace mode without breaking existing
filter call sites.

When the user-typed order diverges from canonical (`--passes
fuse_linear_relu,eliminate_dropout` while canonical is the reverse), CLI
emits a `note:` to stderr explaining that the order was overridden.
This prevents confused-debugging-session "why did fusion not happen?"
mistakes.

### 4.5. No shared helper for the victim/remap pattern

`EliminateDropout` duplicates `FuseLinearRelu`'s 4-step skeleton (consumer
counts → identify victims → rebuild with remap → remap inputs/output).
Extracting a shared helper requires designing an interface across two
non-identical call sites — `FuseLinearRelu` mutates the producer
(`push(PostOp::Relu)`), checks consumer counts, and has 5 victim
criteria; `EliminateDropout` mutates nothing, has 1 criterion. The
resulting abstraction would be either a callback-zoo or a trait with
three methods, larger than the duplication it replaces.

Two examples is below the threshold for clean abstraction. The "three
strikes then refactor" rule applies: when a third pass with the same
structural pattern lands in M6+, the actual common shape will be visible
and the helper can be derived (not guessed). DEVLOG documents this
explicitly so the M6 author knows the pattern was intentional, not
accidental.

## 5. Pass interface — unchanged from M5a

`compiler::passes::UirPass` trait, `PassError`, `default_pipeline`,
`run_pipeline` — all unchanged. M5b extends `default_pipeline()` to
return two passes instead of one (with the canonical-order comment).
`EliminateDropout::run` follows the existing `FuseLinearRelu::run`
contract: takes `&Uir`, returns `Result<Uir, PassError>` with a freshly-
renumbered NodeId graph.

`PassError::InvalidInput { pass, reason, span }` remains the only variant.
Like `FuseLinearRelu` in M5a, `EliminateDropout` in M5b never returns
`Err` on `ir::build`-validated input — the variant is for defensive
future use.

## 6. UIR types — unchanged from M5a

`PostOp` remains `#[non_exhaustive]` with the single variant `Relu`. M5b
does not add a new post-op (Gelu / Tanh / Sigmoid / etc. are not in NFL
v0.1). `NodeKind::Op { fused_post_ops: Vec<PostOp> }` continues to carry
zero or one `Relu` per Linear (no double-fusion still applies).

## 7. Algorithm changes

### 7.1. `FuseLinearRelu` bias-lift

In `compiler/src/passes/fuse_linear_relu.rs`, delete the bias guard
(currently lines 81-83):

```rust
if linear_has_bias(attrs) {
    continue; // M5a scope: bias-aware fusion is M5b.
}
```

That is the entire pass-side change. The remaining four victim criteria
hold:

1. Op is `Relu`
2. Operands.len() == 1
3. Producer is `Linear`
4. Linear's `fused_post_ops` is empty
5. Linear has exactly one consumer (this Relu)

The arm64 `emit_linear` already stacks bias-add (line 80-83) before the
post-op loop (line 103-117) before the store (line 119-123). For a
Linear with `bias_offset.is_some()` AND `fused_post_ops == [Relu]`, the
emitted asm is, in order:

1. `fmov s4, wzr` once at function-header (since `needs_zero` is true).
2. matmul i/j/k loops, writing into `s0`.
3. `ldr s5, [x14, x4, lsl #2]` then `fadd s0, s0, s5` (bias-add, after k-loop).
4. `fmax s0, s0, s4` (post-op).
5. `str s0, [x12, x6, lsl #2]` (store).

Result: `y = relu(x*W + b)`, computed in one pass, no intermediate buffer.
This is the M4a-equivalent shape extended to the bias case.

### 7.2. `EliminateDropout` new pass

New file `compiler/src/passes/eliminate_dropout.rs`. Structure mirrors
`fuse_linear_relu.rs`:

```rust
//! `dropout` elimination pass (M5b).
//!
//! At inference time, dropout is a no-op (it only randomises during
//! training, which NFL v0.1 does not support). This pass removes every
//! Dropout node from the UIR, remapping its consumers to the dropout's
//! operand. After this pass, the `BufferLoc::Alias(operand)` machinery
//! in `profiles/arm64::buffer.rs` is unreachable in default mode (still
//! reachable for `--no-passes` and `--passes` filters that exclude this
//! pass).
//!
//! Functional: returns a fresh Uir with renumbered NodeIds.

use super::{PassError, UirPass};
use crate::ir::types::{Node, NodeKind};
use crate::ir::StdOp;
use crate::{NodeId, Uir, UirModel};
use std::collections::{HashMap, HashSet};

pub struct EliminateDropout;

impl UirPass for EliminateDropout {
    fn name(&self) -> &str {
        "eliminate_dropout"
    }

    fn run(&self, uir: &Uir) -> Result<Uir, PassError> {
        let mut new_models = Vec::with_capacity(uir.models.len());
        for model in &uir.models {
            new_models.push(eliminate_one_model(model)?);
        }
        Ok(Uir { models: new_models })
    }
}

/// Precondition: `model.nodes` is in topological order — every operand
/// NodeId is strictly less than the consumer's NodeId. `ir::build`
/// guarantees this. Same precondition as `FuseLinearRelu::fuse_one_model`.
///
/// Note: this 4-step skeleton (identify victims → rebuild with remap →
/// remap inputs/output) duplicates `FuseLinearRelu`. Extraction into a
/// shared helper is deferred to M6+ when a third pass with the same
/// pattern lands ("three strikes then refactor").
fn eliminate_one_model(model: &UirModel) -> Result<UirModel, PassError> {
    // Step 1: identify victims (every Dropout node).
    let victims: HashSet<NodeId> = model
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(id, node)| match &node.kind {
            NodeKind::Op { op: StdOp::Dropout, .. } => Some(id),
            _ => None,
        })
        .collect();

    // Step 2: build new model — skip victims, remap operands.
    let mut new_nodes: Vec<Node> = Vec::with_capacity(model.nodes.len());
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

    for (old_id, node) in model.nodes.iter().enumerate() {
        if victims.contains(&old_id) {
            // Dropout's operand becomes Dropout's "result" id-wise.
            // NFL grammar guarantees Dropout has exactly one operand.
            let operand_old_id = match &node.kind {
                NodeKind::Op { operands, .. } => operands[0],
                _ => unreachable!("victim must be Op"),
            };
            let operand_new_id = id_map[&operand_old_id];
            id_map.insert(old_id, operand_new_id);
            continue;
        }

        let mut new_node = node.clone();
        if let NodeKind::Op { operands, .. } = &mut new_node.kind {
            for op in operands.iter_mut() {
                *op = id_map[op];
            }
        }

        let new_id = new_nodes.len();
        new_nodes.push(new_node);
        id_map.insert(old_id, new_id);
    }

    // Step 3: remap inputs + output.
    let new_inputs: Vec<NodeId> = model.inputs.iter().map(|id| id_map[id]).collect();
    let new_output = id_map[&model.output];

    Ok(UirModel {
        name: model.name.clone(),
        nodes: new_nodes,
        inputs: new_inputs,
        output: new_output,
        source_span: model.source_span,
    })
}
```

The skeleton fits in ~70 lines (counting doc-comment + tests it grows to
~200, similar to `fuse_linear_relu.rs`).

### 7.3. Pipeline ordering

In `compiler/src/passes/mod.rs`:

```rust
pub mod eliminate_dropout;
pub mod fuse_linear_relu;

#[cfg(test)]
mod tests;

/// The default pipeline of passes, applied in order.
///
/// Order matters: EliminateDropout MUST run before FuseLinearRelu so
/// that `linear → dropout → relu` collapses to `linear → relu` and
/// becomes a fusion candidate. Reversed order leaves the pattern
/// unfused forever.
pub fn default_pipeline() -> Vec<Box<dyn UirPass>> {
    vec![
        Box::new(eliminate_dropout::EliminateDropout),
        Box::new(fuse_linear_relu::FuseLinearRelu),
    ]
}
```

The doc-comment on `default_pipeline` documents the dependency. M6+ may
introduce a fixed-point or dependency-declaration mechanism if a third
pass needs non-trivial coordination.

## 8. Profile (`profiles/arm64`) — no source changes

The arm64 codegen requires zero changes for M5b. Specifically:

- `emit_linear` already stacks bias-add (when `bias_offset.is_some()`)
  before the post-op loop. Bias-aware fusion needs no profile-side work.
- `BufferLoc::Alias(operand)` for `StdOp::Dropout` in `buffer.rs` remains
  the fallback for `--no-passes` mode (and any `--passes` filter that
  excludes `eliminate_dropout`). Stays as-is. No deprecation marks.
- `StdOp::Relu | StdOp::Dropout => BufferLoc::Alias(operands[0])` arm
  unchanged.
- No new `LowerError` variants. No new `emit_*` functions.

This is the principle of profile isolation in action: the profile
consumes whatever UIR it receives, and the upstream pipeline decides what
shape the UIR has.

## 9. CLI changes — `nflc/src/main.rs`

### 9.1. `CompileArgs` struct rename + new field

```rust
struct CompileArgs {
    path: PathBuf,
    profile: String,
    output: Option<PathBuf>,
    no_passes: bool,             // renamed from `no_fuse`
    passes: Option<Vec<String>>, // None = default; Some(vec) = filter
}
```

### 9.2. `parse_compile_args` extension

Argument parsing rules:

- `--no-passes` (no value) → `no_passes = true`. Rename of existing `--no-fuse`.
  M5a's `--no-fuse` is **fully removed** — no alias, no `#[allow(dead_code)]`
  shim, no deprecated marker.
- `--passes <list>` (next-arg value, comma-separated) → `passes = Some(parsed_vec)`.
  Strict split on `,`, no per-token whitespace trimming. `--passes a,b` is
  valid; `--passes "a, b"` (with embedded space) yields tokens `a` and ` b`,
  the second of which fails the unknown-name validation below. Users invoke
  as `--passes a,b` or `--passes "a,b"` (no spaces inside the value).
- Mutually exclusive: setting both `--no-passes` and `--passes <list>` →
  `Err("--no-passes and --passes are mutually exclusive")`.
- Empty value: `--passes ""` or `--passes` with nothing after → `Err("--passes value cannot be empty (use --no-passes to skip the pipeline)")`.

After parsing, `--passes` value is validated against
`default_pipeline().iter().map(|p| p.name())`:

- Unknown name → `Err("unknown pass '<name>' (available: <comma-joined dynamic list>)")`.
- Duplicate name → `Err("pass '<name>' specified more than once in --passes")`.

The available-list in the error message **must be derived dynamically**
from `default_pipeline()`, not hardcoded; otherwise the error message
goes stale when M6+ adds passes.

### 9.3. Filter application + order-divergence warning

In `run_compile`, after `ir::build` succeeds:

```rust
let post_pass_uir = if no_passes {
    eprintln!("note: passes skipped (--no-passes)");
    uir
} else {
    let canonical = compiler::passes::default_pipeline();
    let canonical_names: Vec<&str> = canonical.iter().map(|p| p.name()).collect();

    // Resolve the pipeline: full canonical, or filtered subset.
    let (pipeline, divergent) = match passes {
        None => (canonical, false),
        Some(user_names) => {
            // Filter canonical to retain only user-named passes,
            // preserving canonical order. Compare to user's order to
            // detect divergence.
            let user_set: HashSet<&str> = user_names.iter().map(String::as_str).collect();
            let filtered: Vec<Box<dyn UirPass>> = canonical
                .into_iter()
                .filter(|p| user_set.contains(p.name()))
                .collect();
            let canonical_filtered_names: Vec<&str> =
                filtered.iter().map(|p| p.name()).collect();
            // Order divergence: only meaningful when len >= 2.
            let div = user_names.len() >= 2 && user_names != canonical_filtered_names;
            (filtered, div)
        }
    };

    match compiler::passes::run_pipeline(&uir, &pipeline) {
        Ok(u) => {
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
        Err(e) => { /* render_error_with_snippet path, unchanged from M5a */ }
    }
};
```

Two notes (when divergent), in order:

```
note: applied passes: eliminate_dropout, fuse_linear_relu
note: pass order is canonical (eliminate_dropout, fuse_linear_relu); user-specified order ignored
```

Rationale for two-line vs. combined: each note carries one semantic load,
substring tests are independent (one for `applied passes:`, one for
`user-specified order ignored`), and the implementation is trivial
(applied-note always, divergence-note conditionally). A combined form
would push the conditional decoration into the format string.

### 9.4. `print_usage` update

```rust
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
    println!("                          [--passes <list>]  Run only listed passes (comma-separated)");
}
```

## 10. Acceptance criteria

1. `cargo build --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`,
   `cargo fmt --all -- --check` — all exit 0.
2. **Test suite count ≥ baseline recorded at M5b branch cut, no regression.**
   Implementer records the baseline (current main = 173) before starting Task
   1; final count must be ≥ baseline + new tests committed in this milestone.
   Hard numbers in commit messages are fine; the AC itself is the regression-
   only contract.
3. CLI smoke: `default mode`, `--no-passes`, `--passes <one>`,
   `--passes <invalid>`, `--no-passes --passes <list>` (mutually exclusive)
   — all behave per §9.
4. `fused_vs_unfused_mixed_args_match_numerically` integration test passes —
   bit-exact `assert_eq!` on every output element of `mixed_args.nfl`
   (which has `linear[16, bias=true] → relu` as an internal fusion candidate).
5. `pipeline_eliminates_dropout_before_fusing_linear_relu` integration test
   passes — synthetic UIR `linear → dropout → relu` collapses to two nodes
   (input + fused linear with `fused_post_ops == [Relu]`) end-to-end. Proves
   the order-dependency property.
6. **`--no-fuse` is fully removed.** No `#[allow(dead_code)]` shim, no
   deprecated alias, no `--no-fuse` substring anywhere in `nflc/src/` after
   M5b. `cargo build --workspace` produces no dead-code warnings related to
   former `--no-fuse` plumbing.
7. M5a's `fused_vs_unfused_classifier_match_numerically` continues to pass
   unchanged (regression check — bias-aware fusion lift must not affect the
   classifier path).
8. DEVLOG entry for M5b documents:
   - Bias-aware fusion as 1-line lift (the asm path was always ready);
   - EliminateDropout's no-shared-helper choice + "three strikes then
     refactor" rule for the M6+ author;
   - Dynamic available-list in the error message;
   - M5b → M5c slicing (profile guide doc + PROJECT_SPEC update remaining
     for M5 close-out).
9. `CLAUDE.md` "Current Status" section updated to reflect M5b complete:
   new flag names (`--no-passes`, `--passes`), two passes registered
   (eliminate_dropout, fuse_linear_relu), and M5c (profile guide doc +
   PROJECT_SPEC close-out) as the immediate next step. Mirrors the
   M5a closeout-commit pattern.

## 11. Test plan

### 11.1. Bias-aware fusion (FuseLinearRelu unit tests)

In `compiler/src/passes/fuse_linear_relu.rs::tests`:

- **Invert** `does_not_fuse_when_linear_has_bias` → `fuses_when_linear_has_bias`.
  Same fixture (`linear[2, bias=true] → relu`); now asserts: 2 nodes (input +
  fused linear), `fused_post_ops == [Relu]`, `attrs` still contains
  `bias=true` (preserved by fusion), `model.output == 1`. Net change in count: 0.
- **New** `fuses_chain_with_bias` — `linear → relu → linear[bias=true] → relu`
  via `build_uir`. Asserts: 3 nodes (input, fused1, fused2), both linears
  have `fused_post_ops == [Relu]`, the second linear's attrs include `bias=true`.
  Tests that bias-aware fusion composes with M5a's existing chain support. Net change in count: +1.

Net unit: **+1**. Plus 1 integration test in 11.4.

### 11.2. EliminateDropout (new module unit tests)

In `compiler/src/passes/eliminate_dropout.rs::tests`:

- `pass_name_is_stable` — `name() == "eliminate_dropout"`.
- `empty_uir_passes_unchanged` — `Uir { models: vec![] }` passes through.
- `removes_terminal_dropout` — synthetic UIR where `model.output IS dropout`.
  Asserts: `output` remaps to dropout's operand (which becomes the new
  terminal); dropout node removed from `nodes`.
- `removes_internal_dropout` — hand-built UIR `linear → dropout → softmax`.
  Asserts: 3 nodes (input + linear + softmax), softmax operands point at
  linear's new id. Hand-built (not via `build_uir`) keeps the test
  structural and independent of which fixtures happen to use dropout.
- `removes_chained_dropouts` — hand-built `linear → dropout → dropout → relu`.
  Asserts: 3 nodes (input + linear + relu), relu operands point at linear's
  new id. Both dropouts collapsed.
- `preserves_when_no_dropout` — UIR without any dropout passes through with
  identical structure (modulo NodeId remap, which is identity here).
- `multi_consumer_dropout` — hand-built `linear → dropout` where dropout is
  read by both relu and softmax (mirrors M5a's `fuses_when_relu_has_multiple_consumers`).
  Asserts: dropout removed; both relu and softmax operands now point at linear's new id.
- `model_inputs_and_output_remapped` — verifies inputs/output are remapped
  through the id_map, even when no input or output is itself a dropout
  (defensive coverage of the inputs/output remap arm in case dropout is
  internal).

Net: **+8** unit tests.

### 11.3. Pipeline integration (passes/tests.rs)

In `compiler/src/passes/tests.rs`:

- **Update** `default_pipeline_includes_fuse_linear_relu` →
  `default_pipeline_is_canonical_order`. Asserts: `[eliminate_dropout, fuse_linear_relu]`
  in this exact order via `pipeline.iter().map(|p| p.name())`. Net change in count: 0.
- **New** `pipeline_eliminates_dropout_before_fusing_linear_relu` — hand-built
  synthetic UIR `linear → dropout → relu`, run `run_pipeline(uir, &default_pipeline())`,
  assert: result has 2 nodes (input + fused linear with `fused_post_ops == [Relu]`),
  output points at the fused linear. **This is the load-bearing test that
  proves the order matters end-to-end.**

Net: **+1**.

### 11.4. CLI smoke tests (nflc/tests/cli_compile.rs)

- **Rename** `compile_with_no_fuse_skips_fusion` → `compile_with_no_passes_skips_pipeline`.
  Update assertions: stderr contains `note: passes skipped (--no-passes)`
  (was `--no-fuse`); does NOT contain `note: applied passes:`. Asm shape
  unchanged from M5a's no-fuse case (separate `.Lrelu_*:` loop). Net change in count: 0.
- **New** `compile_with_passes_filter_runs_only_selected` — invoke with
  `--passes fuse_linear_relu` against `tests/fixtures/m4_linear_relu.nfl`
  (no dropout in this fixture; filter exercise is purely about pipeline
  selection). Asserts: exit 0; stderr contains
  `note: applied passes: fuse_linear_relu` and does NOT contain
  `eliminate_dropout`; stdout asm contains `fmax s0, s0, s4` (fusion still
  applied since FuseLinearRelu is in the filtered set).
- **New** `compile_with_passes_unknown_name_rejected` — invoke with
  `--passes foo`. Asserts: exit 1; stderr contains `unknown pass 'foo'`
  AND `available:` (substring check, not full list match — keeps test
  resilient to M6+ pass additions).
- **New** `compile_with_passes_order_warning` — invoke with
  `--passes fuse_linear_relu,eliminate_dropout` (reverse). Asserts: exit 0;
  stderr contains BOTH `note: applied passes: eliminate_dropout, fuse_linear_relu`
  AND `note: pass order is canonical (...); user-specified order ignored`;
  stdout asm equivalent to default mode.
- **New** `compile_no_passes_and_passes_rejected` — invoke with
  `--no-passes --passes fuse_linear_relu`. Asserts: exit 1; stderr contains
  `mutually exclusive`.

Net: **+4** new (1 renamed in place).

### 11.5. Integration tests (profiles/arm64/tests/integration.rs)

- **New** `fused_vs_unfused_mixed_args_match_numerically` — mirrors M5a's
  classifier integration test against `mixed_args.nfl`. Compiles via both
  paths (`default_pipeline()` vs. raw UIR), pre-asserts asm shape (fused has
  `fmax` AND `fadd s0, s0, s5` for bias inside one emit_linear; unfused has
  separate `.Lrelu_*:` and `fadd s5` in the standalone bias-add path),
  compiles both dylibs, FFI-calls with deterministic input/params, asserts
  bit-exact `assert_eq!` on every output element. The integration-side
  assertion that bias-aware fusion preserves numerics. Skips on
  non-aarch64 / no-cc per existing convention.

Net: **+1**.

**M4b/M5a integration tests** — automatic. `mixed_args_runs_correctly` (already
on default-fused path after M5a Task 10) starts fusing `linear[16, bias=true] → relu`
internally after M5b. Numerical assertions hold via bit-exactness. **Test
not rewritten.**

### 11.6. Total

| Surface | Net new |
|---|---|
| FuseLinearRelu unit | +1 |
| EliminateDropout unit | +8 |
| Pipeline integration | +1 |
| CLI smoke | +4 |
| FFI integration | +1 |
| **Total** | **+15** |

Baseline at M5b branch cut: 173 (post-M5a-merge). Expected post-M5b: ~188.
The AC requires `≥ baseline + new`, not `== 188`. Final count documented
in commit messages and DEVLOG.

## 12. Tech debt — carried forward

1. **Profile guide doc updates** (`docs/profile_guide/arm64.md`): bias-aware
   fusion section, `--no-passes` / `--passes` documentation, EliminateDropout
   removal note. → **M5c**.
2. **PROJECT_SPEC.md milestones table** close-out for M5 → **M5c**.
3. **Pass-shared helper for victim/remap pattern** — defer to M6+ "three
   strikes". DEVLOG to capture rationale.
4. **`--passes` reorder mode** (B-variant from Q4) — only if a real
   research/debugging case demands it. M6+.
5. **Pass dependency declaration / fixed-point iteration** — when a third
   pass with non-trivial interaction lands (M6+).
6. **Multi-pattern fusion** (e.g., `linear → softmax_max` for attention) —
   M6+.
7. **Snapshot tests via `insta`** — M5c+ if a real case demands it.

## 13. Open questions

None. All design decisions captured in §4. Implementation can proceed
straight to writing-plans.
