// SPDX-License-Identifier: Apache-2.0
#![cfg(all(target_os = "linux", target_arch = "x86_64"))]

//! M9 end-to-end FFI integration tests for the x86_64 Linux ELF profile.
//!
//! Mirrors profiles/arm64/tests/integration.rs. Each test loads a
//! fixture, lowers via X86_64Profile, assembles via cc -shared -fPIC,
//! dlopens the .so, calls the FFI symbol, and asserts numerical
//! agreement against a Rust-computed reference.

mod common;

use common::{reference_bias_add, reference_matmul, reference_relu};

// ─── Reference implementations ────────────────────────────────────────────────

fn reference_softmax_stable(input: &[f32], b: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; b * k];
    for i in 0..b {
        let row = &input[i * k..(i + 1) * k];
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for kk in 0..k {
            let e = common::exp_ref(row[kk] - max);
            out[i * k + kk] = e;
            sum += e;
        }
        for kk in 0..k {
            out[i * k + kk] /= sum;
        }
    }
    out
}

fn reference_linear_relu(input: &[f32; 32], params: &[f32; 8]) -> [f32; 16] {
    const B: usize = 8;
    const K: usize = 4;
    const N: usize = 2;
    let mut out = [0.0f32; 16];
    for i in 0..B {
        for j in 0..N {
            let mut sum = 0.0f32;
            for k in 0..K {
                sum = f32::mul_add(input[i * K + k], params[k * N + j], sum);
            }
            out[i * N + j] = sum.max(0.0);
        }
    }
    out
}

/// Architecture-matched x86_64 reference for SelfAttention. Uses
/// separate `mul + add` (no FMA) to match emit_matmul x86_64's
/// deliberate non-FMA design from M9 — intentional divergence from
/// the arm64 reference, not a defect. `common::exp_ref` matches the
/// M17 inline polynomial exp (replaces `f32::exp` / libm expf).
fn reference_self_attention_x86_64(
    x: &[f32],
    batch: usize,
    heads: usize,
    seq: usize,
    head_dim: usize,
) -> Vec<f32> {
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let head_stride = seq * head_dim;
    let head_count = batch * heads;
    let mut out = vec![0.0f32; head_count * head_stride];

    let mut scores = vec![0.0f32; seq * seq];
    let mut attn = vec![0.0f32; seq * seq];

    for head in 0..head_count {
        let x_head = &x[head * head_stride..(head + 1) * head_stride];
        let out_head = &mut out[head * head_stride..(head + 1) * head_stride];

        for i in 0..seq {
            for j in 0..seq {
                let mut acc = 0.0f32;
                for k in 0..head_dim {
                    let prod = x_head[i * head_dim + k] * x_head[j * head_dim + k];
                    acc += prod; // separate mul + add (NOT mul_add)
                }
                scores[i * seq + j] = acc * scale;
            }
        }

        // attn = softmax(scores, last_axis) — M17: use exp_ref to match inline polynomial
        for i in 0..seq {
            let row = &scores[i * seq..(i + 1) * seq];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for j in 0..seq {
                let e = common::exp_ref(row[j] - max);
                attn[i * seq + j] = e;
                sum += e;
            }
            for j in 0..seq {
                attn[i * seq + j] /= sum;
            }
        }

        for i in 0..seq {
            for k in 0..head_dim {
                let mut acc = 0.0f32;
                for j in 0..seq {
                    let prod = attn[i * seq + j] * x_head[j * head_dim + k];
                    acc += prod;
                }
                out_head[i * head_dim + k] = acc;
            }
        }
    }

    out
}

fn deterministic_input(total: usize) -> Vec<f32> {
    (0..total).map(|i| (i as f32).sin() * 0.1).collect()
}

// ─── Reference-validation unit tests (always run on linux-x86_64 CI) ─────────

#[test]
fn reference_softmax_stable_known_values() {
    let input = [1.0f32, 2.0, 3.0];
    let output = reference_softmax_stable(&input, 1, 3);
    assert!((output[0] - 0.0900).abs() < 1e-4, "got {}", output[0]);
    assert!((output[1] - 0.2447).abs() < 1e-4, "got {}", output[1]);
    assert!((output[2] - 0.6652).abs() < 1e-4, "got {}", output[2]);
}

#[test]
fn reference_bias_add_known_values() {
    let acc = [1.0f32, 2.0, 3.0];
    let bias = [0.5f32, -1.0, 2.5];
    let out = reference_bias_add(&acc, &bias, 3);
    assert_eq!(out, vec![1.5, 1.0, 5.5]);
}

// ---------------------------------------------------------------------------
// M4a fixture (linear → relu, no softmax)
// ---------------------------------------------------------------------------

#[test]
fn m4a_no_softmax_still_runs() {
    if !common::cc_available() {
        eprintln!("skip: integration test requires `cc` on PATH");
        return;
    }

    let src = std::fs::read_to_string("../../tests/fixtures/m4_linear_relu.nfl")
        .expect("fixture readable");
    let ast = compiler::parse(&src).expect("parse");
    let uir = compiler::ir::build(&ast).expect("ir::build");
    // M5a: exercise the default (fused) path — same as `nflc compile`.
    let uir = compiler::passes::run_pipeline(&uir, &compiler::passes::default_pipeline())
        .expect("pipeline ok");

    let asm = profiles_x86_64::lower(&uir).expect("lower");
    assert_eq!(asm.functions.len(), 1);
    let sig = &asm.functions[0];
    assert_eq!(sig.name, "nfl_forward_M4Demo");
    assert_eq!(sig.inputs_floats, vec![32]);
    assert_eq!(sig.params_floats, 8);
    assert_eq!(sig.output_floats, 16);

    // Verify the params_layout is what we expect for M4a fixture (single LinearWeight slot).
    assert_eq!(sig.params_layout.len(), 1);
    let slot = &sig.params_layout[0];
    assert_eq!(slot.kind, profiles_x86_64::ParamKind::LinearWeight);
    assert_eq!(slot.offset, 0);
    assert_eq!(slot.size, 8);

    let so_path = common::compile_to_so(&asm.source, "m4a_linear_relu");

    let lib = unsafe { libloading::Library::new(&so_path) }.expect("open");
    let forward: libloading::Symbol<unsafe extern "C" fn(*const f32, *const f32, *mut f32)> =
        unsafe { lib.get(b"nfl_forward_M4Demo") }.expect("dlsym");

    let mut input = [0.0f32; 32];
    for (i, v) in input.iter_mut().enumerate() {
        *v = (i as f32) * 0.1 - 1.5;
    }
    let mut params = [0.0f32; 8];
    for (i, v) in params.iter_mut().enumerate() {
        *v = ((i as f32) - 4.0) * 0.25;
    }
    let mut output = [0.0f32; 16];
    unsafe {
        forward(input.as_ptr(), params.as_ptr(), output.as_mut_ptr());
    }

    let expected = reference_linear_relu(&input, &params);
    for (i, (a, b)) in output.iter().zip(expected.iter()).enumerate() {
        assert!((a - b).abs() < 1e-5, "output[{i}]: got {a}, expected {b}");
    }
}

// ---------------------------------------------------------------------------
// M4b fixtures
// ---------------------------------------------------------------------------

#[test]
fn tinymlp_full_with_softmax_runs_correctly() {
    if !common::cc_available() {
        eprintln!("skip: requires cc");
        return;
    }

    let src = std::fs::read_to_string("../../tests/fixtures/tiny_mlp.nfl").unwrap();
    let ast = compiler::parse(&src).unwrap();
    let uir = compiler::ir::build(&ast).unwrap();
    // M5a: exercise the default (fused) path — same as `nflc compile`.
    let uir = compiler::passes::run_pipeline(&uir, &compiler::passes::default_pipeline())
        .expect("pipeline ok");
    let asm = profiles_x86_64::lower(&uir).expect("lower");
    let so_path = common::compile_to_so(&asm.source, "tinymlp_softmax");

    let lib = unsafe { libloading::Library::new(&so_path) }.unwrap();
    let forward: libloading::Symbol<unsafe extern "C" fn(*const f32, *const f32, *mut f32)> =
        unsafe { lib.get(b"nfl_forward_TinyMLP") }.unwrap();

    // batch=8, input=4, output=2
    let mut input = [0.0f32; 32]; // batch=8, in=4
    for (i, v) in input.iter_mut().enumerate() {
        *v = (i as f32) * 0.1 - 1.5;
    }
    let mut params = [0.0f32; 8]; // 4*2
    for (i, v) in params.iter_mut().enumerate() {
        *v = ((i as f32) - 4.0) * 0.25;
    }
    let mut output = [0.0f32; 16]; // 8*2
    unsafe {
        forward(input.as_ptr(), params.as_ptr(), output.as_mut_ptr());
    }

    let intermediate = reference_matmul(&input, &params, 8, 4, 2);
    let expected = reference_softmax_stable(&intermediate, 8, 2);
    for (i, (a, b)) in output.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-4,
            "tinymlp[{i}]: asm got {a}, ref got {b}"
        );
    }
}

#[test]
fn mixed_args_runs_correctly() {
    if !common::cc_available() {
        eprintln!("skip: requires cc");
        return;
    }

    let src = std::fs::read_to_string("../../tests/fixtures/mixed_args.nfl").unwrap();
    let ast = compiler::parse(&src).unwrap();
    let uir = compiler::ir::build(&ast).unwrap();
    // M5a: exercise the default (fused) path — same as `nflc compile`.
    let uir = compiler::passes::run_pipeline(&uir, &compiler::passes::default_pipeline())
        .expect("pipeline ok");
    let asm = profiles_x86_64::lower(&uir).expect("lower");

    // Confirm layout: linear[16, bias=true] + linear[output=2] (no bias) + softmax.
    // params: weight(8*16=128) + bias(16) + weight(16*2=32) = 176 floats.
    let sig = &asm.functions[0];
    assert_eq!(sig.name, "nfl_forward_MixedArgs");
    assert_eq!(sig.params_floats, 8 * 16 + 16 + 16 * 2);

    let so_path = common::compile_to_so(&asm.source, "mixed_args");
    let lib = unsafe { libloading::Library::new(&so_path) }.unwrap();
    let forward: libloading::Symbol<unsafe extern "C" fn(*const f32, *const f32, *mut f32)> =
        unsafe { lib.get(b"nfl_forward_MixedArgs") }.unwrap();

    // batch=4, input=8, output=2
    let mut input = vec![0.0f32; 4 * 8];
    for (i, v) in input.iter_mut().enumerate() {
        *v = (i as f32) * 0.05 - 0.8;
    }
    let mut params = vec![0.0f32; sig.params_floats];
    for (i, v) in params.iter_mut().enumerate() {
        *v = ((i as f32) - 50.0) * 0.01;
    }
    let mut output = vec![0.0f32; 4 * 2];
    unsafe {
        forward(input.as_ptr(), params.as_ptr(), output.as_mut_ptr());
    }

    // Reference: matmul → bias → relu → matmul → softmax
    let weights1 = &params[0..128];
    let bias1 = &params[128..144];
    let weights2 = &params[144..176];

    let mm1 = reference_matmul(&input, weights1, 4, 8, 16);
    let mm1_b = reference_bias_add(&mm1, bias1, 16);
    let r1 = reference_relu(&mm1_b);
    let mm2 = reference_matmul(&r1, weights2, 4, 16, 2);
    let expected = reference_softmax_stable(&mm2, 4, 2);

    for (i, (a, b)) in output.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-3,
            "mixed_args[{i}]: asm got {a}, ref got {b}"
        );
    }
}

#[test]
fn classifier_runs_correctly() {
    if !common::cc_available() {
        eprintln!("skip: requires cc");
        return;
    }

    let src = std::fs::read_to_string("../../tests/fixtures/classifier.nfl").unwrap();
    let ast = compiler::parse(&src).unwrap();
    let uir = compiler::ir::build(&ast).unwrap();
    // M5a: exercise the default (fused) path — same as `nflc compile`.
    let uir = compiler::passes::run_pipeline(&uir, &compiler::passes::default_pipeline())
        .expect("pipeline ok");
    let asm = profiles_x86_64::lower(&uir).expect("lower");

    let sig = &asm.functions[0];
    assert_eq!(sig.name, "nfl_forward_Classifier");
    // batch=32, input=784, hidden=512+256, output=10.
    // Linears: 784*512 + 512*256 + 256*10 = 401408 + 131072 + 2560 = 535040
    assert_eq!(sig.params_floats, 535040);

    let so_path = common::compile_to_so(&asm.source, "classifier");
    let lib = unsafe { libloading::Library::new(&so_path) }.unwrap();
    let forward: libloading::Symbol<unsafe extern "C" fn(*const f32, *const f32, *mut f32)> =
        unsafe { lib.get(b"nfl_forward_Classifier") }.unwrap();

    // Use small deterministic values to avoid NaN from huge accumulators.
    let mut input = vec![0.0f32; 32 * 784];
    for (i, v) in input.iter_mut().enumerate() {
        *v = ((i as f32) % 100.0) * 0.001;
    }
    let mut params = vec![0.0f32; sig.params_floats];
    for (i, v) in params.iter_mut().enumerate() {
        *v = (((i as f32) % 1000.0) - 500.0) * 0.0001;
    }
    let mut output = vec![0.0f32; 32 * 10];
    unsafe {
        forward(input.as_ptr(), params.as_ptr(), output.as_mut_ptr());
    }

    // Sanity: each row of output sums to ~1 (softmax property).
    for i in 0..32 {
        let row_sum: f32 = output[i * 10..(i + 1) * 10].iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-3,
            "classifier row {i} sum = {row_sum}, expected ~1.0"
        );
    }

    // Sanity: all output elements in [0, 1].
    for (i, v) in output.iter().enumerate() {
        assert!(
            *v >= 0.0 && *v <= 1.0,
            "classifier[{i}] = {v} not in [0, 1]"
        );
    }
}

#[test]
fn pipeline_styles_runs_correctly() {
    if !common::cc_available() {
        eprintln!("skip: requires cc");
        return;
    }

    let src = std::fs::read_to_string("../../tests/fixtures/pipeline_styles.nfl").unwrap();
    let ast = compiler::parse(&src).unwrap();
    let uir = compiler::ir::build(&ast).unwrap();
    // M5a: exercise the default (fused) path — same as `nflc compile`.
    let uir = compiler::passes::run_pipeline(&uir, &compiler::passes::default_pipeline())
        .expect("pipeline ok");
    let asm = profiles_x86_64::lower(&uir).expect("lower");

    // Three models with same signature shape.
    assert_eq!(asm.functions.len(), 3);
    let names: Vec<&str> = asm.functions.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(
        names,
        vec![
            "nfl_forward_SingleLine",
            "nfl_forward_PerStepWrap",
            "nfl_forward_MixedWrap"
        ]
    );

    let so_path = common::compile_to_so(&asm.source, "pipeline_styles");
    let lib = unsafe { libloading::Library::new(&so_path) }.unwrap();

    // Each model: batch=4, input=10, linear[8] -> relu -> linear[output=2] -> softmax
    // params = 10*8 + 8*2 = 96 floats.
    let mut input = [0.0f32; 4 * 10];
    for (i, v) in input.iter_mut().enumerate() {
        *v = (i as f32) * 0.05 - 0.5;
    }
    let mut params = [0.0f32; 96];
    for (i, v) in params.iter_mut().enumerate() {
        *v = ((i as f32) - 48.0) * 0.01;
    }

    for name in &names {
        let sym_bytes = format!("{}\0", name).into_bytes();
        let forward: libloading::Symbol<unsafe extern "C" fn(*const f32, *const f32, *mut f32)> =
            unsafe { lib.get(&sym_bytes) }.unwrap();
        let mut output = vec![0.0f32; 4 * 2];
        unsafe {
            forward(input.as_ptr(), params.as_ptr(), output.as_mut_ptr());
        }

        // Sanity: rows sum to ~1.
        for i in 0..4 {
            let row_sum: f32 = output[i * 2..(i + 1) * 2].iter().sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-3,
                "pipeline {name} row {i} sum = {row_sum}"
            );
        }
    }
}

#[test]
fn comments_runs_correctly() {
    if !common::cc_available() {
        eprintln!("skip: requires cc");
        return;
    }

    let src = std::fs::read_to_string("../../tests/fixtures/comments.nfl").unwrap();
    let ast = compiler::parse(&src).unwrap();
    let uir = compiler::ir::build(&ast).unwrap();
    // M5a: exercise the default (fused) path — same as `nflc compile`.
    let uir = compiler::passes::run_pipeline(&uir, &compiler::passes::default_pipeline())
        .expect("pipeline ok");
    let asm = profiles_x86_64::lower(&uir).expect("lower");

    let sig = &asm.functions[0];
    assert_eq!(sig.name, "nfl_forward_Commented");

    let so_path = common::compile_to_so(&asm.source, "comments");
    let lib = unsafe { libloading::Library::new(&so_path) }.unwrap();
    let forward: libloading::Symbol<unsafe extern "C" fn(*const f32, *const f32, *mut f32)> =
        unsafe { lib.get(b"nfl_forward_Commented") }.unwrap();

    let mut input = vec![0.0f32; sig.inputs_floats[0]];
    for (i, v) in input.iter_mut().enumerate() {
        *v = (i as f32) * 0.1 - 1.0;
    }
    let mut params = vec![0.0f32; sig.params_floats];
    for (i, v) in params.iter_mut().enumerate() {
        *v = ((i as f32) - 10.0) * 0.05;
    }
    let mut output = vec![0.0f32; sig.output_floats];
    unsafe {
        forward(input.as_ptr(), params.as_ptr(), output.as_mut_ptr());
    }

    // Final op is softmax → rows sum to ~1.
    let last_dim = sig.output_floats / 4; // batch=4 (per fixture header)
    for i in 0..4 {
        let row_sum: f32 = output[i * last_dim..(i + 1) * last_dim].iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-3,
            "comments row {i} sum = {row_sum}"
        );
    }
}

#[test]
fn fused_vs_unfused_classifier_match_numerically() {
    if !common::cc_available() {
        eprintln!("skip: requires cc");
        return;
    }

    let src = std::fs::read_to_string("../../tests/fixtures/classifier.nfl").unwrap();
    let ast = compiler::parse(&src).unwrap();
    let uir = compiler::ir::build(&ast).unwrap();

    // Build BOTH paths.
    let fused_uir = compiler::passes::run_pipeline(&uir, &compiler::passes::default_pipeline())
        .expect("pipeline ok");
    let fused_asm = profiles_x86_64::lower(&fused_uir).expect("fused lower");
    let unfused_asm = profiles_x86_64::lower(&uir).expect("unfused lower");

    // Asm shape differs as expected.
    assert!(
        fused_asm.source.contains("maxss   %xmm4, %xmm0"),
        "fused asm missing inline maxss"
    );
    assert!(
        !fused_asm.source.contains(".Lrelu_"),
        "fused asm should NOT have separate relu loops"
    );
    assert!(
        unfused_asm.source.contains(".Lrelu_0_0:"),
        "unfused asm missing first relu loop label"
    );
    assert!(
        unfused_asm.source.contains(".Lrelu_0_1:"),
        "unfused asm missing second relu loop label (classifier has 2 relus)"
    );

    // Compile both, run both with same input/params, compare numerically.
    let fused_so = common::compile_to_so(&fused_asm.source, "fused_classifier");
    let unfused_so = common::compile_to_so(&unfused_asm.source, "unfused_classifier");

    let fused_lib = unsafe { libloading::Library::new(&fused_so).unwrap() };
    let unfused_lib = unsafe { libloading::Library::new(&unfused_so).unwrap() };

    let fused_forward: libloading::Symbol<unsafe extern "C" fn(*const f32, *const f32, *mut f32)> =
        unsafe { fused_lib.get(b"nfl_forward_Classifier") }.unwrap();
    let unfused_forward: libloading::Symbol<
        unsafe extern "C" fn(*const f32, *const f32, *mut f32),
    > = unsafe { unfused_lib.get(b"nfl_forward_Classifier") }.unwrap();

    // Same deterministic input + params as classifier_runs_correctly test.
    // Both asms describe the same model; pull params length from the FnSig
    // instead of hardcoding so the test follows the fixture if it changes.
    let params_len = fused_asm.functions[0].params_floats;
    assert_eq!(
        params_len, unfused_asm.functions[0].params_floats,
        "fused/unfused params_floats disagree — pipeline changed param layout"
    );
    let mut input = vec![0.0f32; 32 * 784];
    for (i, v) in input.iter_mut().enumerate() {
        *v = ((i as f32) % 100.0) * 0.001;
    }
    let mut params = vec![0.0f32; params_len];
    for (i, v) in params.iter_mut().enumerate() {
        *v = (((i as f32) % 1000.0) - 500.0) * 0.0001;
    }

    let mut fused_out = vec![0.0f32; 32 * 10];
    let mut unfused_out = vec![0.0f32; 32 * 10];

    unsafe {
        fused_forward(input.as_ptr(), params.as_ptr(), fused_out.as_mut_ptr());
        unfused_forward(input.as_ptr(), params.as_ptr(), unfused_out.as_mut_ptr());
    }

    // assert_eq! exact equality: f32 store+load is bit-preserving;
    // fusion only relocates WHERE relu is applied, not WHICH floats compute.
    for (i, (a, b)) in fused_out.iter().zip(unfused_out.iter()).enumerate() {
        assert_eq!(
            *a, *b,
            "fused[{i}]={a} unfused[{i}]={b} — fusion changed numerics"
        );
    }
}

#[test]
fn fused_vs_unfused_softmax_match_numerically() {
    if !common::cc_available() {
        eprintln!("skip: requires cc");
        return;
    }

    // Cover BOTH no-bias and bias-aware fused-softmax paths.
    for (fixture_path, fn_name, batch, input_dim, output_dim) in [
        (
            "../../tests/fixtures/classifier.nfl",
            "nfl_forward_Classifier",
            32_usize,
            784_usize,
            10_usize,
        ),
        (
            "../../tests/fixtures/softmax_with_bias.nfl",
            "nfl_forward_SoftmaxWithBias",
            4_usize,
            8_usize,
            3_usize,
        ),
    ] {
        let src =
            std::fs::read_to_string(fixture_path).unwrap_or_else(|e| panic!("{fixture_path}: {e}"));
        let ast = compiler::parse(&src).unwrap();
        let uir = compiler::ir::build(&ast).unwrap();

        let fused_uir = compiler::passes::run_pipeline(&uir, &compiler::passes::default_pipeline())
            .expect("pipeline ok");
        let fused_asm = profiles_x86_64::lower(&fused_uir).expect("fused lower");
        let unfused_asm = profiles_x86_64::lower(&uir).expect("unfused lower");

        // Asm structural validation using ACTUAL label prefixes:
        // fused RowWise tail uses .Lfsmx_*, standalone emit_softmax uses .Lsm_*.
        // Asm structural validation. Note: classifier.nfl is also
        // covered by M5a's fused_vs_unfused_classifier_match_numerically,
        // which pins relu fusion (maxss %xmm4, %xmm0 + .Lrelu_). This test
        // pins softmax fusion (.Lfsmx_ + inline exp M17) — complementary, not
        // redundant.
        assert!(
            !fused_asm.source.contains("call    expf@PLT"),
            "{fixture_path}: fused asm must not call expf@PLT after M17 inlining"
        );
        assert!(
            fused_asm.source.contains(".Lexp_c7(%rip)"),
            "{fixture_path}: fused asm missing .Lexp_c7(%rip) (inline exp range reduction)"
        );
        assert!(
            !fused_asm.source.contains(".Lsm_"),
            "{fixture_path}: fused asm should NOT have standalone softmax loop labels (.Lsm_)"
        );
        assert!(
            fused_asm.source.contains(".Lfsmx_"),
            "{fixture_path}: fused asm missing row-wise softmax tail labels (.Lfsmx_)"
        );
        assert!(
            unfused_asm.source.contains(".Lsm_"),
            "{fixture_path}: unfused asm should have standalone softmax loop labels (.Lsm_)"
        );

        let label = fn_name.trim_start_matches("nfl_forward_");
        let fused_so = common::compile_to_so(&fused_asm.source, &format!("fused_{label}"));
        let unfused_so = common::compile_to_so(&unfused_asm.source, &format!("unfused_{label}"));

        let fused_lib = unsafe { libloading::Library::new(&fused_so).unwrap() };
        let unfused_lib = unsafe { libloading::Library::new(&unfused_so).unwrap() };

        // Dynamic symbol lookup — match M5b's idiomatic style.
        let sym_bytes = format!("{fn_name}\0").into_bytes();
        let fused_forward: libloading::Symbol<
            unsafe extern "C" fn(*const f32, *const f32, *mut f32),
        > = unsafe { fused_lib.get(&sym_bytes).unwrap() };
        let unfused_forward: libloading::Symbol<
            unsafe extern "C" fn(*const f32, *const f32, *mut f32),
        > = unsafe { unfused_lib.get(&sym_bytes).unwrap() };

        let params_len = fused_asm.functions[0].params_floats;
        assert_eq!(
            params_len, unfused_asm.functions[0].params_floats,
            "{fixture_path}: fused/unfused param layout mismatch"
        );

        let input_floats = fused_asm.functions[0].inputs_floats[0];
        let output_floats = fused_asm.functions[0].output_floats;
        assert_eq!(input_floats, batch * input_dim,
            "{fixture_path}: FnSig.inputs_floats[0]={input_floats} disagrees with hardcoded batch*input_dim={}",
            batch * input_dim);
        assert_eq!(output_floats, batch * output_dim,
            "{fixture_path}: FnSig.output_floats={output_floats} disagrees with hardcoded batch*output_dim={}",
            batch * output_dim);
        assert_eq!(
            input_floats, unfused_asm.functions[0].inputs_floats[0],
            "{fixture_path}: fused/unfused inputs_floats[0] mismatch"
        );
        assert_eq!(
            output_floats, unfused_asm.functions[0].output_floats,
            "{fixture_path}: fused/unfused output_floats mismatch"
        );

        let mut input = vec![0.0f32; input_floats];
        for (i, v) in input.iter_mut().enumerate() {
            *v = ((i as f32) % 100.0) * 0.001;
        }
        let mut params = vec![0.0f32; params_len];
        for (i, v) in params.iter_mut().enumerate() {
            *v = (((i as f32) % 1000.0) - 500.0) * 0.0001;
        }

        let mut fused_out = vec![0.0f32; output_floats];
        let mut unfused_out = vec![0.0f32; output_floats];

        unsafe {
            fused_forward(input.as_ptr(), params.as_ptr(), fused_out.as_mut_ptr());
            unfused_forward(input.as_ptr(), params.as_ptr(), unfused_out.as_mut_ptr());
        }

        for (i, (a, b)) in fused_out.iter().zip(unfused_out.iter()).enumerate() {
            assert_eq!(
                *a, *b,
                "{fixture_path}: fused[{i}]={a} unfused[{i}]={b} — fusion changed numerics"
            );
        }
    }
}

#[test]
fn fused_vs_unfused_mixed_args_match_numerically() {
    if !common::cc_available() {
        eprintln!("skip: requires cc");
        return;
    }

    let src = std::fs::read_to_string("../../tests/fixtures/mixed_args.nfl").unwrap();
    let ast = compiler::parse(&src).unwrap();
    let uir = compiler::ir::build(&ast).unwrap();

    // Build BOTH paths.
    let fused_uir = compiler::passes::run_pipeline(&uir, &compiler::passes::default_pipeline())
        .expect("pipeline ok");
    let fused_asm = profiles_x86_64::lower(&fused_uir).expect("fused lower");
    let unfused_asm = profiles_x86_64::lower(&uir).expect("unfused lower");

    // Asm shape pre-asserts.
    assert!(
        fused_asm.source.contains("maxss   %xmm4, %xmm0"),
        "fused asm missing inline maxss"
    );
    // mixed_args.nfl has `linear[16, bias=true] → relu` as the first
    // fusion candidate; the fused asm must still contain bias-add
    // (addss %xmm5, %xmm0) immediately before the maxss (within one
    // emit_linear, not in a separate function).
    assert!(
        fused_asm.source.contains("addss   %xmm5, %xmm0"),
        "fused asm missing bias-add (addss %xmm5, %xmm0):\n{}",
        fused_asm.source
    );
    assert!(
        !fused_asm.source.contains(".Lrelu_"),
        "fused asm should NOT have separate relu loops"
    );
    assert!(
        unfused_asm.source.contains(".Lrelu_0_0:"),
        "unfused asm missing relu loop label"
    );

    // Compile both, run both with same input/params, compare numerically.
    let fused_so = common::compile_to_so(&fused_asm.source, "fused_mixed_args");
    let unfused_so = common::compile_to_so(&unfused_asm.source, "unfused_mixed_args");

    let fused_lib = unsafe { libloading::Library::new(&fused_so).unwrap() };
    let unfused_lib = unsafe { libloading::Library::new(&unfused_so).unwrap() };

    let fused_forward: libloading::Symbol<unsafe extern "C" fn(*const f32, *const f32, *mut f32)> =
        unsafe { fused_lib.get(b"nfl_forward_MixedArgs") }.unwrap();
    let unfused_forward: libloading::Symbol<
        unsafe extern "C" fn(*const f32, *const f32, *mut f32),
    > = unsafe { unfused_lib.get(b"nfl_forward_MixedArgs") }.unwrap();

    // Same deterministic input + params formula as the classifier
    // integration test. mixed_args has batch=4, input=8, output=2.
    let params_len = fused_asm.functions[0].params_floats;
    assert_eq!(
        params_len, unfused_asm.functions[0].params_floats,
        "fused/unfused params_floats disagree — pipeline changed param layout"
    );

    let mut input = [0.0f32; 4 * 8];
    for (i, v) in input.iter_mut().enumerate() {
        *v = ((i as f32) % 100.0) * 0.001;
    }
    let mut params = vec![0.0f32; params_len];
    for (i, v) in params.iter_mut().enumerate() {
        *v = (((i as f32) % 1000.0) - 500.0) * 0.0001;
    }

    let mut fused_out = [0.0f32; 4 * 2];
    let mut unfused_out = [0.0f32; 4 * 2];

    unsafe {
        fused_forward(input.as_ptr(), params.as_ptr(), fused_out.as_mut_ptr());
        unfused_forward(input.as_ptr(), params.as_ptr(), unfused_out.as_mut_ptr());
    }

    // assert_eq! exact equality: f32 store+load is bit-preserving;
    // bias-aware fusion (matmul → bias-add → maxss → store) only
    // relocates WHERE relu/bias is applied, not WHICH floats compute.
    for (i, (a, b)) in fused_out.iter().zip(unfused_out.iter()).enumerate() {
        assert_eq!(
            *a, *b,
            "fused[{i}]={a} unfused[{i}]={b} — bias-aware fusion changed numerics"
        );
    }
}

// ---------------------------------------------------------------------------
// M8 fixture: dropout-only model (dropout IS model.output).
// Triggers the BufferLoc::OutputReg branch in walk_model::Dropout.
// ---------------------------------------------------------------------------

#[test]
fn dropout_only_b2_k4_no_passes() {
    if !common::cc_available() {
        eprintln!("skip: cc not available");
        return;
    }

    let src =
        std::fs::read_to_string("../../tests/fixtures/dropout_only.nfl").expect("read fixture");
    let ast = compiler::parse(&src).expect("parse");
    let uir = compiler::ir::build(&ast).expect("ir::build");
    // No run_pipeline — exercise raw UIR (mirror of `--no-passes`).
    let asm = profiles_x86_64::lower(&uir).expect("lower");

    let so_path = common::compile_to_so(&asm.source, "dropout_only_b2_k4");
    let lib = unsafe { libloading::Library::new(&so_path) }.expect("open");

    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let params: [f32; 0] = [];
    let mut output = [0.0f32; 8];

    unsafe {
        let forward: libloading::Symbol<unsafe extern "C" fn(*const f32, *const f32, *mut f32)> =
            lib.get(b"nfl_forward_OnlyDropout\0")
                .expect("symbol not found");
        forward(input.as_ptr(), params.as_ptr(), output.as_mut_ptr());
    }

    assert_eq!(
        output, input,
        "dropout-as-output must copy input verbatim; got {:?}",
        output
    );
}

#[test]
fn dropout_only_b1_k8_no_passes() {
    if !common::cc_available() {
        eprintln!("skip: cc not available");
        return;
    }

    // Same total floats (8) as b=2,k=4 but b=1 — closes single-row
    // coverage gap noted in the M8 audit.
    let nfl_src =
        "model OnlyDropout1 [b=1, k=8]:\n    x: Tensor[b, k]\n    x -> dropout[rate=0.1]\n";
    let ast = compiler::parse(nfl_src).expect("parse");
    let uir = compiler::ir::build(&ast).expect("ir::build");
    let asm = profiles_x86_64::lower(&uir).expect("lower");

    let so_path = common::compile_to_so(&asm.source, "dropout_only_b1_k8");
    let lib = unsafe { libloading::Library::new(&so_path) }.expect("open");

    let input = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
    let params: [f32; 0] = [];
    let mut output = [0.0f32; 8];

    unsafe {
        let forward: libloading::Symbol<unsafe extern "C" fn(*const f32, *const f32, *mut f32)> =
            lib.get(b"nfl_forward_OnlyDropout1\0")
                .expect("symbol not found");
        forward(input.as_ptr(), params.as_ptr(), output.as_mut_ptr());
    }

    assert_eq!(output, input, "b=1 dropout-as-output must copy verbatim");
}

// ---------------------------------------------------------------------------
// M8 fixture: dim > 4095 along k-axis.
// ---------------------------------------------------------------------------

#[test]
fn large_classifier_k_8192() {
    if !common::cc_available() {
        eprintln!("skip: cc not available");
        return;
    }

    const B: usize = 2;
    const K: usize = 8192;
    const N: usize = 10;

    let src = std::fs::read_to_string("../../tests/fixtures/large_classifier_k.nfl")
        .expect("read fixture");
    let ast = compiler::parse(&src).expect("parse");
    let uir_pre = compiler::ir::build(&ast).expect("ir::build");
    let uir = compiler::passes::run_pipeline(&uir_pre, &compiler::passes::default_pipeline())
        .expect("pipeline");
    let asm = profiles_x86_64::lower(&uir).expect("lower");

    let so_path = common::compile_to_so(&asm.source, "large_classifier_k");
    let lib = unsafe { libloading::Library::new(&so_path) }.expect("open");

    // Deterministic input: x[i, j] = (i * K + j) as f32 / 10000.0
    let input: Vec<f32> = (0..B * K).map(|i| (i as f32) / 10000.0).collect();
    // Deterministic weights: w[k, n] = ((k + n) % 7) as f32 / 100.0
    let weights: Vec<f32> = (0..K * N)
        .map(|i| {
            let kk = i / N;
            let nn = i % N;
            (((kk + nn) % 7) as f32) / 100.0
        })
        .collect();
    let mut output = vec![0.0f32; B * N];

    unsafe {
        let forward: libloading::Symbol<unsafe extern "C" fn(*const f32, *const f32, *mut f32)> =
            lib.get(b"nfl_forward_LargeK\0").expect("symbol not found");
        forward(input.as_ptr(), weights.as_ptr(), output.as_mut_ptr());
    }

    let matmul = reference_matmul(&input, &weights, B, K, N);
    let expected = reference_softmax_stable(&matmul, B, N);

    for (i, (got, want)) in output.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-3,
            "k=8192 output[{i}] = {got}, expected {want}"
        );
    }
}

#[test]
fn large_classifier_n_5120() {
    if !common::cc_available() {
        eprintln!("skip: cc not available");
        return;
    }

    const B: usize = 2;
    const K: usize = 8;
    const N: usize = 5120;

    let src = std::fs::read_to_string("../../tests/fixtures/large_classifier_n.nfl")
        .expect("read fixture");
    let ast = compiler::parse(&src).expect("parse");
    let uir_pre = compiler::ir::build(&ast).expect("ir::build");
    let uir = compiler::passes::run_pipeline(&uir_pre, &compiler::passes::default_pipeline())
        .expect("pipeline");
    let asm = profiles_x86_64::lower(&uir).expect("lower");

    let so_path = common::compile_to_so(&asm.source, "large_classifier_n");
    let lib = unsafe { libloading::Library::new(&so_path) }.expect("open");

    let input: Vec<f32> = (0..B * K).map(|i| (i as f32) / 10.0).collect();
    let weights: Vec<f32> = (0..K * N)
        .map(|i| {
            let kk = i / N;
            let nn = i % N;
            (((kk + nn) % 5) as f32) / 100.0
        })
        .collect();
    let mut output = vec![0.0f32; B * N];

    unsafe {
        let forward: libloading::Symbol<unsafe extern "C" fn(*const f32, *const f32, *mut f32)> =
            lib.get(b"nfl_forward_LargeN\0").expect("symbol not found");
        forward(input.as_ptr(), weights.as_ptr(), output.as_mut_ptr());
    }

    let matmul = reference_matmul(&input, &weights, B, K, N);
    let expected = reference_softmax_stable(&matmul, B, N);

    for (i, (got, want)) in output.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-3,
            "n=5120 output[{i}] = {got}, expected {want}"
        );
    }
}

// ---------------------------------------------------------------------------
// Task 5.3 — xmm-spill survival test (x86_64-specific)
// ---------------------------------------------------------------------------

#[test]
fn fused_softmax_xmm_spill_x86_64() {
    // x86_64-specific test — direct numerical proof that the xmm-spill
    // strategy (spec §7.4) works. The fixture has row dim > 1, so Phase 3
    // calls expf@PLT multiple times per row; spill correctness manifests.
    if !common::cc_available() {
        eprintln!("skip: requires cc");
        return;
    }

    let src = std::fs::read_to_string("../../tests/fixtures/softmax_with_bias.nfl").unwrap();
    let ast = compiler::parse(&src).unwrap();
    let uir = compiler::ir::build(&ast).unwrap();
    let fused_uir = compiler::passes::run_pipeline(&uir, &compiler::passes::default_pipeline())
        .expect("pipeline");
    let asm = profiles_x86_64::lower(&fused_uir).expect("lower");

    // Asm shape pre-asserts.
    assert!(
        asm.source.contains(".Lfsmx_"),
        "fused softmax tail labels missing"
    );
    assert!(
        !asm.source.contains("call    expf@PLT"),
        "fused softmax tail must not call expf@PLT after M17 inlining"
    );
    assert!(
        asm.source.contains(".Lexp_c7(%rip)"),
        "fused softmax tail missing .Lexp_c7(%rip) (inline exp M17)"
    );
    // Row_max/row_sum still live on the stack across the (now inline) exp loop:
    // the §7.4 stack-slot mechanism is RETAINED in M17 — M18 moves it to xmm
    // registers once the leaf cleanup lands. These spills are the point of this
    // test ("xmm_spill") and must not regress.
    assert!(
        asm.source.contains("movss   %xmm8, (%rsp)"),
        "fused tail must spill row_max to (%rsp) [offset 0; spec §7.4]"
    );
    assert!(
        asm.source.contains("movss   %xmm1, 8(%rsp)"),
        "fused tail must spill row_sum to 8(%rsp) [offset 8; spec §7.4]"
    );

    let so_path = common::compile_to_so(&asm.source, "fused_softmax_xmm_spill");
    let lib = unsafe { libloading::Library::new(&so_path).unwrap() };
    let forward: libloading::Symbol<unsafe extern "C" fn(*const f32, *const f32, *mut f32)> =
        unsafe { lib.get(b"nfl_forward_SoftmaxWithBias") }.unwrap();

    let sig = &asm.functions[0];
    let input_floats = sig.inputs_floats[0];
    let params_len = sig.params_floats;
    let output_floats = sig.output_floats;

    let mut input = vec![0.0f32; input_floats];
    for (i, v) in input.iter_mut().enumerate() {
        *v = ((i as f32) % 100.0) * 0.001;
    }
    let mut params = vec![0.0f32; params_len];
    for (i, v) in params.iter_mut().enumerate() {
        *v = (((i as f32) % 1000.0) - 500.0) * 0.0001;
    }
    let mut output = vec![0.0f32; output_floats];
    unsafe {
        forward(input.as_ptr(), params.as_ptr(), output.as_mut_ptr());
    }

    // softmax_with_bias.nfl: batch=4, output_dim=3 (per arm64 mirror).
    // Each row sums to ~1.
    let n = 3usize;
    for i in 0..(output_floats / n) {
        let row_sum: f32 = output[i * n..(i + 1) * n].iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-3,
            "row {i} sum = {row_sum}, xmm-spill produced bogus normalisation"
        );
        for v in &output[i * n..(i + 1) * n] {
            assert!(
                *v >= 0.0 && *v <= 1.0,
                "row {i}: element {v} outside [0,1] — xmm-spill corrupted exp result"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// M10 acceptance: SelfAttention end-to-end FFI.
//
// Bit-exact assertion against an architecture-matched x86_64 reference.
// x86_64 emit_matmul uses `mulss + addss` (no FMA, two roundings);
// reference uses separate `let prod = a*b; acc = acc + prod;`. M17: both
// asm and reference use the inline polynomial exp (common::exp_ref), so
// softmax outputs are bit-exact.
// ---------------------------------------------------------------------------

#[test]
fn self_attention_ffi_matches_reference() {
    // x86_64 FFI tests are cfg-gated at the module level via
    // #![cfg(all(target_os = "linux", target_arch = "x86_64"))] —
    // see existing module attribute. This test runs only on the
    // Linux x86_64 CI job.
    if !common::cc_available() {
        eprintln!("skip: integration test requires `cc` on PATH");
        return;
    }

    const BATCH: usize = 2;
    const HEADS: usize = 4;
    const SEQ: usize = 16;
    const HEAD_DIM: usize = 16;
    const TOTAL: usize = BATCH * HEADS * SEQ * HEAD_DIM;

    let src = std::fs::read_to_string("../../tests/fixtures/self_attention.nfl")
        .expect("fixture readable");
    let ast = compiler::parse(&src).expect("parse");
    let uir = compiler::ir::build(&ast).expect("ir::build");
    let uir = compiler::passes::run_pipeline(&uir, &compiler::passes::default_pipeline())
        .expect("pipeline ok");

    let asm = profiles_x86_64::lower(&uir).expect("lower");
    let sig = &asm.functions[0];
    assert_eq!(sig.inputs_floats, vec![TOTAL]);
    assert_eq!(sig.output_floats, TOTAL);
    assert_eq!(sig.params_floats, 0);

    let so_path = common::compile_to_so(&asm.source, "self_attention");

    let input = deterministic_input(TOTAL);
    let mut output = vec![0.0f32; TOTAL];
    let params: Vec<f32> = vec![0.0f32; sig.params_floats];

    unsafe {
        let lib = libloading::Library::new(&so_path).expect("dlopen");
        let forward: libloading::Symbol<unsafe extern "C" fn(*const f32, *const f32, *mut f32)> =
            lib.get(sig.name.as_bytes()).expect("dlsym");
        forward(input.as_ptr(), params.as_ptr(), output.as_mut_ptr());
    }

    let reference = reference_self_attention_x86_64(&input, BATCH, HEADS, SEQ, HEAD_DIM);

    assert_eq!(
        output, reference,
        "SelfAttention FFI output must match x86_64 reference bit-exactly"
    );
}

// ---------------------------------------------------------------------------
// M12 acceptance: multi-input fixtures (x86_64 mirror of arm64 D.4/D.5/D.6).
//
// D.7: two_input_matmul — N=2 sanity, bit-exact vs Rust reference matmul.
// D.8: multi_input_attention — N=3 acceptance, exercises post-FFI register
//      survival for v-pointer (%rdx) across `call expf@PLT`.
// D.9: too_many_inputs — N=5 negative, expects LowerError::TooManyInputs.
//
// These tests are gated to x86_64 Linux via the module-level cfg attribute.
// ---------------------------------------------------------------------------

/// Pure-Rust reference attention for x86_64: Q @ K, scale by 0.25,
/// softmax(last-axis), then attn @ V. Uses separate mul + add (no FMA) to
/// match x86_64 emit_matmul's deliberate non-FMA design (mulss + addss).
/// M17: softmax uses common::exp_ref (inline polynomial) to match emitter.
/// Shapes: Q=[B,H,S,D], K=[B,H,D,S] (transposed), V=[B,H,S,D] → out=[B,H,S,D].
fn reference_attention_x86_64(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    b: usize,
    h: usize,
    s: usize,
    d: usize,
) -> Vec<f32> {
    let mut scores = vec![0.0f32; b * h * s * s];
    let mut out = vec![0.0f32; b * h * s * d];

    for bi in 0..b {
        for hi in 0..h {
            // scores[bi,hi,i,j] = sum_kk Q[bi,hi,i,kk] * K[bi,hi,kk,j] * 0.25
            let q_off = (bi * h + hi) * s * d;
            let k_off = (bi * h + hi) * d * s;
            let sc_off = (bi * h + hi) * s * s;
            for i in 0..s {
                for j in 0..s {
                    let mut acc = 0.0f32;
                    for kk in 0..d {
                        // separate mul + add (NOT FMA) to match x86_64 emit_matmul.
                        acc += q[q_off + i * d + kk] * k[k_off + kk * s + j];
                    }
                    scores[sc_off + i * s + j] = acc * 0.25;
                }
            }
            // softmax row-wise (stable, matching 3-pass emit_softmax — M17: use exp_ref).
            for i in 0..s {
                let row = &mut scores[sc_off + i * s..sc_off + (i + 1) * s];
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                for x in row.iter_mut() {
                    *x = common::exp_ref(*x - max);
                }
                let sum: f32 = row.iter().sum();
                for x in row.iter_mut() {
                    *x /= sum;
                }
            }
            // out[bi,hi,i,j] = sum_kk scores[bi,hi,i,kk] * V[bi,hi,kk,j]
            let v_off = (bi * h + hi) * s * d;
            let o_off = (bi * h + hi) * s * d;
            for i in 0..s {
                for j in 0..d {
                    let mut acc = 0.0f32;
                    for kk in 0..s {
                        acc += scores[sc_off + i * s + kk] * v[v_off + kk * d + j];
                    }
                    out[o_off + i * d + j] = acc;
                }
            }
        }
    }
    out
}

#[test]
fn two_input_matmul_match_numerically() {
    if !common::cc_available() {
        eprintln!("skip: integration test requires `cc` on PATH");
        return;
    }

    let src = std::fs::read_to_string("../../tests/fixtures/two_input_matmul.nfl")
        .expect("fixture readable");
    let nfl = compiler::parse(&src).expect("parse");
    let uir = compiler::ir::build(&nfl).expect("ir::build");
    let asm = profiles_x86_64::lower(&uir).expect("lower");

    let sig = &asm.functions[0];
    assert_eq!(sig.inputs_floats.len(), 2, "two_input_matmul has arity 2");
    assert_eq!(sig.inputs_floats[0], 4 * 8, "a is [4,8]=32 floats");
    assert_eq!(sig.inputs_floats[1], 8 * 4, "b is [8,4]=32 floats");

    let so_path = common::compile_to_so(&asm.source, "two_input_matmul");
    let lib = unsafe { libloading::Library::new(&so_path) }.expect("dlopen");

    // SysV ABI: a (%rdi), b (%rsi), params (%rdx, empty), out (%rcx).
    type ForwardFn = unsafe extern "C" fn(*const f32, *const f32, *const f32, *mut f32);
    let forward: libloading::Symbol<ForwardFn> =
        unsafe { lib.get(b"nfl_forward_TwoInputMatmul") }.expect("dlsym");

    let m = 4usize;
    let k = 8usize;
    let n = 4usize;
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.07).collect();
    let params: Vec<f32> = vec![]; // matmul has no params
    let mut out = vec![0.0f32; m * n];

    unsafe {
        forward(a.as_ptr(), b.as_ptr(), params.as_ptr(), out.as_mut_ptr());
    }

    // Reference: a @ b using separate mul + add (matching x86_64 emit_matmul).
    let mut expected = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            expected[i * n + j] = sum;
        }
    }

    for (i, (got, want)) in out.iter().zip(expected.iter()).enumerate() {
        assert!(
            got.to_bits() == want.to_bits(),
            "mismatch at index {i}: got {got} ({:#010x}), want {want} ({:#010x})",
            got.to_bits(),
            want.to_bits()
        );
    }

    drop(lib);
}

#[test]
fn multi_input_attention_match_numerically() {
    if !common::cc_available() {
        eprintln!("skip: integration test requires `cc` on PATH");
        return;
    }

    let src = std::fs::read_to_string("../../tests/fixtures/multi_input_attention.nfl")
        .expect("fixture readable");
    let nfl = compiler::parse(&src).expect("parse");
    let uir = compiler::ir::build(&nfl).expect("ir::build");
    let asm = profiles_x86_64::lower(&uir).expect("lower");

    let sig = &asm.functions[0];
    assert_eq!(
        sig.inputs_floats.len(),
        3,
        "multi_input_attention has arity 3"
    );

    // Shape: batch=2, heads=4, seq=16, head_dim=16.
    const BATCH: usize = 2;
    const HEADS: usize = 4;
    const SEQ: usize = 16;
    const HEAD_DIM: usize = 16;
    const TOTAL: usize = BATCH * HEADS * SEQ * HEAD_DIM; // 2048

    assert_eq!(sig.inputs_floats[0], TOTAL, "q is [2,4,16,16]=2048 floats");
    assert_eq!(sig.inputs_floats[1], TOTAL, "k is [2,4,16,16]=2048 floats");
    assert_eq!(sig.inputs_floats[2], TOTAL, "v is [2,4,16,16]=2048 floats");
    assert_eq!(sig.output_floats, TOTAL);
    assert_eq!(sig.params_floats, 0, "no learnable params");

    let so_path = common::compile_to_so(&asm.source, "multi_input_attention");
    let lib = unsafe { libloading::Library::new(&so_path) }.expect("dlopen");

    // SysV ABI: q (%rdi), k (%rsi), v (%rdx), params (%rcx, empty), out (%r8).
    type ForwardFn = unsafe extern "C" fn(*const f32, *const f32, *const f32, *const f32, *mut f32);
    let forward: libloading::Symbol<ForwardFn> =
        unsafe { lib.get(b"nfl_forward_SelfAttention") }.expect("dlsym");

    let q: Vec<f32> = (0..TOTAL).map(|i| (i as f32) * 1e-3).collect();
    let k: Vec<f32> = (0..TOTAL).map(|i| (i as f32) * 1.5e-3).collect();
    let v: Vec<f32> = (0..TOTAL).map(|i| (i as f32) * 0.7e-3).collect();
    let params: Vec<f32> = vec![];
    let mut out = vec![0.0f32; TOTAL];

    unsafe {
        forward(
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            params.as_ptr(),
            out.as_mut_ptr(),
        );
    }

    let expected = reference_attention_x86_64(&q, &k, &v, BATCH, HEADS, SEQ, HEAD_DIM);

    // Bit-exact: x86_64 mulss+addss in both asm and reference (no FMA).
    for (i, (got, want)) in out.iter().zip(expected.iter()).enumerate() {
        assert!(
            got.to_bits() == want.to_bits(),
            "mismatch at index {i}: got {got} ({:#010x}), want {want} ({:#010x})",
            got.to_bits(),
            want.to_bits()
        );
    }

    drop(lib);
}

#[test]
fn too_many_inputs_returns_too_many_inputs_error() {
    use profile_api::{LowerError, Profile};
    let src = std::fs::read_to_string("../../tests/fixtures/profile-negative/too_many_inputs.nfl")
        .expect("fixture readable");
    let nfl = compiler::parse(&src).expect("parse");
    let uir = compiler::ir::build(&nfl).expect("ir::build");
    match profiles_x86_64::X86_64Profile.lower(&uir) {
        Err(LowerError::TooManyInputs { n, max, .. }) => {
            assert_eq!(n, 5, "expected n=5 inputs");
            assert_eq!(max, 4, "expected max=4");
        }
        Err(other) => panic!("expected TooManyInputs, got {other:?}"),
        Ok(_) => panic!("expected Err, got Ok"),
    }
}

// ---------------------------------------------------------------------------
// M13 Group E: residual_add + four_input_matmul (x86_64).
//
// E.2: residual_add_match_numerically — same fixture as arm64 E.1 but with
//      x86_64-specific reference: matmul uses separate mul+add (NOT FMA) to
//      match x86_64 mulss+addss codegen.
// E.3: four_input_matmul_match_numerically — closes Group A (Task 1)
//      end-to-end on x86_64. Exercises N=4 ABI mapping, matmul (the M12
//      bug surface closed by %rbp j-counter relocation), and emit_add at
//      N=4 in one fixture.
// ---------------------------------------------------------------------------

#[test]
fn residual_add_match_numerically() {
    if !common::cc_available() {
        eprintln!("skip: integration test requires `cc` on PATH");
        return;
    }

    let src =
        std::fs::read_to_string("../../tests/fixtures/residual_add.nfl").expect("fixture readable");
    let nfl = compiler::parse(&src).expect("parse");
    let uir = compiler::ir::build(&nfl).expect("ir::build");
    let asm = profiles_x86_64::lower(&uir).expect("lower");

    let sig = &asm.functions[0];
    assert_eq!(sig.inputs_floats.len(), 2, "residual_add has arity 2");
    assert_eq!(sig.inputs_floats[0], 2 * 4, "x is [2,4]=8 floats");
    assert_eq!(sig.inputs_floats[1], 2 * 4, "skip is [2,4]=8 floats");
    // linear[dim] with no bias=true → params = dim*dim = 16 floats (weights only).
    assert_eq!(
        sig.params_floats,
        4 * 4,
        "linear weights only, 4x4=16 floats"
    );

    let so_path = common::compile_to_so(&asm.source, "residual_add");
    let lib = unsafe { libloading::Library::new(&so_path) }.expect("dlopen");

    // SysV ABI: x (%rdi), skip (%rsi), params (%rdx), out (%rcx).
    type ForwardFn = unsafe extern "C" fn(*const f32, *const f32, *const f32, *mut f32);
    let forward: libloading::Symbol<ForwardFn> =
        unsafe { lib.get(b"nfl_forward_ResidualBlock") }.expect("dlsym");

    let batch = 2usize;
    let dim = 4usize;
    let x: Vec<f32> = (0..batch * dim).map(|i| (i as f32) * 0.1).collect();
    let skip: Vec<f32> = (0..batch * dim).map(|i| (i as f32) * 0.07).collect();
    // Linear weights: dim×dim row-major (no bias).
    let weights: Vec<f32> = (0..dim * dim).map(|i| (i as f32) * 0.05).collect();
    let mut out = vec![0.0f32; batch * dim];

    unsafe {
        forward(
            x.as_ptr(),
            skip.as_ptr(),
            weights.as_ptr(),
            out.as_mut_ptr(),
        );
    }

    // Reference: relu(x @ W) + skip, element-wise.
    // Matmul uses separate mul+add (NOT FMA) to match x86_64 mulss+addss.
    // ReLU uses f32::max(0.0) to match the fused fmax in emit_matmul.
    let mut expected = vec![0.0f32; batch * dim];
    for b in 0..batch {
        for j in 0..dim {
            let mut sum = 0.0f32;
            for kk in 0..dim {
                sum += x[b * dim + kk] * weights[kk * dim + j];
            }
            let relu_out = sum.max(0.0f32);
            expected[b * dim + j] = relu_out + skip[b * dim + j];
        }
    }

    for (i, (got, want)) in out.iter().zip(expected.iter()).enumerate() {
        assert!(
            got.to_bits() == want.to_bits(),
            "mismatch at index {i}: got {got} ({:#010x}), want {want} ({:#010x})",
            got.to_bits(),
            want.to_bits()
        );
    }

    drop(lib);
}

#[test]
fn four_input_matmul_match_numerically() {
    if !common::cc_available() {
        eprintln!("skip: integration test requires `cc` on PATH");
        return;
    }

    let src = std::fs::read_to_string("../../tests/fixtures/four_input_matmul.nfl")
        .expect("fixture readable");
    let nfl = compiler::parse(&src).expect("parse");
    let uir = compiler::ir::build(&nfl).expect("ir::build");
    // Pre-M13 this would fail: matmul at N=4 used %r9 as j-counter (collision
    // with output pointer). M13 Task 1 relocated j-counter to %rbp.
    let asm = profiles_x86_64::lower(&uir).expect("lower (M13 closed N=4 + matmul gap)");

    let sig = &asm.functions[0];
    assert_eq!(sig.inputs_floats.len(), 4, "four_input_matmul has arity 4");
    let m = 4usize;
    let k = 8usize;
    let n = 4usize;
    assert_eq!(sig.inputs_floats[0], m * k, "a is [4,8]=32 floats");
    assert_eq!(sig.inputs_floats[1], k * n, "b is [8,4]=32 floats");
    assert_eq!(sig.inputs_floats[2], m * n, "c is [4,4]=16 floats");
    assert_eq!(sig.inputs_floats[3], m * n, "d is [4,4]=16 floats");
    assert_eq!(
        sig.params_floats, 0,
        "matmul + add have no learnable params"
    );

    let so_path = common::compile_to_so(&asm.source, "four_input_matmul");
    let lib = unsafe { libloading::Library::new(&so_path) }.expect("dlopen");

    // SysV ABI: a (%rdi), b (%rsi), c (%rdx), d (%rcx), params (%r8, empty), out (%r9).
    type ForwardFn =
        unsafe extern "C" fn(*const f32, *const f32, *const f32, *const f32, *const f32, *mut f32);
    let forward: libloading::Symbol<ForwardFn> =
        unsafe { lib.get(b"nfl_forward_FourInputMatmul") }.expect("dlsym");

    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.07).collect();
    let c: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.03).collect();
    let d: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.02).collect();
    let params: Vec<f32> = vec![]; // matmul + add have no params
    let mut out = vec![0.0f32; m * n];

    unsafe {
        forward(
            a.as_ptr(),
            b.as_ptr(),
            c.as_ptr(),
            d.as_ptr(),
            params.as_ptr(),
            out.as_mut_ptr(),
        );
    }

    // Reference: (a @ b) + c + d.
    // Matmul uses separate mul+add (NOT FMA) to match x86_64 mulss+addss.
    let mut expected = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            expected[i * n + j] = sum + c[i * n + j] + d[i * n + j];
        }
    }

    for (i, (got, want)) in out.iter().zip(expected.iter()).enumerate() {
        assert!(
            got.to_bits() == want.to_bits(),
            "mismatch at index {i}: got {got} ({:#010x}), want {want} ({:#010x})",
            got.to_bits(),
            want.to_bits()
        );
    }

    drop(lib);
}

// ---------------------------------------------------------------------------
// M14 Group F: LayerNorm FFI tests (x86_64).
//
// F.1: layernorm_no_affine_match_numerically — N=1, no γ/β. Validates bare
//      `layernorm` emitter path bit-exactly against layernorm_ref.
// F.2: layernorm_affine_match_numerically — N=1, γ/β params. Validates
//      Pass 3 affine multiply-add and γ-before-β ParamSlot order.
// F.3: pre_ln_block_match_numerically — N=2 transformer block. Validates
//      LH-1 closure (PR#31 fix) end-to-end: pre-fix, linear[w,b] at N=2
//      would produce silent corruption; post-fix, bit-exact green.
//
// Note: x86_64 matmul uses separate mulss+addss (NOT FMA) — pre_ln_block
// reference uses separate mul+add to match.
// ---------------------------------------------------------------------------

#[test]
fn layernorm_no_affine_match_numerically() {
    if !common::cc_available() {
        eprintln!("skip: integration test requires `cc` on PATH");
        return;
    }

    let src = std::fs::read_to_string("../../tests/fixtures/layernorm_no_affine.nfl")
        .expect("fixture readable");
    let nfl = compiler::parse(&src).expect("parse");
    let uir = compiler::ir::build(&nfl).expect("ir::build");
    let asm = profiles_x86_64::lower(&uir).expect("lower");

    let sig = &asm.functions[0];
    assert_eq!(sig.name, "nfl_forward_LayerNormNoAffine");
    assert_eq!(sig.inputs_floats, vec![8], "x is [2,4]=8 floats");
    assert_eq!(sig.params_floats, 0, "no-affine: no params");
    assert_eq!(sig.output_floats, 8, "output is [2,4]=8 floats");

    let so_path = common::compile_to_so(&asm.source, "layernorm_no_affine");
    let lib = unsafe { libloading::Library::new(&so_path) }.expect("dlopen");

    // SysV ABI: x (%rdi), params (%rsi, empty), out (%rdx).
    type ForwardFn = unsafe extern "C" fn(*const f32, *const f32, *mut f32);
    let forward: libloading::Symbol<ForwardFn> =
        unsafe { lib.get(b"nfl_forward_LayerNormNoAffine") }.expect("dlsym");

    // Deterministic input: sin-based values give non-trivial mean/var.
    let input: Vec<f32> = (0..8).map(|i| ((i as f32) * 0.7 + 1.0).sin()).collect();
    let params: Vec<f32> = vec![]; // no params
    let mut output = vec![0.0_f32; 8];

    unsafe {
        forward(input.as_ptr(), params.as_ptr(), output.as_mut_ptr());
    }

    let expected = common::layernorm_ref(&input, &[2, 4], None, None);

    for (i, (got, exp)) in output.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            got.to_bits(),
            exp.to_bits(),
            "bit-exact mismatch at idx {i}: got={got} ({:#010x}), expected={exp} ({:#010x})",
            got.to_bits(),
            exp.to_bits()
        );
    }

    drop(lib);
}

#[test]
fn layernorm_affine_match_numerically() {
    if !common::cc_available() {
        eprintln!("skip: integration test requires `cc` on PATH");
        return;
    }

    let src = std::fs::read_to_string("../../tests/fixtures/layernorm_affine.nfl")
        .expect("fixture readable");
    let nfl = compiler::parse(&src).expect("parse");
    let uir = compiler::ir::build(&nfl).expect("ir::build");
    let asm = profiles_x86_64::lower(&uir).expect("lower");

    let sig = &asm.functions[0];
    assert_eq!(sig.name, "nfl_forward_LayerNormAffine");
    assert_eq!(sig.inputs_floats, vec![8], "x is [2,4]=8 floats");
    assert_eq!(sig.params_floats, 8, "affine: γ(4) + β(4) = 8 floats");
    assert_eq!(sig.output_floats, 8, "output is [2,4]=8 floats");

    // Verify γ-before-β ParamSlot order contract.
    assert_eq!(sig.params_layout.len(), 2);
    assert_eq!(
        sig.params_layout[0].kind,
        profiles_x86_64::ParamKind::LayerNormScale
    );
    assert_eq!(sig.params_layout[0].offset, 0);
    assert_eq!(sig.params_layout[0].size, 4);
    assert_eq!(
        sig.params_layout[1].kind,
        profiles_x86_64::ParamKind::LayerNormBias
    );
    assert_eq!(sig.params_layout[1].offset, 4);
    assert_eq!(sig.params_layout[1].size, 4);

    let so_path = common::compile_to_so(&asm.source, "layernorm_affine");
    let lib = unsafe { libloading::Library::new(&so_path) }.expect("dlopen");

    // SysV ABI: x (%rdi), params (%rsi), out (%rdx).
    type ForwardFn = unsafe extern "C" fn(*const f32, *const f32, *mut f32);
    let forward: libloading::Symbol<ForwardFn> =
        unsafe { lib.get(b"nfl_forward_LayerNormAffine") }.expect("dlsym");

    let input: Vec<f32> = (0..8).map(|i| ((i as f32) * 0.7 + 1.0).sin()).collect();
    // Params: γ (4 floats) then β (4 floats).
    let gamma: Vec<f32> = vec![1.5, 0.8, 1.2, 0.5];
    let beta: Vec<f32> = vec![0.1, -0.2, 0.3, -0.1];
    let params: Vec<f32> = gamma.iter().chain(beta.iter()).copied().collect();
    let mut output = vec![0.0_f32; 8];

    unsafe {
        forward(input.as_ptr(), params.as_ptr(), output.as_mut_ptr());
    }

    let expected = common::layernorm_ref(&input, &[2, 4], Some(&params[0..4]), Some(&params[4..8]));

    for (i, (got, exp)) in output.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            got.to_bits(),
            exp.to_bits(),
            "bit-exact mismatch at idx {i}: got={got} ({:#010x}), expected={exp} ({:#010x})",
            got.to_bits(),
            exp.to_bits()
        );
    }

    drop(lib);
}

#[test]
fn pre_ln_block_match_numerically() {
    if !common::cc_available() {
        eprintln!("skip: integration test requires `cc` on PATH");
        return;
    }

    let src =
        std::fs::read_to_string("../../tests/fixtures/pre_ln_block.nfl").expect("fixture readable");
    let nfl = compiler::parse(&src).expect("parse");
    let uir = compiler::ir::build(&nfl).expect("ir::build");
    let asm = profiles_x86_64::lower(&uir).expect("lower");

    let sig = &asm.functions[0];
    assert_eq!(sig.name, "nfl_forward_PreLnBlock");
    assert_eq!(sig.inputs_floats.len(), 2, "N=2: x + skip");
    assert_eq!(sig.inputs_floats[0], 2 * 4, "x is [2,4]=8 floats");
    assert_eq!(sig.inputs_floats[1], 2 * 4, "skip is [2,4]=8 floats");
    // Params: γ(4) + β(4) + W(4×8=32) + b(8) = 48 floats.
    assert_eq!(sig.params_floats, 48, "γ(4)+β(4)+W(32)+b(8)=48");
    assert_eq!(sig.output_floats, 2 * 8, "output is [2,8]=16 floats");

    let so_path = common::compile_to_so(&asm.source, "pre_ln_block");
    let lib = unsafe { libloading::Library::new(&so_path) }.expect("dlopen");

    // SysV ABI (N=2 + params): x (%rdi), skip (%rsi), params (%rdx), out (%rcx).
    type ForwardFn = unsafe extern "C" fn(*const f32, *const f32, *const f32, *mut f32);
    let forward: libloading::Symbol<ForwardFn> =
        unsafe { lib.get(b"nfl_forward_PreLnBlock") }.expect("dlsym");

    let batch = 2usize;
    let dim = 4usize;
    let out_dim = 8usize;
    let x: Vec<f32> = (0..batch * dim).map(|i| (i as f32) * 0.15 + 0.5).collect();
    let skip: Vec<f32> = (0..batch * dim).map(|i| (i as f32) * 0.07 - 0.3).collect();
    // Params layout (traversal order): γ(4), β(4), W(4×8=32 row-major), b(8).
    let gamma: Vec<f32> = vec![1.2, 0.9, 1.1, 0.8];
    let beta: Vec<f32> = vec![0.05, -0.1, 0.15, -0.05];
    let weights: Vec<f32> = (0..dim * out_dim)
        .map(|i| (i as f32) * 0.03 - 0.5)
        .collect();
    let bias: Vec<f32> = (0..out_dim).map(|i| (i as f32) * 0.02 - 0.07).collect();
    let params: Vec<f32> = gamma
        .iter()
        .chain(beta.iter())
        .chain(weights.iter())
        .chain(bias.iter())
        .copied()
        .collect();
    assert_eq!(params.len(), 48);
    let mut out = vec![0.0_f32; batch * out_dim];

    unsafe {
        forward(x.as_ptr(), skip.as_ptr(), params.as_ptr(), out.as_mut_ptr());
    }

    // Rust reference: add → layernorm[affine] → linear[bias].
    // Step 1: residual add.
    let added: Vec<f32> = x.iter().zip(skip.iter()).map(|(a, b)| a + b).collect();
    // Step 2: layernorm with affine (γ-before-β matches ParamSlot order).
    let normalized = common::layernorm_ref(
        &added,
        &[batch, dim],
        Some(&params[0..4]),
        Some(&params[4..8]),
    );
    // Step 3: linear with bias (4 → 8). x86_64 uses separate mulss+addss
    // (NOT FMA); reference uses separate mul+add to match.
    let w = &params[8..40]; // 4×8 = 32 floats
    let b = &params[40..48]; // 8 floats
    let mut expected = vec![0.0_f32; batch * out_dim];
    for row in 0..batch {
        for j in 0..out_dim {
            let mut acc = 0.0_f32;
            for kk in 0..dim {
                acc += normalized[row * dim + kk] * w[kk * out_dim + j];
            }
            expected[row * out_dim + j] = acc + b[j];
        }
    }

    for (i, (got, want)) in out.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "bit-exact mismatch at idx {i}: got={got} ({:#010x}), want={want} ({:#010x})\n\
             (pre_ln_block validates LH-1 closure — bit mismatch may indicate a regression)",
            got.to_bits(),
            want.to_bits()
        );
    }

    drop(lib);
}

// ─── M17 exp_ref unit test (pure Rust; this file is Linux-gated, so it runs on
// Linux CI — the arm64 copy of this test covers macOS) ───────────────────────

#[test]
fn exp_ref_within_one_ulp_of_libm() {
    let ulp_diff = |a: f32, b: f32| (a.to_bits() as i64 - b.to_bits() as i64).abs();
    let mut x = -80.0_f32;
    while x <= 0.0 {
        let (got, want) = (common::exp_ref(x), x.exp());
        assert!(
            ulp_diff(got, want) <= 1,
            "x={x}: {} ulp",
            ulp_diff(got, want)
        );
        x += 0.0009765625;
    }
    for &x in &[0.0_f32, -std::f32::consts::LN_2, -1.0, -10.0, -50.0] {
        assert!(ulp_diff(common::exp_ref(x), x.exp()) <= 1, "x={x}");
    }
}

// ─── M15 FFN integration tests ───────────────────────────────────────────────

#[test]
fn ffn_ffi() {
    // M15 A2 third brick — FFN as compositional NFL pattern (x86_64).
    // Linux x86_64 only; macOS skipped via file-level cfg.
    //
    // Fixture: tests/fixtures/ffn.nfl (N=1, dim=4, hidden=8).
    // Expected: bit-exact match against common::ffn_ref.

    if !common::cc_available() {
        eprintln!("skip: requires cc");
        return;
    }

    let src = std::fs::read_to_string("../../tests/fixtures/ffn.nfl").unwrap();
    let ast = compiler::parse(&src).unwrap();
    let uir = compiler::ir::build(&ast).unwrap();
    let uir = compiler::passes::run_pipeline(&uir, &compiler::passes::default_pipeline())
        .expect("pipeline ok");
    let asm = profiles_x86_64::lower(&uir).expect("lower");

    let sig = &asm.functions[0];
    assert_eq!(sig.name, "nfl_forward_Ffn");
    // params: w1 (4*8=32) + b1 (8) + w2 (8*4=32) + b2 (4) = 76 floats.
    assert_eq!(sig.params_floats, 76);

    let so_path = common::compile_to_so(&asm.source, "ffn");
    let lib = unsafe { libloading::Library::new(&so_path) }.unwrap();
    let forward: libloading::Symbol<unsafe extern "C" fn(*const f32, *const f32, *mut f32)> =
        unsafe { lib.get(b"nfl_forward_Ffn").unwrap() };

    let mut input = vec![0.0f32; 2 * 4];
    for (i, v) in input.iter_mut().enumerate() {
        *v = (i as f32) * 0.1 - 0.4;
    }
    let mut params = vec![0.0f32; sig.params_floats];
    for (i, v) in params.iter_mut().enumerate() {
        *v = ((i as f32) - 38.0) * 0.01;
    }
    let mut output = vec![0.0f32; 2 * 4];
    unsafe {
        forward(input.as_ptr(), params.as_ptr(), output.as_mut_ptr());
    }

    let w1 = &params[0..32];
    let b1 = &params[32..40];
    let w2 = &params[40..72];
    let b2 = &params[72..76];

    let expected = common::ffn_ref(&input, w1, b1, w2, b2, 2, 4, 8);

    // Bit-exact comparison: x86_64 reference_matmul uses non-FMA `+= a * b`
    // (two roundings) matching the emitter's `mulss + addss` pattern.
    // M14 layernorm test precedent — same to_bits() discipline.
    for (i, (a, b)) in output.iter().zip(expected.iter()).enumerate() {
        assert!(
            a.to_bits() == b.to_bits(),
            "ffn[{i}]: asm got {a} (bits 0x{:08x}), ref got {b} (bits 0x{:08x})",
            a.to_bits(),
            b.to_bits()
        );
    }
    drop(lib);
}

#[test]
fn transformer_block_ffi() {
    // M15 — N=3 transformer block (LayerNorm + FFN + dual residual).
    //
    // x86_64 / Linux only (file-level #![cfg]). THE LH-4 RUNTIME EVIDENCE
    // TEST. Pre-T0 (without LH-4 fix): layernorm body clobbers output_reg=%r8
    // → segfault or wrong-output bit-mismatch vs Rust reference. Post-T0:
    // %r8 untouched (relocated to %r15), bit-exact match.

    if !common::cc_available() {
        eprintln!("skip: requires cc");
        return;
    }

    let src = std::fs::read_to_string("../../tests/fixtures/transformer_block.nfl").unwrap();
    let ast = compiler::parse(&src).unwrap();
    let uir = compiler::ir::build(&ast).unwrap();
    let uir = compiler::passes::run_pipeline(&uir, &compiler::passes::default_pipeline())
        .expect("pipeline ok");
    let asm = profiles_x86_64::lower(&uir).expect("lower");

    let sig = &asm.functions[0];
    assert_eq!(sig.name, "nfl_forward_TransformerBlock");
    // params: γ(4) + β(4) + w1(4*8=32) + b1(8) + w2(8*4=32) + b2(4) = 84.
    assert_eq!(sig.params_floats, 84);

    let so_path = common::compile_to_so(&asm.source, "transformer_block");
    let lib = unsafe { libloading::Library::new(&so_path) }.unwrap();
    let forward: libloading::Symbol<
        unsafe extern "C" fn(*const f32, *const f32, *const f32, *const f32, *mut f32),
    > = unsafe { lib.get(b"nfl_forward_TransformerBlock").unwrap() };

    let mut input = vec![0.0f32; 2 * 4];
    for (i, v) in input.iter_mut().enumerate() {
        *v = (i as f32) * 0.1 - 0.4;
    }
    let mut skip1 = vec![0.0f32; 2 * 4];
    for (i, v) in skip1.iter_mut().enumerate() {
        *v = (i as f32) * 0.05 - 0.2;
    }
    let mut skip2 = vec![0.0f32; 2 * 4];
    for (i, v) in skip2.iter_mut().enumerate() {
        *v = (i as f32) * 0.03 + 0.1;
    }
    let mut params = vec![0.0f32; sig.params_floats];
    for (i, v) in params.iter_mut().enumerate() {
        *v = ((i as f32) - 42.0) * 0.01;
    }
    let mut output = vec![0.0f32; 2 * 4];
    unsafe {
        forward(
            input.as_ptr(),
            skip1.as_ptr(),
            skip2.as_ptr(),
            params.as_ptr(),
            output.as_mut_ptr(),
        );
    }

    // Param blob slicing per compute_offsets traversal order.
    let gamma = &params[0..4];
    let beta = &params[4..8];
    let w1 = &params[8..40];
    let b1 = &params[40..48];
    let w2 = &params[48..80];
    let b2 = &params[80..84];

    let expected =
        common::transformer_block_ref(&input, &skip1, &skip2, gamma, beta, w1, b1, w2, b2, 2, 4, 8);

    // Bit-exact comparison: composition of layernorm_ref (M14 verified
    // bit-exact) + ffn_ref (uses x86_64's non-FMA `+= a * b` reference_matmul
    // matching `mulss + addss`) + element-wise add. Determinism preserved
    // through the chain.
    //
    // THIS IS THE LH-4 RUNTIME EVIDENCE TEST. Pre-T0: layernorm body
    // clobbers output_reg=%r8 → segfault or bit-mismatch here. Post-T0:
    // %r8 untouched → bit-exact match.
    for (i, (a, b)) in output.iter().zip(expected.iter()).enumerate() {
        assert!(
            a.to_bits() == b.to_bits(),
            "transformer_block[{i}]: asm got {a} (bits 0x{:08x}), ref got {b} (bits 0x{:08x})",
            a.to_bits(),
            b.to_bits()
        );
    }

    drop(lib);
}

// ---------------------------------------------------------------------------
// M17 Layer-1 FFI: softmax_only fixture — bit-exact + underflow-clamp evidence.
//
// T.1: softmax_only_ffi_bit_exact_vs_exp_ref
//      Compile softmax_only.nfl to a .so, run it, assert every output
//      element matches reference_softmax_stable bit-for-bit. This is the
//      definitive evidence that the x86_64 inline exp matches exp_ref.
//
// T.2: softmax_only_ffi_underflow_clamp_agrees_with_libm
//      Feed a row with huge logit spread (x[0]=120, x[1..8]=-120) so
//      the shifted exponents z = (x[j] - row_max) * log2e underflow past
//      the bias floor (z < -127). The branchless flush sets those outputs
//      to exactly +0.0. Assert the flushed terms are 0.0 and the row sums
//      to 1 within 1e-6.
// ---------------------------------------------------------------------------

#[test]
fn softmax_only_ffi_bit_exact_vs_exp_ref() {
    if !common::cc_available() {
        eprintln!("skip: cc unavailable");
        return;
    }

    const B: usize = 4;
    const K: usize = 8;
    const TOTAL: usize = B * K;

    let src =
        std::fs::read_to_string("../../tests/fixtures/softmax_only.nfl").expect("fixture readable");
    let ast = compiler::parse(&src).expect("parse");
    let uir = compiler::ir::build(&ast).expect("ir::build");
    let uir = compiler::passes::run_pipeline(&uir, &compiler::passes::default_pipeline())
        .expect("pipeline ok");
    let asm = profiles_x86_64::lower(&uir).expect("lower");

    let sig = &asm.functions[0];
    assert_eq!(sig.name, "nfl_forward_SoftmaxOnly");
    assert_eq!(sig.inputs_floats, vec![TOTAL], "input is [4,8]=32 floats");
    assert_eq!(sig.params_floats, 0, "softmax has no learnable params");
    assert_eq!(sig.output_floats, TOTAL, "output is [4,8]=32 floats");

    let so_path = common::compile_to_so(&asm.source, "softmax_only_bit_exact");

    let input = deterministic_input(TOTAL);
    let params = [0.0f32; 1]; // non-empty dummy; never dereferenced
    let mut output = vec![0.0f32; TOTAL];

    unsafe {
        let lib = libloading::Library::new(&so_path).expect("dlopen");
        let forward: libloading::Symbol<unsafe extern "C" fn(*const f32, *const f32, *mut f32)> =
            lib.get(b"nfl_forward_SoftmaxOnly").expect("dlsym");
        forward(input.as_ptr(), params.as_ptr(), output.as_mut_ptr());
    }

    let reference = reference_softmax_stable(&input, B, K);

    for (i, (got, want)) in output.iter().zip(reference.iter()).enumerate() {
        assert!(
            got.to_bits() == want.to_bits(),
            "bit-exact mismatch at index {i}: got={got} ({:#010x}), want={want} ({:#010x})",
            got.to_bits(),
            want.to_bits()
        );
    }
}

#[test]
fn softmax_only_ffi_underflow_clamp_agrees_with_libm() {
    if !common::cc_available() {
        eprintln!("skip: cc unavailable");
        return;
    }

    let src =
        std::fs::read_to_string("../../tests/fixtures/softmax_only.nfl").expect("fixture readable");
    let ast = compiler::parse(&src).expect("parse");
    let uir = compiler::ir::build(&ast).expect("ir::build");
    let uir = compiler::passes::run_pipeline(&uir, &compiler::passes::default_pipeline())
        .expect("pipeline ok");
    let asm = profiles_x86_64::lower(&uir).expect("lower");

    let so_path = common::compile_to_so(&asm.source, "softmax_only_underflow");

    // batch=4, k=8. Row 0 has a huge spread: x[0]=120, x[1..8]=-120.
    // After max-subtraction: x[0]-max = 0, x[j]-max ≈ -240 for j in 1..8.
    // (−240) * log2e ≈ −346.4, well below the flush floor of z < −127.
    let mut input = deterministic_input(32);
    input[0] = 120.0_f32;
    for v in input.iter_mut().take(8).skip(1) {
        *v = -120.0_f32;
    }
    let params = [0.0f32; 1]; // non-empty dummy; never dereferenced
    let mut output = vec![0.0f32; 32];

    unsafe {
        let lib = libloading::Library::new(&so_path).expect("dlopen");
        let forward: libloading::Symbol<unsafe extern "C" fn(*const f32, *const f32, *mut f32)> =
            lib.get(b"nfl_forward_SoftmaxOnly").expect("dlsym");
        forward(input.as_ptr(), params.as_ptr(), output.as_mut_ptr());
    }

    // Row 0: indices 1..8 must be exactly +0.0 (flushed by branchless clamp).
    for (j, got) in output.iter().enumerate().take(8).skip(1) {
        assert!(
            got.to_bits() == 0.0f32.to_bits(),
            "output[{j}] should be exactly +0.0 (underflow flush), got {got}"
        );
    }

    // Row 0 must still be a valid probability distribution (sums to ~1).
    let row0_sum: f32 = output[0..8].iter().sum();
    assert!(
        (row0_sum - 1.0).abs() < 1e-6,
        "row 0 sum = {row0_sum}, expected ~1.0"
    );
}
