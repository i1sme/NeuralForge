// SPDX-License-Identifier: Apache-2.0
#![cfg(all(target_os = "linux", target_arch = "x86_64"))]

//! M9 end-to-end FFI integration tests for the x86_64 Linux ELF profile.
//!
//! Mirrors profiles/arm64/tests/integration.rs. Each test loads a
//! fixture, lowers via X86_64Profile, assembles via cc -shared -fPIC,
//! dlopens the .so, calls the FFI symbol, and asserts numerical
//! agreement against a Rust-computed reference.

mod common;

// ─── Reference implementations (verbatim from arm64 — pure Rust) ─────────────

fn reference_matmul(input: &[f32], weights: &[f32], b: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; b * n];
    for i in 0..b {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum = f32::mul_add(input[i * k + kk], weights[kk * n + j], sum);
            }
            out[i * n + j] = sum;
        }
    }
    out
}

fn reference_bias_add(acc: &[f32], bias: &[f32], n: usize) -> Vec<f32> {
    let b = acc.len() / n;
    let mut out = acc.to_vec();
    for i in 0..b {
        for j in 0..n {
            out[i * n + j] += bias[j];
        }
    }
    out
}

fn reference_relu(input: &[f32]) -> Vec<f32> {
    input.iter().map(|x| x.max(0.0)).collect()
}

fn reference_softmax_stable(input: &[f32], b: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; b * k];
    for i in 0..b {
        let row = &input[i * k..(i + 1) * k];
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for kk in 0..k {
            let e = (row[kk] - max).exp();
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
/// the arm64 reference, not a defect. `f32::exp` wraps platform libm
/// `expf` (glibc on Linux x86_64).
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

        for i in 0..seq {
            let row = &scores[i * seq..(i + 1) * seq];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for j in 0..seq {
                let e = (row[j] - max).exp();
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
        // pins softmax fusion (.Lfsmx_ + call expf@PLT) — complementary, not
        // redundant.
        assert!(
            fused_asm.source.contains("call    expf@PLT"),
            "{fixture_path}: fused asm missing call expf@PLT in row-wise tail"
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
        asm.source.contains("call    expf@PLT"),
        "fused softmax tail must call expf@PLT"
    );
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
// reference uses separate `let prod = a*b; acc = acc + prod;`. libm expf
// (glibc on Linux x86_64) is the same function called in both asm and
// reference.
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
