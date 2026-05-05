//! M4a + M4b end-to-end integration tests.

mod common;

// ---------------------------------------------------------------------------
// Reference implementations
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Reference-validation unit tests (always run, no FFI)
// ---------------------------------------------------------------------------

#[test]
fn reference_softmax_stable_known_values() {
    let input = [1.0f32, 2.0, 3.0];
    let output = reference_softmax_stable(&input, 1, 3);
    // softmax([1,2,3]) ≈ [0.0900, 0.2447, 0.6652]
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
    if !cfg!(target_arch = "aarch64") {
        eprintln!("skip: integration test requires aarch64 host");
        return;
    }
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

    let asm = profiles_arm64::lower(&uir).expect("lower");
    assert_eq!(asm.functions.len(), 1);
    let sig = &asm.functions[0];
    assert_eq!(sig.name, "nfl_forward_M4Demo");
    assert_eq!(sig.input_floats, 32);
    assert_eq!(sig.params_floats, 8);
    assert_eq!(sig.output_floats, 16);

    // Verify the params_layout is what we expect for M4a fixture (single LinearWeight slot).
    assert_eq!(sig.params_layout.len(), 1);
    let slot = &sig.params_layout[0];
    assert_eq!(slot.kind, profiles_arm64::ParamKind::LinearWeight);
    assert_eq!(slot.offset, 0);
    assert_eq!(slot.size, 8);

    let dylib_path = common::compile_to_dylib(&asm.source, "m4a_linear_relu");

    let lib = unsafe { libloading::Library::new(&dylib_path) }.expect("open");
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
    if !cfg!(target_arch = "aarch64") {
        eprintln!("skip: requires aarch64");
        return;
    }
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
    let asm = profiles_arm64::lower(&uir).expect("lower");
    let dylib_path = common::compile_to_dylib(&asm.source, "tinymlp_softmax");

    let lib = unsafe { libloading::Library::new(&dylib_path) }.unwrap();
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
    if !cfg!(target_arch = "aarch64") {
        eprintln!("skip: requires aarch64");
        return;
    }
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
    let asm = profiles_arm64::lower(&uir).expect("lower");

    // Confirm layout: linear[16, bias=true] + linear[output=2] (no bias) + softmax.
    // params: weight(8*16=128) + bias(16) + weight(16*2=32) = 176 floats.
    let sig = &asm.functions[0];
    assert_eq!(sig.name, "nfl_forward_MixedArgs");
    assert_eq!(sig.params_floats, 8 * 16 + 16 + 16 * 2);

    let dylib_path = common::compile_to_dylib(&asm.source, "mixed_args");
    let lib = unsafe { libloading::Library::new(&dylib_path) }.unwrap();
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
    if !cfg!(target_arch = "aarch64") {
        eprintln!("skip: requires aarch64");
        return;
    }
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
    let asm = profiles_arm64::lower(&uir).expect("lower");

    let sig = &asm.functions[0];
    assert_eq!(sig.name, "nfl_forward_Classifier");
    // batch=32, input=784, hidden=512+256, output=10.
    // Linears: 784*512 + 512*256 + 256*10 = 401408 + 131072 + 2560 = 535040
    assert_eq!(sig.params_floats, 535040);

    let dylib_path = common::compile_to_dylib(&asm.source, "classifier");
    let lib = unsafe { libloading::Library::new(&dylib_path) }.unwrap();
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
    if !cfg!(target_arch = "aarch64") {
        eprintln!("skip: requires aarch64");
        return;
    }
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
    let asm = profiles_arm64::lower(&uir).expect("lower");

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

    let dylib_path = common::compile_to_dylib(&asm.source, "pipeline_styles");
    let lib = unsafe { libloading::Library::new(&dylib_path) }.unwrap();

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
    if !cfg!(target_arch = "aarch64") {
        eprintln!("skip: requires aarch64");
        return;
    }
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
    let asm = profiles_arm64::lower(&uir).expect("lower");

    let sig = &asm.functions[0];
    assert_eq!(sig.name, "nfl_forward_Commented");

    let dylib_path = common::compile_to_dylib(&asm.source, "comments");
    let lib = unsafe { libloading::Library::new(&dylib_path) }.unwrap();
    let forward: libloading::Symbol<unsafe extern "C" fn(*const f32, *const f32, *mut f32)> =
        unsafe { lib.get(b"nfl_forward_Commented") }.unwrap();

    let mut input = vec![0.0f32; sig.input_floats];
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
    if !cfg!(target_arch = "aarch64") {
        eprintln!("skip: requires aarch64");
        return;
    }
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
    let fused_asm = profiles_arm64::lower(&fused_uir).expect("fused lower");
    let unfused_asm = profiles_arm64::lower(&uir).expect("unfused lower");

    // Asm shape differs as expected.
    assert!(
        fused_asm.source.contains("fmax    s0, s0, s4"),
        "fused asm missing inline fmax"
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
    let fused_dylib = common::compile_to_dylib(&fused_asm.source, "fused_classifier");
    let unfused_dylib = common::compile_to_dylib(&unfused_asm.source, "unfused_classifier");

    let fused_lib = unsafe { libloading::Library::new(&fused_dylib).unwrap() };
    let unfused_lib = unsafe { libloading::Library::new(&unfused_dylib).unwrap() };

    let fused_forward: libloading::Symbol<unsafe extern "C" fn(*const f32, *const f32, *mut f32)> =
        unsafe { fused_lib.get(b"nfl_forward_Classifier") }.unwrap();
    let unfused_forward: libloading::Symbol<
        unsafe extern "C" fn(*const f32, *const f32, *mut f32),
    > = unsafe { unfused_lib.get(b"nfl_forward_Classifier") }.unwrap();

    // Same deterministic input + params as classifier_runs_correctly test.
    // Both asms describe the same model; pull params length from the FnSig
    // instead of hardcoding so the test follows the fixture if it changes.
    let params_len = fused_asm.functions[0].params_floats;
    debug_assert_eq!(
        params_len, unfused_asm.functions[0].params_floats,
        "fused/unfused FnSig params_floats must agree"
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
fn fused_vs_unfused_mixed_args_match_numerically() {
    if !cfg!(target_arch = "aarch64") {
        eprintln!("skip: requires aarch64");
        return;
    }
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
    let fused_asm = profiles_arm64::lower(&fused_uir).expect("fused lower");
    let unfused_asm = profiles_arm64::lower(&uir).expect("unfused lower");

    // Asm shape pre-asserts.
    assert!(
        fused_asm.source.contains("fmax    s0, s0, s4"),
        "fused asm missing inline fmax"
    );
    // mixed_args.nfl has `linear[16, bias=true] → relu` as the first
    // fusion candidate; the fused asm must still contain bias-add
    // (fadd s0, s0, s5) immediately before the fmax (within one
    // emit_linear, not in a separate function).
    assert!(
        fused_asm.source.contains("fadd    s0, s0, s5"),
        "fused asm missing bias-add (fadd s0, s0, s5):\n{}",
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
    let fused_dylib = common::compile_to_dylib(&fused_asm.source, "fused_mixed_args");
    let unfused_dylib = common::compile_to_dylib(&unfused_asm.source, "unfused_mixed_args");

    let fused_lib = unsafe { libloading::Library::new(&fused_dylib).unwrap() };
    let unfused_lib = unsafe { libloading::Library::new(&unfused_dylib).unwrap() };

    let fused_forward: libloading::Symbol<unsafe extern "C" fn(*const f32, *const f32, *mut f32)> =
        unsafe { fused_lib.get(b"nfl_forward_MixedArgs") }.unwrap();
    let unfused_forward: libloading::Symbol<
        unsafe extern "C" fn(*const f32, *const f32, *mut f32),
    > = unsafe { unfused_lib.get(b"nfl_forward_MixedArgs") }.unwrap();

    // Same deterministic input + params formula as the classifier
    // integration test. mixed_args has batch=4, input=8, output=2.
    let params_len = fused_asm.functions[0].params_floats;
    debug_assert_eq!(
        params_len, unfused_asm.functions[0].params_floats,
        "fused/unfused FnSig params_floats must agree"
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
    // bias-aware fusion (matmul → bias-add → fmax → store) only
    // relocates WHERE relu/bias is applied, not WHICH floats compute.
    for (i, (a, b)) in fused_out.iter().zip(unfused_out.iter()).enumerate() {
        assert_eq!(
            *a, *b,
            "fused[{i}]={a} unfused[{i}]={b} — bias-aware fusion changed numerics"
        );
    }
}
