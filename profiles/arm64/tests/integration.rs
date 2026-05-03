//! M4a end-to-end integration test.

mod common;

#[test]
fn tinymlp_no_softmax_runs_correctly() {
    // Pre-flight gates.
    if !cfg!(target_arch = "aarch64") {
        eprintln!("skip: integration test requires aarch64 host");
        return;
    }
    if !common::cc_available() {
        eprintln!("skip: integration test requires `cc` on PATH");
        return;
    }

    // 1. Read fixture (path is relative to the integration-test crate root,
    //    which is profiles/arm64/. So the fixture is two dirs up.)
    let src = std::fs::read_to_string("../../tests/fixtures/m4_linear_relu.nfl")
        .expect("fixture readable");
    let ast = compiler::parse(&src).expect("parse");
    let uir = compiler::ir::build(&ast).expect("ir::build");

    // 2. Lower.
    let asm = profiles_arm64::lower(&uir).expect("lower");
    assert_eq!(asm.functions.len(), 1, "one function expected");
    let sig = &asm.functions[0];
    assert_eq!(sig.name, "nfl_forward_M4Demo");
    assert_eq!(sig.input_floats, 32);   // 8*4
    assert_eq!(sig.weight_floats, 8);   // 4*2
    assert_eq!(sig.output_floats, 16);  // 8*2

    // 3. Assemble + link.
    let dylib_path = common::compile_to_dylib(&asm.source, "m4_linear_relu");

    // 4. dlopen + call via FFI.
    let lib = unsafe { libloading::Library::new(&dylib_path) }
        .expect("libloading: open dylib");
    let forward: libloading::Symbol<
        unsafe extern "C" fn(*const f32, *const f32, *mut f32),
    > = unsafe { lib.get(b"nfl_forward_M4Demo") }
        .expect("dlsym: nfl_forward_M4Demo not found");

    // Deterministic test inputs.
    let mut input = [0.0f32; 32];
    for (i, v) in input.iter_mut().enumerate() {
        *v = (i as f32) * 0.1 - 1.5; // mix of negatives + positives so relu has work
    }
    let mut weights = [0.0f32; 8];
    for (i, v) in weights.iter_mut().enumerate() {
        *v = ((i as f32) - 4.0) * 0.25;
    }
    let mut output = [0.0f32; 16];

    unsafe { forward(input.as_ptr(), weights.as_ptr(), output.as_mut_ptr()); }

    // 5. Compare against pure-Rust reference.
    let expected = reference_linear_relu(&input, &weights);
    for (i, (a, b)) in output.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "output[{i}]: asm got {a}, reference got {b}, diff {}",
            (a - b).abs()
        );
    }
}

/// Reference: matmul (input [B,K] × weights [K,N]) followed by elementwise relu.
/// Mirrors the asm spec exactly. B=8, K=4, N=2 hardcoded for the M4a fixture.
fn reference_linear_relu(input: &[f32; 32], weights: &[f32; 8]) -> [f32; 16] {
    const B: usize = 8;
    const K: usize = 4;
    const N: usize = 2;
    let mut out = [0.0f32; 16];
    for i in 0..B {
        for j in 0..N {
            let mut sum = 0.0f32;
            for k in 0..K {
                sum += input[i * K + k] * weights[k * N + j];
            }
            out[i * N + j] = sum.max(0.0);
        }
    }
    out
}
