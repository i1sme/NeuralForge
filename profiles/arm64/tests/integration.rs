//! M4a end-to-end integration test.

mod common;

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

    let asm = profiles_arm64::lower(&uir).expect("lower");
    assert_eq!(asm.functions.len(), 1);
    let sig = &asm.functions[0];
    assert_eq!(sig.name, "nfl_forward_M4Demo");
    assert_eq!(sig.input_floats, 32);
    assert_eq!(sig.params_floats, 8);
    assert_eq!(sig.output_floats, 16);

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

fn reference_linear_relu(input: &[f32; 32], params: &[f32; 8]) -> [f32; 16] {
    const B: usize = 8;
    const K: usize = 4;
    const N: usize = 2;
    let mut out = [0.0f32; 16];
    for i in 0..B {
        for j in 0..N {
            let mut sum = 0.0f32;
            for k in 0..K {
                sum += input[i * K + k] * params[k * N + j];
            }
            out[i * N + j] = sum.max(0.0);
        }
    }
    out
}
