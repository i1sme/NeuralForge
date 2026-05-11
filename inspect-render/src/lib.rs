// SPDX-License-Identifier: Apache-2.0

//! Renders `profile_api::Inspection` to human-readable text for
//! `nflc inspect` and per-profile golden integration tests.
//!
//! Format spec: `docs/superpowers/specs/2026-05-11-a3-viewer-annotations-design.md` §5.

use profile_api::{BufferLoc, FnAnnotations, Inspection, NodeAnnotation};
use std::path::Path;

/// CLI-invocation context for the rendered header. Kept out of
/// `Inspection` because file path / profile / pass list are not analysis
/// outputs — they're inputs to the invocation.
pub struct RenderHeader<'a> {
    pub source_path: &'a Path,
    pub profile: &'a str,
    /// `Some(names)` when a pipeline ran (default or filtered);
    /// `None` when `--no-passes` skipped the pipeline.
    pub applied_passes: Option<&'a [&'a str]>,
}

/// Render a full Inspection. Output ends with a trailing newline.
pub fn render_inspection(insp: &Inspection, header: RenderHeader<'_>) -> String {
    let mut out = String::new();

    // Header: command-style line + applied-passes status.
    out.push_str(&format!(
        "inspect {} --profile {}\n",
        header.source_path.display(),
        header.profile
    ));
    match header.applied_passes {
        Some(names) => {
            out.push_str(&format!("  passes applied: {}\n", names.join(", ")));
        }
        None => {
            out.push_str("  passes: skipped\n");
        }
    }
    out.push('\n');

    for fa in &insp.functions {
        render_fn_annotations(&mut out, fa);
    }

    out
}

fn render_fn_annotations(out: &mut String, fa: &FnAnnotations) {
    out.push_str(&format!("inspect-model {}\n", fa.fn_sig.model));

    // Inputs line: real NodeId refs from input_nodes (NOT positional).
    let total_input_floats: usize = fa.fn_sig.inputs_floats.iter().sum();
    let total_input_bytes = total_input_floats * 4;
    let n_inputs = fa.input_nodes.len();
    let input_node_refs: Vec<String> = fa.input_nodes.iter().map(|id| format!("n{}", id)).collect();
    // Only emit `(N B each)` when all inputs share the same byte-size.
    // For non-uniform inputs (e.g. four_input_matmul.nfl with mixed shapes),
    // "each" would be a factual lie — emit no per-input clause, the total
    // bytes alone is honest.
    let inputs_uniform = fa.fn_sig.inputs_floats.windows(2).all(|w| w[0] == w[1]);
    let inputs_per_count_clause = if n_inputs > 1 && inputs_uniform {
        format!(" ({} B each)", total_input_bytes / n_inputs)
    } else {
        String::new()
    };
    out.push_str(&format!(
        "  inputs:        [{}]                {} floats ({} B){}\n",
        input_node_refs.join(", "),
        total_input_floats,
        total_input_bytes,
        inputs_per_count_clause
    ));

    // Output line. Real NodeId from output_node field.
    let output_bytes = fa.fn_sig.output_floats * 4;
    out.push_str(&format!(
        "  output:        n{}                  {} floats ({} B)\n",
        fa.output_node, fa.fn_sig.output_floats, output_bytes
    ));

    let params_bytes = fa.fn_sig.params_floats * 4;
    out.push_str(&format!(
        "  params:        {} floats            ({} B)\n",
        fa.fn_sig.params_floats, params_bytes
    ));

    out.push_str(&format!(
        "  stack frame:   {} bytes             (16-byte aligned)\n",
        fa.stack_bytes
    ));

    out.push_str(&format!(
        "  callee-saved:  [{}]\n",
        fa.callee_saved.join(", ")
    ));

    out.push_str(&format!(
        "  leaf:          {}\n",
        if fa.leaf { "yes" } else { "no" }
    ));

    out.push_str("\n  nodes:\n");
    for (node_idx, na) in fa.nodes.iter().enumerate() {
        render_node_annotation(out, node_idx, na);
    }
    out.push('\n');
}

fn render_node_annotation(out: &mut String, node_idx: usize, na: &NodeAnnotation) {
    // Line 1: node id ref + pre-rendered label (op kind + shape +
    // operands + attrs + fused). Format mirrors `--uir-verbose`
    // per-node line — visual continuity.
    out.push_str(&format!("    n{}  {}\n", node_idx, na.label));

    // Line 2: annotation row.
    let mut parts: Vec<String> = Vec::new();
    parts.push(format!("loc={}", format_buffer_loc(na.buffer_loc)));
    parts.push(format!("out={} B", na.output_bytes));
    if let Some(p) = na.params_floats {
        parts.push(format!("params={} floats ({} B)", p, p * 4));
    }
    for note in &na.extra_notes {
        parts.push(note.clone());
    }
    out.push_str(&format!("          {}\n", parts.join("    ")));
}

fn format_buffer_loc(loc: BufferLoc) -> String {
    match loc {
        BufferLoc::InputReg(idx) => format!("InputReg({})", idx),
        BufferLoc::OutputReg => "OutputReg".to_string(),
        BufferLoc::StackOffset(off) => format!("StackOffset({})", off),
        BufferLoc::Alias(node_id) => format!("Alias(n{})", node_id),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use profile_api::{FnSig, Inspection, ParamKind, ParamSlot};
    use std::path::PathBuf;

    fn dummy_inspection() -> Inspection {
        Inspection {
            functions: vec![FnAnnotations {
                fn_sig: FnSig {
                    name: "nfl_forward_M".to_string(),
                    model: "M".to_string(),
                    inputs_floats: vec![6],
                    output_floats: 4,
                    params_floats: 6,
                    params_layout: vec![ParamSlot {
                        kind: ParamKind::LinearWeight,
                        origin_node: 1,
                        offset: 0,
                        size: 6,
                    }],
                },
                stack_bytes: 0,
                callee_saved: vec![],
                leaf: true,
                input_nodes: vec![0],
                output_node: 1,
                nodes: vec![
                    NodeAnnotation {
                        label: "input \"x\"        :: Tensor[2, 3]".to_string(),
                        buffer_loc: BufferLoc::InputReg(0),
                        output_bytes: 24,
                        params_floats: None,
                        extra_notes: vec![],
                    },
                    NodeAnnotation {
                        label:
                            "linear           :: Tensor[2, 2]    operands=[n0]    attrs=[out_dim=2]"
                                .to_string(),
                        buffer_loc: BufferLoc::OutputReg,
                        output_bytes: 16,
                        params_floats: Some(6),
                        extra_notes: vec![],
                    },
                ],
            }],
        }
    }

    #[test]
    fn render_contains_required_markers() {
        let path = PathBuf::from("test.nfl");
        let passes = ["fuse_linear_relu"];
        let header = RenderHeader {
            source_path: &path,
            profile: "arm64",
            applied_passes: Some(&passes),
        };
        let out = render_inspection(&dummy_inspection(), header);
        assert!(
            out.contains("inspect-model M"),
            "missing model header: {}",
            out
        );
        assert!(
            out.contains("loc=InputReg(0)"),
            "missing loc render: {}",
            out
        );
        assert!(
            out.contains("loc=OutputReg"),
            "missing OutputReg render: {}",
            out
        );
        assert!(out.contains("out=24 B"), "missing output_bytes render");
        assert!(
            out.contains("params=6 floats (24 B)"),
            "missing params line: {}",
            out
        );
        assert!(
            out.contains("passes applied: fuse_linear_relu"),
            "missing passes line"
        );
        // Label rendered on line 1.
        assert!(
            out.contains("n0  input \"x\""),
            "line-1 label missing for input node: {}",
            out
        );
        assert!(
            out.contains("n1  linear"),
            "line-1 label missing for op node: {}",
            out
        );
    }

    #[test]
    fn render_no_passes_marker() {
        let path = PathBuf::from("test.nfl");
        let header = RenderHeader {
            source_path: &path,
            profile: "arm64",
            applied_passes: None,
        };
        let out = render_inspection(&dummy_inspection(), header);
        assert!(
            out.contains("passes: skipped"),
            "missing skipped marker: {}",
            out
        );
    }
}
