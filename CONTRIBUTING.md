# Contributing to NeuralForge

Thanks for your interest. The notes below are short on purpose.

## Contributor License Agreement

By submitting any contribution (pull request, patch, suggestion, or any
other code or content) to this project, you agree to the following:

1. **License grant.** You license your contribution under the GNU Affero
   General Public License v3.0 (`AGPL-3.0-only`) — the same license that
   governs the rest of the project.

2. **Commercial relicensing grant.** You grant the project owner
   (Arsenii Voloshyn) a perpetual, irrevocable, worldwide, royalty-free,
   non-exclusive right to relicense your contribution under additional
   terms, including proprietary commercial licenses. This is what enables
   NeuralForge's AGPL-3.0 + commercial dual-licensing model. See
   `README.md` for the commercial-licensing contact.

3. **Future patches.** This agreement covers not only the contribution as
   initially submitted but also every subsequent update within the same
   pull request, branch, or revision history — follow-up commits,
   force-pushes, amendments, rebases, and review-fixup patches all fall
   under the same grant. No additional action is required from you when
   you push updates.

4. **Original work.** You confirm that the contribution is your own work,
   or that you have the right to submit it under these terms.

If you cannot agree (for example, because your employer or another party
holds copyright on your work), please arrange the necessary permissions
before submitting.

## Development workflow

The non-negotiable rules are in `CLAUDE.md`. The short version:

- Test-driven: red → green → refactor.
- Before any commit: `cargo fmt --all`, `cargo clippy --workspace
  --all-targets -- -D warnings`, `cargo test --workspace`. All three
  must be clean.
- Update `DEVLOG.md` with a session entry. The format is documented at
  the top of that file.
- For substantial changes, open an issue first to discuss the design.

## Adding a new operation, profile, or pass

`CLAUDE.md` has step-by-step recipes for the three most common kinds of
change (new operation, new profile, new pass). Follow those rather than
inventing your own structure.
