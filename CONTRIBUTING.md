# Contributing to NeuralForge

Thanks for your interest. The notes below are short on purpose.

## License of contributions

NeuralForge is licensed under the [Apache License 2.0](LICENSE). Per
§5 of the Apache License ("Submission of Contributions"), every
contribution you submit is implicitly licensed to the project under
the same terms unless you state otherwise. No separate Contributor
License Agreement is required.

## Development workflow

The non-negotiable rules are in `CLAUDE.md`. The short version:

- Test-driven: red → green → refactor.
- Before any commit: `cargo fmt --all`, `cargo clippy --workspace
  --all-targets -- -D warnings`, `cargo test --workspace`. All three
  must be clean.
- Update `DEVLOG.md` with a session entry. The format is documented
  at the top of that file.
- For substantial changes, open an issue first to discuss the design.

## Adding a new operation, profile, or pass

`CLAUDE.md` has step-by-step recipes for the three most common kinds
of change (new operation, new profile, new pass). Follow those rather
than inventing your own structure.
