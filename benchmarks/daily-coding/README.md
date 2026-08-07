# Daily Coding Qualification

`manifest.yaml` defines the frozen ten-task suite used to compare native
ToolLoopAgent model protocols. Controlled fixture repositories and downloaded
third-party sources are materialized outside this repository. The manifest
contains no machine-specific paths and reference patches are never supplied to
the model.

The suite intentionally mixes localized fixes, multi-file features, failing
tests, build configuration, behavior-preserving refactoring, validation,
Python tooling, and a historical public defect. The `qwen_baseline_subset` is
selected before primary-model results are observed.

Each run must use a fresh clean clone, the exact prompt from the manifest, the
same eight semantic tools, per-call approval, symbolic checks, an exported
patch, and independent validation. A model run never creates a commit.
