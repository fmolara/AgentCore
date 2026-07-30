# Planner v2 Finalization

Planner v2 finalizes an explored task through bounded candidate generation,
deterministic diagnostics, independent structured review, and at most one
revision. These steps are planning only. They cannot invoke `TaskExecutor`,
change the workspace, run checks, create checkpoints, mutate Git, or grant
approval.

## Compact Candidates

The final planner prompt prefers `replace_text` for localized changes to
existing files. `write_file` remains available for new files and justified
whole-file replacement. Deterministic diagnostics identify:

- whole-file writes targeting existing files;
- embedded action payloads above the configured warning threshold;
- executable reads that repeat completed exploration reads;
- explicitly named configured checks that are absent.

Existing-file writes are warnings by default. A repository may set
`forbid_existing_file_write: true` to reject them.

## Phase Budgets

Finalization has separate configured caps and minimum safe output sizes for
exploration, final candidates, format recovery, review, revision, and final
review. Every effective limit respects its phase cap, an explicit caller cap,
and remaining model context. Planning fails before generation when the
available output budget is below the configured safe minimum.

`planning.generation_budget` records the phase, requested and effective limits,
minimum, prompt size, and context size.

## Format Recovery

Malformed structured output receives exactly one recovery attempt for that
step. Recovery uses a fresh model session and requests a complete replacement
JSON object. AgentCore never appends to, guesses, parses actions from, or
executes the malformed fragment. A second malformed response fails planning.
Malformed output retained in events is bounded.

## Independent Review

After deterministic validation, a fresh model session reviews the candidate
against the original task, observations, concrete actions, diagnostics, and
trusted check names. It returns strict JSON with one verdict:

- `accept`: no material defect was identified from the available evidence;
- `revise`: structured findings require changes;
- `cannot_verify`: available evidence is insufficient.

Acceptance is a safety and quality gate, not a formal proof of correctness.
`cannot_verify` fails safely.

A `revise` verdict permits one complete replacement candidate. The revision is
validated again and receives a fresh final review. A second rejection,
`cannot_verify`, or malformed unrecoverable response ends planning. Candidates
have distinct IDs, and no proposal or approval exists until a candidate is
accepted.

## Approval Boundary

Only an accepted candidate becomes a normal `PlanProposal`. The existing
approval policy is then evaluated, and the operator must explicitly approve
before execution. Local and distributed composition roots share the same
`IterativeLLMPlanner` and finalization implementation.

All finalization event names are additive. Assistant events contain only
visible model output; candidate, recovery, review, revision, and failure
lifecycle data use structured planning events.
