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

## Context Capacity

Planner v2 distinguishes the active runtime capacity from model architecture
metadata. Trusted AgentCore configuration and reliable runtime-reported
capacity are combined using the lower value. Model `max_position_embeddings`
is diagnostic only. When neither active source is available, Planner v2 uses a
documented conservative 4096-token compatibility fallback.

Before each request, Planner v2 tokenizes the exact rendered chat request and
emits `planning.context.preflight`. The event records section counts, context
source, safety margin, requested output, safe minimum, and effective output.
No request is sent below its phase minimum.

The configuration order is:

1. phase values under `planner.context`;
2. legacy `planner.finalization.budgets` and `minimum_tokens`, when present;
3. compatibility defaults derived from `planner.max_tokens`.

## Evidence Packs

Full bounded observations remain in exploration events. Later model calls use
deterministic `EvidencePack` objects containing observation IDs, normalized
paths, digests, exact line-numbered spans, omitted ranges, and selection
reasons. File role, recent successful reads, search/listing results, and
filtered task terms drive selection. No embedding or additional model call is
used.

Four finite compaction levels progressively remove lower-priority evidence and
reduce span context. Each level is rendered and tokenized before selection.
Evidence events contain IDs, ranges, digests, and truncation metadata, not a
second copy of file contents.

Exploration, final generation, review, revision, final review, and recovery
use phase-specific instructions and schemas. The verbatim original task occurs
once in every stateless request.

## Format Recovery

Malformed structured output receives exactly one recovery attempt for that
step. Recovery uses a fresh model session and requests a complete replacement
JSON object. AgentCore never appends to, guesses, parses actions from, or
executes the malformed fragment. A second malformed response fails planning.
Malformed output retained in events is bounded. The recovery prompt contains
only the parse diagnostic, SHA-256 digest, and independently bounded prefix
and suffix excerpts. It never carries the complete malformed action payload.

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
