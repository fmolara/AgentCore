# DS4-Style Tool Agent Milestone

## Decision

AgentCore's normal local coding-agent architecture is:

```text
ToolLoopAgent
  -> QwenToolProtocol | MistralToolProtocol | HarmonyToolProtocol
  -> Workspace / Approval / run_check / Git
  -> Task / TaskReport / Event trace
```

The model works incrementally in one persistent native transcript. It inspects
the workspace, requests concrete tools, receives their results, corrects failed
work, runs checks, inspects Git diff, and writes its own final response. This
replaces complete up-front ActionPlan generation as the recommended workflow.
ActionPlan, PlanProposal, TaskExecutor, and planner CLI/API behavior remain
available for legacy compatibility.

The behavior is inspired by antirez/ds4's incremental native agent. AgentCore
does not copy DS4 source code and is not affiliated with or endorsed by
antirez. Deliberate differences include model-native OpenAI-compatible
protocols, symbolic checks instead of arbitrary bash, complete per-call
approval, workspace confinement, and JSONL audit traces.

## Native Protocols

- Qwen uses SGLang's structured Qwen tool parser.
- Mistral uses native Mistral function calling through vLLM.
- gpt-oss uses OpenAI Harmony through vLLM; hidden reasoning is excluded from
  visible assistant output and public traces.

All adapters share one tool registry and loop. Adapters own only transcript
roles, schema encoding, stream decoding, tool-call assembly, tool-result
encoding, and protocol-specific request options.

## Safety Guarantees

- Paths stay inside the configured workspace and symlink escapes are blocked.
- Existing-file changes use exact unique replacement; zero or multiple matches
  do not mutate.
- Every edit, write, and symbolic check has a complete pre-mutation preview.
- Approval binds one tool-call ID to one preview digest and is never reused.
- A preview is recomputed before execution; stale effects are rejected.
- Checks use trusted argv, `shell=False`, a workspace cwd, timeout, bounded
  output, and controlled environment.
- No model-visible arbitrary shell, network, commit, or push tool exists.
- Rejections and ordinary failures are returned to the model for bounded
  recovery.
- AgentCore produces one authoritative terminal `TaskReport` and never commits
  automatically.

## Model Qualification

The frozen ten-task daily-coding suite covered C bugs and features, error
handling, CMake, test generation, refactoring, validation, failed-check
recovery, Python tooling, and a historical public defect.

- **gpt-oss-120b:** 9/10 tasks on the full suite; primary experimental model
  for controlled daily coding trials and the strong local candidate for broad
  or reasoning-heavy work.
- **Qwen3.6-27B:** 5/5 tasks on a preselected representative subset; fast,
  lower-latency fallback.
- **Qwen3-Coder-30B-A3B-Instruct:** evaluated but not qualified as a primary
  model.
- **Devstral Small 2 24B:** evaluated but not qualified as a primary model.

These are milestone qualification results, not universal success rates. Every
accepted patch was independently applied, built, tested, and reviewed without
manual repair. Complete approval remained in the loop.

## Benchmark And Operation

The durable task definition is
[`benchmarks/daily-coding/manifest.yaml`](../../benchmarks/daily-coding/manifest.yaml).
It contains no machine-specific paths, credentials, transient workspaces, or
reference diffs exposed to models. Runtime logs, model files, disposable
fixtures, and qualification artifacts are not committed.

Use `agentcore-local --agent tool-loop` with an ignored deployment config.
Public strong and fast examples are `config/vllm-harmony-tools.yaml` and
`config/sglang-qwen-tools.yaml`.

## Known Limitations

- Native ToolLoopAgent operation is local-first; distributed HTTP exposure is
  deferred.
- Normal text transcripts are resent rather than persisted as native KV cache.
- Model correctness remains review-dependent, especially for subtle semantic
  and portability requirements.
- gpt-oss-120b requires nearly a dedicated A100 80 GB and has higher first-token
  latency; Qwen3.6 is faster but weaker on some portability-heavy reasoning.
- Planner v2 finalization and planner context-compaction branches are retained
  as research only and are intentionally excluded from the product milestone.
