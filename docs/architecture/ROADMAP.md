# Roadmap

This roadmap records architectural milestones only. It is not an implementation
plan and should not be treated as a sprint backlog.

## Milestone 1: Core Platform

Status: COMPLETE

Goal:

- establish the runtime-independent AgentCore foundation;
- support persistent sessions;
- support `AgentLab`, `Agent`, and `Session` as the normal public API;
- validate Transformers, SGLang, and LMDeploy backends;
- provide health, warmup, benchmarks, structured logging, and contract tests.

Milestone 1 is the architectural baseline for future work.

## Milestone 2: Coding Workspace

Status: COMPLETE

Goal:

- provide a managed workspace boundary;
- add safe file editing APIs;
- add constrained local Git workflows;
- introduce tasks, reports, checkpoints, restore planning, and restore execution;
- support explicit ActionPlans, PlanProposals, and ApprovalPolicy;
- execute approved plans through TaskExecutor;
- stream structured operational events and visible assistant output;
- support cooperative cancellation.

Milestone 2 is the architectural baseline for coding workspace workflows. It
remains non-autonomous: proposals require review and mutating plans require
explicit approval.

## Milestone 3: Incremental Coding Agent

Status: COMPLETE

Goal:

- provide a persistent native `ToolLoopAgent` over Qwen, Mistral, and Harmony;
- expose bounded workspace discovery and exact localized edits;
- bind complete effect previews to per-call approval;
- return failures and check results to the model for bounded recovery;
- retain legacy planner behavior for compatibility.

The normal architecture is native incremental tool use, not complete up-front
ActionPlan generation. Distributed ToolLoopAgent exposure remains deferred.

## Milestone 4: Daily Local Operation

Status: ACTIVE

Goal:

- use Qwen3.6/SGLang as the fast default local profile;
- retain gpt-oss/vLLM/Harmony as the optional strong experimental profile;
- collect privacy-bounded passive task metrics and complete traces;
- validate daily work while retaining explicit human approval.
