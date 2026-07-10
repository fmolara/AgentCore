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

## Milestone 3: Autonomous Coding Agent

Status: PLANNED

Goal:

- evolve from reviewable execution toward controlled autonomous coding work.

High-level areas:

- planner improvements;
- long-running tasks;
- tool registry;
- memory;
- workspace reasoning;
- multi-agent collaboration.

No implementation sequence is defined here.

## Milestone 4: Coding Agent

Goal:

- combine runtime, sessions, tools, planning, and memory into a practical
  coding-agent workflow;
- support code inspection, edits, tests, and Git-aware review flows;
- provide safe defaults for persistent developer use;
- preserve runtime independence where possible.
