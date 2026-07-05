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

## Milestone 2: External Tools

Goal:

- introduce a controlled tool abstraction;
- add filesystem access through explicit, reviewable APIs;
- add Git operations through explicit, reviewable APIs;
- define tool result schemas;
- keep tool execution observable and auditable.

No model-native function calling policy is assumed by this milestone.

## Milestone 3: Planning & Memory

Goal:

- represent task plans;
- track short-term working memory;
- track project/session memory;
- define what is stored, when it is updated, and how it is surfaced to agents;
- keep memory behavior inspectable and resettable.

## Milestone 4: Coding Agent

Goal:

- combine runtime, sessions, tools, planning, and memory into a practical
  coding-agent workflow;
- support code inspection, edits, tests, and Git-aware review flows;
- provide safe defaults for persistent developer use;
- preserve runtime independence where possible.
