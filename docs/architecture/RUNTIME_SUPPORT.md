# Runtime Support

AgentCore is runtime-independent at the application layer. Runtime adapters are
allowed to contain backend-specific lifecycle and protocol details, but user
code should not need to change when switching backends through configuration.

Current model used for validation:

- `models/Qwen3.6-27B`
- NVIDIA A100 80GB PCIe
- BF16
- persistent chat/coding-agent style prompts

Numbers below are local reference measurements, not universal claims.

## Summary

| Runtime | Role | Maturity | Typical Generation |
| --- | --- | --- | --- |
| SGLang | Recommended production backend | High | ~27-28 tokens/s |
| LMDeploy | Secondary production backend | Good | ~28 tokens/s |
| Transformers | Reference backend | Stable but slower | ~15 tokens/s |

## Transformers

Current maturity:

- implemented and working;
- in-process runtime;
- simplest reference implementation;
- useful for correctness checks and API development.

Strengths:

- minimal operational surface;
- no server process;
- easy debugging inside Python;
- good reference for tokenizer and model behavior;
- useful when isolating platform bugs from server-runtime behavior.

Weaknesses:

- slower generation than production server runtimes on the current A100 setup;
- startup involves direct model load inside the application process;
- less suitable as the primary production serving backend;
- memory and runtime lifecycle are tied to the application process.

Benchmark summary:

- model load around 14-16 seconds in local tests;
- warm TTFT around 0.3-0.5 seconds;
- generation around 15 tokens/sec;
- resident VRAM around 52 GiB.

Operational notes:

- requires the AgentCore Python environment;
- no subprocess management;
- health reports in-process model readiness and Torch memory details;
- good fallback and reference runtime.

## SGLang

Current maturity:

- implemented and working;
- production-oriented server runtime;
- selected as the current recommended backend.

Strengths:

- strong throughput on Qwen3.6-27B on A100 PCIe;
- OpenAI-compatible server API;
- explicit process lifecycle managed by AgentCore;
- good warm steady-state latency;
- practical foundation for persistent agent/server workloads.

Weaknesses:

- operationally more complex than Transformers;
- startup is slower than simple in-process health checks;
- requires correct environment/PATH handling for the existing SGLang install;
- backend logs must be inspected when startup fails.

Benchmark summary:

- server ready time around 30-31 seconds;
- post-warmup TTFT commonly around 0.2-0.4 seconds;
- persistent 10-turn generation around 27-28 tokens/sec;
- 10-turn wall time around 26-27 seconds in local tests.

Operational notes:

- current config expects `python` to resolve to an environment with SGLang
  installed;
- set `server.path_prefix` locally if the SGLang executable environment is not
  already on `PATH`;
- server logs are written under `experiments/logs/`;
- health includes endpoint, PID, ready time, GPU memory, and last error.

## LMDeploy

Current maturity:

- implemented and working;
- second production backend;
- validates that the AgentCore API is not tied to SGLang.

Strengths:

- comparable generation throughput to SGLang in local tests;
- fast warmup;
- production-oriented server deployment model;
- useful secondary backend for architecture validation and comparison.

Weaknesses:

- operational behavior and logs differ from SGLang;
- fewer platform-level assumptions have been exercised so far;
- adapter still relies on OpenAI-compatible chat behavior;
- not currently the primary target for future agent-server policy work.

Benchmark summary:

- server ready time around 18-19 seconds;
- warmup around 0.17 seconds in local tests;
- persistent 10-turn generation around 28 tokens/sec;
- 10-turn wall time around 24-25 seconds in local tests;
- resident VRAM around 58-59 GiB in the tested configuration.

Operational notes:

- current config expects `lmdeploy` to be available on `PATH`;
- health includes endpoint, PID, ready time, GPU memory, utilization, power,
  and last error;
- useful as a production-capable alternate backend.

## Why SGLang Is Recommended Today

SGLang is the current recommended production backend because it provides the
best balance of:

- production-oriented serving model;
- strong throughput;
- low warm TTFT;
- stable OpenAI-compatible interface;
- manageable operational complexity;
- alignment with persistent coding-agent/server use.

LMDeploy remains important as the secondary backend. Transformers remains the
reference backend. Future decisions should continue to be measurement-driven.
