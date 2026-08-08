# Daily Local Operation

AgentCore's normal coding path is the local DS4-style `ToolLoopAgent`. Qwen3.6
is the default fast profile. gpt-oss is an explicit strong experimental profile.
There is no automatic routing between them.

## Install Profiles

Create local profile files outside tracked source:

```bash
mkdir -p ~/.config/agentcore/profiles
cp config/sglang-qwen-tools.yaml ~/.config/agentcore/profiles/fast.yaml
cp config/vllm-harmony-tools.yaml ~/.config/agentcore/profiles/strong.yaml
chmod 600 ~/.config/agentcore/profiles/*.yaml
```

Set the real model path, runtime Python or `path_prefix`, port, context limit,
and trusted workspace checks in each file. The checked-in configurations are
portable examples and intentionally contain no machine-specific absolute path.
An alternative profile directory can be selected with
`AGENTCORE_PROFILE_DIR`. A repository-local `config/local/fast.yaml` or
`config/local/strong.yaml` is also recognized and is ignored by Git.

## Run A Task

The default invocation uses `fast`:

```bash
agentcore-local \
  --agent tool-loop \
  --workspace /absolute/path/to/project \
  --prompt-file /absolute/path/to/task.txt
```

Select gpt-oss only when the additional reasoning capacity justifies its
latency and near-dedicated GPU footprint:

```bash
agentcore-local \
  --agent tool-loop \
  --profile strong \
  --workspace /absolute/path/to/project \
  --prompt-file /absolute/path/to/task.txt
```

AgentCore starts the configured inference runtime, performs warmup, and shuts
it down when the local task exits. Only one GPU-heavy profile should run on the
single A100 at a time.

## Approval And Recovery

Read-only tools execute under policy. Every `edit`, `write_file`, and
`run_check` pauses with a complete AgentCore-generated preview. Review the
target, complete effect, tool-call ID, and preview digest before `/approve`.
Use `/reject reason` for an unsafe or incomplete call; rejection does not
mutate and is returned to the model for bounded recovery. `/preview` reopens
the retained complete artifact.

The model has no arbitrary shell, network, package installation, Git commit,
or Git push tool. Checks are symbolic names mapped to trusted argument vectors
in the selected profile. AgentCore never commits automatically.

## Passive Metrics And Artifacts

`logging.trace_path` enables an automatic unique JSONL trace when no explicit
`--trace-file` is supplied. `logging.metrics_path` receives one append-only JSON
record after every terminal tool-loop task. The record includes:

- profile, model, runtime, protocol, status, and trace path;
- model turns, tool calls, approvals, rejections, and check outcomes;
- prompt/generated token totals, peak prompt size, TTFT, and generation/wall time;
- changed-file names, diff byte count, and final-response presence.

Metrics deliberately exclude task prompts, source content, tool arguments,
diff content, environment values, credentials, and hidden reasoning. Runtime
generation logs remain separately configured through `logging.path`.

## Legacy Planner

ActionPlan and PlanProposal remain compatibility-only. They require an explicit
configuration and planner selection:

```bash
agentcore-local \
  --config config/sglang-a100-iterative.yaml \
  --planner iterative \
  --workspace /absolute/path/to/project \
  --prompt-file /absolute/path/to/task.txt \
  --proposal-only
```

New coding work should use `--agent tool-loop`.
