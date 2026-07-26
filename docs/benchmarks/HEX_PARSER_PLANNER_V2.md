# Planner v2 Hexadecimal Parser Benchmark

This is a durable record of a deliberately constructed correctness benchmark.
It is not a general performance benchmark.

## Versions

- AgentCore: `fbfed1c0372f5d8e2cd4a04319b4c3fb6b577f46`
- Fixture baseline: `bf5c8d93d6078d15b1f21a4367eb381bee023b92`
- Result patch SHA-256:
  `c7add48815bdff19137511f511043dfe4309067afb8dea4256b76047c9c7bf87`
- External execution trace SHA-256:
  `ad3a16c279411ea39bbd63bb4d7e6a3bf1db41f3edd441ae87d83ba7b03dab36`

The trace remains an external artifact because it contains verbose runtime
output. It contains 1,311 ordered public events and no hidden chain-of-thought.

## Exact Prompt

```text
Locate the implementation of the hexadecimal integer parser.

The parser currently rejects valid uppercase hexadecimal literals.

Modify the implementation so that both

    0x1a
    0x1A

are accepted.

Keep the existing behaviour for decimal numbers.

Do not change unrelated code.

Before editing:

- inspect the parser;
- inspect the existing tests;
- explain the bug;
- propose a minimal ActionPlan;
- wait for explicit approval.

After approval:

- implement the fix;
- update or add tests only if required;
- run the existing test suite;
- produce the final Git diff;
- produce the TaskReport.

Do not create a Git commit.
```

## Planning And Approval

Planner v2 performed bounded, read-only exploration. It listed the source,
test, and include directories, then inspected `src/parser.c`,
`tests/test_parser.c`, and `include/parser.h`. It identified that the parser
handled `a-f` but not `A-F`, while the existing tests already contained the
uppercase regression cases.

The final proposal replaced one exact fragment in `src/parser.c`, ran the
trusted symbolic `test` check, captured the Git diff, and requested a task
report. The proposal required approval for both the mutation and the configured
check. No action ran before the operator explicitly approved it.

## Result

The functional diff added two lines: an uppercase `A-F` branch mapping those
characters to values 10 through 15. No header or test change was necessary.
The configured `make test` check passed, and AgentCore did not create a Git
commit.

Independent review reproduced the baseline failure and applied only the
generated patch in an isolated copy. It verified:

- decimal and lowercase hexadecimal behavior;
- uppercase and mixed-case hexadecimal behavior;
- malformed-input and trailing-garbage rejection;
- decimal and hexadecimal range boundaries;
- a clean build and passing test suite.

The independent verdict was **PASS**, with acceptance **YES**.

## Known Limitations At Benchmark Time

- Supplying `--prompt-file` selected non-interactive mode, so interactive
  approval required a wrapper. The benchmark-hardening change makes a supplied
  prompt seed the normal interactive command loop.
- A legacy executable `task_report` action captured a pre-terminal
  `status=running` snapshot. It was accurate but was not explicitly labeled as
  intermediate. The authoritative report emitted after execution was complete.
