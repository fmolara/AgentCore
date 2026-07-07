from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from a100_agent_lab import AgentLab


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demonstrate safe file editing in an AgentCore workspace.")
    parser.add_argument("--config", default="config/sglang-a100.yaml", help="AgentLab YAML configuration path.")
    parser.add_argument(
        "--workspace",
        default=str(ROOT / "workspace" / "edit-demo"),
        help="Workspace root for the demo.",
    )
    return parser.parse_args()


def ensure_demo_git_identity() -> None:
    os.environ.setdefault("GIT_AUTHOR_NAME", "AgentCore Demo")
    os.environ.setdefault("GIT_AUTHOR_EMAIL", "agentcore-demo@example.invalid")
    os.environ.setdefault("GIT_COMMITTER_NAME", "AgentCore Demo")
    os.environ.setdefault("GIT_COMMITTER_EMAIL", "agentcore-demo@example.invalid")


def main() -> None:
    args = parse_args()
    ensure_demo_git_identity()

    lab = AgentLab.from_config(args.config)
    agent = lab.create_agent(
        system_prompt="You are a concise coding assistant.",
        workspace_root=args.workspace,
        workspace_metadata={"purpose": "file editing demo"},
    )

    if not agent.git.is_repo():
        agent.git.init()

    initial = (
        "#include <stdio.h>\n"
        "\n"
        "int answer(void) {\n"
        "    return 1;\n"
        "}\n"
        "\n"
        "int main(void) {\n"
        "    printf(\"%d\\n\", answer());\n"
        "    return 0;\n"
        "}\n"
    )
    agent.files.write_text("main.c", initial)
    if agent.git.status().stdout.strip():
        agent.git.add(["main.c"])
        agent.git.commit("Add initial demo C file")

    replace = agent.files.replace_text(
        "main.c",
        "int answer(void) {\n    return 1;\n}",
        "int answer(void) {\n    return 41 + 1;\n}",
    )

    diff_text = """--- a/main.c
+++ b/main.c
@@ -6,5 +6,6 @@

 int main(void) {
+    puts("AgentCore safe editing demo");
     printf("%d\\n", answer());
     return 0;
 }
"""
    patch = agent.files.apply_unified_diff(diff_text)

    print("replace result:")
    print(replace)
    print()
    print("patch result:")
    print(patch)
    print()
    print("git diff:")
    print(agent.git.diff().stdout.strip())


if __name__ == "__main__":
    main()
