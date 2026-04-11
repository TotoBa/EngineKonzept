from __future__ import annotations

import subprocess
import sys

def test_arena_module_imports_without_selfplay_cycle() -> None:
    repo_root = __file__.split("/python/tests/", 1)[0]
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                f"sys.path.insert(0, {repo_root!r} + '/python'); "
                "from train.eval.arena import write_selfplay_arena_spec; "
                "print(callable(write_selfplay_arena_spec))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert process.stdout.strip() == "True"
