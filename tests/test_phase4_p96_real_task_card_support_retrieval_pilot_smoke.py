# tests/test_phase4_p96_real_task_card_support_retrieval_pilot_smoke.py

import subprocess
import sys
import json
import os


def test_p96_smoke_40_runner_no_args():
    """Verify smoke runner executes correctly with no args."""
    result = subprocess.run(
        [sys.executable, "tools/phase4/run_p96_real_task_card_support_retrieval_pilot_smoke.py"],
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
    )
    try:
        out = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise AssertionError(f"Smoke runner output is not JSON:\n{result.stdout[:500]}\nstderr: {result.stderr[:500]}")
    assert out["phase"] == "P96"
    assert out["verdict"] == "P96_READY_FOR_REVIEW"
    assert result.returncode == 0


def test_p96_smoke_41_runner_rejects_args():
    """Verify smoke runner rejects CLI arguments."""
    result = subprocess.run(
        [sys.executable, "tools/phase4/run_p96_real_task_card_support_retrieval_pilot_smoke.py", "--unexpected"],
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
    )
    assert result.returncode != 0
