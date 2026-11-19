"""Opt-in, PAID remote-model acceptance run. Never collected by pytest.

Run explicitly: uv run python scripts/manual_integration.py
"""

from pathlib import Path
import shutil
import sys
import tempfile
import uuid

from research_agent.agent import ConfigurationError, load_model
from research_agent.main import AgentRunError, normalize_evidence, run_agent
from research_agent.tools.experiment import make_analyze_experiment, project_root


def run_checks() -> None:
    root = project_root()
    model = load_model(root)  # Explicit invocation permits a real paid API call.
    output = f"reports/manual_{uuid.uuid4().hex[:12]}.md"
    result = run_agent(root=root, model=model, output=output, trace=print)
    expected = make_analyze_experiment(root).invoke({
        "csv_path": "data/sample_experiments.csv", "function_name": "F1",
    })
    assert result["analysis"] == expected, "Remote tool evidence differs from independent recomputation"
    report = Path(result["report_path"])
    assert normalize_evidence(expected["evidence_markdown"]) in normalize_evidence(report.read_text(encoding="utf-8"))
    print(result["summary"])
    print(f"PASS: real tool calls, numeric evidence, disk file, final paths: {report}")

    # An ordinary file named reports prevents any report from being written.
    # This exercises the real built-in write_file failure path, without chmod
    # assumptions (which are unreliable for privileged users).
    with tempfile.TemporaryDirectory(prefix="research-agent-write-failure-") as directory:
        failure_root = Path(directory)
        (failure_root / "data").mkdir()
        shutil.copyfile(root / "data/sample_experiments.csv", failure_root / "data/sample_experiments.csv")
        (failure_root / "reports").write_text("intentionally blocks directory creation", encoding="utf-8")
        events = []

        def trace(event: str) -> None:
            events.append(event)
            print(event)

        try:
            run_agent(root=failure_root, model=model, output="reports/forced_failure.md", trace=trace)
        except AgentRunError:
            assert any("tool_result write_file" in e and "status=error" in e for e in events), "No real write_file failure observed"
            assert not (failure_root / "reports/forced_failure.md").exists()
            print("PASS: real write_file failure rejected by acceptance checks; no success result returned")
        else:
            raise AssertionError("Write failure incorrectly accepted")
    print("Remote protocol checks passed. Review the report's prose against the tables manually.")


def main() -> int:
    try:
        run_checks()
    except (ConfigurationError, AgentRunError, AssertionError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"FAIL: remote integration error ({type(exc).__name__}); no raw API response logged", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
