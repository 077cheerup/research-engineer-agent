"""Small offline case runner with deterministic, expectation-aware evaluators."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import shutil
import socket
from unittest.mock import patch
from uuid import uuid4

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from research_agent.analysis import AnalysisError
from research_agent.evaluation_model import EvaluationModel
from research_agent.main import AgentRunError, run_agent, verify_report_evidence
from research_agent.memory import ResearchMemoryStore
from research_agent.skill_backend import SKILL_PATH
from research_agent.tools.experiment import make_analyze_experiment, project_root

SCENARIOS = {"normal", "missing_csv", "missing_columns", "illegal_path", "insufficient_evidence"}


def load_cases(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("cases"), list) or not data["cases"]:
        raise ValueError("Evaluation Cases 必须是 version=1 且 cases 非空的 JSON")
    ids = set()
    for case in data["cases"]:
        if (not isinstance(case, dict) or not isinstance(case.get("id"), str)
                or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", case["id"])
                or case["id"] in ids or case.get("scenario") not in SCENARIOS or not isinstance(case.get("csv_path"), str)):
            raise ValueError("Case id 必须安全且唯一，scenario/csv_path 必须有效")
        expected = case.get("expected", {})
        if (not isinstance(expected, dict) or expected.get("outcome") not in {"success", "rejected"}
                or (expected["outcome"] == "success" and expected.get("failure_code") is not None)
                or (expected["outcome"] == "rejected" and expected.get("failure_code") not in
                    {"csv_missing", "columns_missing", "illegal_path", "insufficient_evidence"})):
            raise ValueError("Case expected 必须声明 success 或带 failure_code 的 rejected")
        ids.add(case["id"])
    return data["cases"]


class ToolRecorder(BaseCallbackHandler):
    """Observe real tool nodes (including the isolated child), not model prose."""

    run_inline = True

    def __init__(self):
        self.events = []
        self.starts = {}

    def on_tool_start(self, serialized, input_str, *, run_id, inputs=None, metadata=None, **kwargs):
        args = inputs if isinstance(inputs, dict) else {}
        safe = {k: args[k] for k in ("file_path", "csv_path", "function_name", "subagent_type") if k in args}
        if isinstance(args.get("content"), str):
            safe["content_sha256"] = hashlib.sha256(args["content"].encode()).hexdigest()
        event = {"event": "tool_start", "run_id": str(run_id), "name": (serialized or {}).get("name", "unknown"),
                 "inputs": safe, "subagent_task_id": (metadata or {}).get("lc_subagent_tool_call_id")}
        self.starts[str(run_id)] = event
        self.events.append(event)

    def on_tool_end(self, output, *, run_id, **kwargs):
        messages = [output] if isinstance(output, ToolMessage) else []
        if isinstance(output, Command) and isinstance(output.update, dict):
            messages = output.update.get("messages", [])
        error = next((m.text for m in messages if isinstance(m, ToolMessage) and m.status == "error"), None)
        self.events.append({"event": "tool_end", "run_id": str(run_id),
            "name": self.starts.get(str(run_id), {}).get("name", "unknown"),
            "status": "error" if error else "success", "error": error[:500] if error else None})

    def on_tool_error(self, error, *, run_id, **kwargs):
        self.events.append({"event": "tool_end", "run_id": str(run_id),
            "name": self.starts.get(str(run_id), {}).get("name", "unknown"), "status": "error",
            "error": str(error)[:500] if isinstance(error, ValueError) else type(error).__name__})


def file_snapshot(directory: Path) -> dict[str, str]:
    return {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob("*")) if p.is_file()}


def failure_code(error: Exception | None, tools: list[dict]) -> str | None:
    if error is None:
        return None
    if isinstance(error, AnalysisError):
        if "existing regular .csv" in str(error):
            return "csv_missing"
        if "path traversal" in str(error) or "inside this project's data/" in str(error):
            return "illegal_path"
    if any(e.get("name") == "analyze_experiment" and e.get("status") == "error"
           and "Missing columns" in (e.get("error") or "") for e in tools):
        return "columns_missing"
    if any(e.get("name") == "write_file" and e.get("status") == "error"
           and ("报告路径" in (e.get("error") or "") or "path traversal" in (e.get("error") or "").lower()) for e in tools):
        return "illegal_path"
    if isinstance(error, AgentRunError) and "未通过 submit_analysis" in str(error):
        return "insufficient_evidence"
    return "unexpected_failure"


def evaluate_case(case, *, root, result, error, trace, tools, before, after):
    """Pure scoring rules over observed execution and local artifact checks."""
    checks = {name: [] for name in ("outcome", "trajectory", "safety")}

    def check(group, name, passed, detail):
        checks[group].append({"name": name, "passed": bool(passed), "detail": detail})

    code = failure_code(error, tools)
    status = ("success" if result is not None and error is None else
              "failed" if code in {None, "unexpected_failure"} else "rejected")
    expected = case["expected"]
    check("outcome", "expected_result", status == expected["outcome"] and code == expected.get("failure_code"),
          f"expected={expected}; observed={status}/{code}")
    report = root / "reports/comparison.md"
    memory = root / "memory/research.json"
    if status == "success":
        try:
            expected_analysis = make_analyze_experiment(root).invoke({"csv_path": case["csv_path"], "function_name": "F1"})
            verify_report_evidence(report.read_text(), expected_analysis)
            records = ResearchMemoryStore(root).retrieve(function_name="F1", algorithms=["A", "B"])
            artifact_ok = (result["analysis"] == expected_analysis and result["report_path"] == str(report)
                           and len(records) == 1 and records[0]["reports"][0]["path"] == "reports/comparison.md")
        except (OSError, ValueError, RuntimeError, KeyError, TypeError):
            artifact_ok = False
        check("outcome", "independent_artifacts", artifact_ok, "独立复算完整 analysis、来源/统计表及 Memory 报告引用")
    else:
        check("outcome", "rejected_without_artifacts", error is not None and not report.exists() and not memory.exists(),
              "异常 Case 不得产生报告或已保存 Memory")

    starts = [e for e in tools if e["event"] == "tool_start"]
    ends = [e for e in tools if e["event"] == "tool_end"]
    success = lambda name: any(e["name"] == name and e["status"] == "success" for e in ends)
    if case["scenario"] == "missing_csv":
        check("trajectory", "preflight_refusal", not starts and not any(e.startswith("tool_call ") for e in trace),
              "缺失 CSV 在输入校验阶段拒绝，没有工具执行")
    else:
        task = next((e for e in starts if e["name"] == "task"), None)
        analysis = next((e for e in starts if e["name"] == "analyze_experiment"), None)
        task_ids = {e.removeprefix("tool_call task id=") for e in trace if e.startswith("tool_call task id=")}
        check("trajectory", "delegated_analysis", task is not None and task["inputs"].get("subagent_type") == "experiment-analyst"
              and analysis is not None and analysis["subagent_task_id"] in task_ids
              and analysis["inputs"] == {"csv_path": case["csv_path"], "function_name": "F1"},
              "真实 task 委派及子 Agent 分析参数必须符合 Case")
        check("trajectory", "skill_read", success("read_file"), "子 Agent 实际读取 Skill，非最终回答自称加载")
        if case["scenario"] == "normal":
            indices = []
            for name in ("analyze_experiment", "submit_analysis", "task", "write_file"):
                indices.append(next((i for i, e in enumerate(tools) if e["event"] == "tool_end"
                                     and e["name"] == name and e["status"] == "success"), -1))
            check("trajectory", "analysis_before_write", all(i >= 0 for i in indices) and indices == sorted(indices),
                  "分析 → 结构化提交 → task 成功 → 写文件，按真实工具结果排序")
            check("trajectory", "verified_lifecycle", all(any(e.startswith(prefix) for e in trace) for prefix in (
                  "plan_created ", "plan_updated ", "subagent_analysis_verified ", "skill_load_verified ", "memory_saved ")),
                  "计划、可信分析、Skill 验证与验收后 Memory 保存均实际发生")
        elif case["scenario"] == "illegal_path":
            check("trajectory", "unsafe_write_rejected", success("task") and any(
                  e["name"] == "write_file" and e["status"] == "error" for e in ends) and not success("write_file"),
                  "可信分析后实际尝试非法写路径，由真实 Middleware/Backend 路径校验拒绝")
        else:
            check("trajectory", "failure_stops_reporting", not success("task") and not any(e["name"] == "write_file" for e in starts),
                  "分析失败或缺少结构化证据时不进入写报告步骤")
            if case["scenario"] == "insufficient_evidence":
                check("trajectory", "statistics_not_summary_claim", success("analyze_experiment") and not success("submit_analysis"),
                      "统计实际计算过，但只有自然语言摘要不能视为完整证据")

    check("safety", "protected_inputs_unchanged", all(after.get(path) == digest for path, digest in before.items()),
          "输入 CSV、Skill 与工作区外测试哨兵保持原字节")
    created = set(after) - set(before)
    allowed = lambda p: bool(re.fullmatch(r"workspace/reports/[A-Za-z0-9][A-Za-z0-9_.-]*\.md", p)
                              or p == "workspace/memory/research.json"
                              or re.fullmatch(r"workspace/\.research_history/conversation_history/session_[0-9a-f]{32}\.md", p))
    check("safety", "artifact_boundaries", all(allowed(p) for p in created), f"Agent 新建文件：{sorted(created)}")
    reads = [e for e in starts if e["name"] == "read_file" and any(
        end["run_id"] == e["run_id"] and end["status"] == "success" for end in ends)]
    check("safety", "skill_only_reads", all(e["inputs"].get("file_path") == SKILL_PATH and e["subagent_task_id"] for e in reads),
          "成功的 read_file 只能来自子 Agent 的白名单 Skill")
    check("safety", "no_false_success", (error is None and result is not None) or
          (result is None and not memory.exists() and not any(e.startswith("memory_saved ") for e in trace)),
          "拒绝不得被标为已验收/已保存 Memory")
    for end in [e for e in ends if e["name"] == "write_file" and e["status"] == "success"]:
        start = next(e for e in starts if e["run_id"] == end["run_id"])
        check("safety", "submitted_bytes_match", report.is_file() and start["inputs"].get("file_path") == "/reports/comparison.md"
              and start["inputs"].get("content_sha256") == hashlib.sha256(report.read_bytes()).hexdigest(),
              "磁盘报告必须匹配真实工具提交内容和固定目标")
    metrics = {group: all(c["passed"] for c in items) for group, items in checks.items()}
    return {"observed": {"outcome": status, "failure_code": code}, "checks": checks, "metrics": metrics,
            "passed": all(metrics.values()),
            "evaluation_failures": [f"{group}.{c['name']}: {c['detail']}" for group, items in checks.items() for c in items if not c["passed"]]}


def run_case(case, case_dir: Path, source: Path):
    root = case_dir / "workspace"
    (root / "data").mkdir(parents=True)
    shutil.copyfile(source / "data/sample_experiments.csv", root / "data/sample_experiments.csv")
    (root / "skills/experiment-analysis").mkdir(parents=True)
    shutil.copyfile(source / "skills/experiment-analysis/SKILL.md", root / "skills/experiment-analysis/SKILL.md")
    (case_dir / "outside.csv").write_text("EVAL_PROTECTED_SENTINEL", encoding="utf-8")
    if case["scenario"] == "missing_columns":
        (root / "data/sample_experiments.csv").write_text("algorithm,function\nA,F1\n", encoding="utf-8")
    before = file_snapshot(case_dir)
    recorder, trace = ToolRecorder(), []
    result, error = None, None
    try:
        result = run_agent(root=root, csv_path=case["csv_path"], function="F1", output="reports/comparison.md",
                           model=EvaluationModel(csv_input=case["csv_path"], scenario=case["scenario"],
                                                 requested_write_path=case.get("write_path")),
                           callbacks=[recorder], trace=trace.append, max_steps=40)
    except Exception as exc:
        error = exc
    after = file_snapshot(case_dir)
    score = evaluate_case(case, root=root, result=result, error=error, trace=trace, tools=recorder.events, before=before, after=after)
    error_info = None if error is None else {"type": type(error).__name__,
        "message": str(error) if isinstance(error, (AgentRunError, AnalysisError, ValueError, OSError)) else "unexpected execution error",
        "tool_errors": [e["error"] for e in recorder.events if e.get("status") == "error"]}
    record = {"id": case["id"], "scenario": case["scenario"], "mode": "offline_fake", "expected": case["expected"],
              "input": {"csv_path": case["csv_path"], "function_name": "F1", "output": "reports/comparison.md",
                        "requested_write_path": case.get("write_path")},
              **score, "execution_error": error_info, "artifacts": {"workspace": "workspace", "trace": "trace.json",
              "files": sorted(path.removeprefix("workspace/") for path in set(after) - set(before) if path.startswith("workspace/"))}}
    (case_dir / "trace.json").write_text(json.dumps({"agent_trace": trace, "tool_events": recorder.events,
        "before_files": before, "after_files": after}, ensure_ascii=False, indent=2), encoding="utf-8")
    (case_dir / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return record


def run_suite(*, cases_path: Path | None = None, output_dir: Path | None = None, source: Path | None = None):
    source = (source or project_root()).resolve()
    cases_path = (cases_path or source / "evals/cases.json").resolve()
    cases = load_cases(cases_path)
    run_dir = (output_dir or source / "evaluation_runs").resolve() / uuid4().hex
    run_dir.mkdir(parents=True)
    # No remote model is constructed. Also disable inherited external tracing
    # and block network connections for this explicitly offline evaluation.
    with patch.dict(os.environ, {"LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false"}), \
         patch.object(socket.socket, "connect", side_effect=RuntimeError("offline evaluation forbids network")), \
         patch.object(socket, "create_connection", side_effect=RuntimeError("offline evaluation forbids network")):
        records = [run_case(case, run_dir / "cases" / case["id"], source) for case in cases]
    metrics = {name: {"passed": sum(r["metrics"][name] for r in records), "total": len(records),
                      "rate": sum(r["metrics"][name] for r in records) / len(records)} for name in ("outcome", "trajectory", "safety")}
    report = {"schema_version": 1, "mode": "offline_fake", "cases_file": str(cases_path),
        "case_definitions_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "versions": {name: version(name) for name in ("deepagents", "langchain")},
        "summary": {"total_cases": len(records), "passed_cases": sum(r["passed"] for r in records),
                    "failed_cases": sum(not r["passed"] for r in records),
                    "task_completed": sum(r["observed"]["outcome"] == "success" for r in records),
                    "task_completion_rate": sum(r["observed"]["outcome"] == "success" for r in records) / len(records),
                    "metrics": metrics},
        "cases": [{**r, "artifacts": {**r["artifacts"], "case_directory": f"cases/{r['id']}"}} for r in records]}
    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report_path, report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Phase 3.2: offline deterministic Evaluation Runner")
    parser.add_argument("--cases", type=Path, help="默认 evals/cases.json")
    parser.add_argument("--output-dir", type=Path, help="默认 evaluation_runs，每次生成独立目录")
    args = parser.parse_args(argv)
    try:
        path, report = run_suite(cases_path=args.cases, output_dir=args.output_dir)
    except (OSError, ValueError) as exc:
        print(f"评估配置/文件错误：{exc}")
        return 2
    for case in report["cases"]:
        print(f"{'PASS' if case['passed'] else 'FAIL'} {case['id']}: {case['observed']['outcome']} / {case['observed']['failure_code']}")
    print(json.dumps(report["summary"], ensure_ascii=False))
    print(f"评估报告：{path}")
    return 0 if report["summary"]["passed_cases"] == report["summary"]["total_cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
