"""Real HITL interrupts/checkpoints/resumes, with no paid model calls."""

from copy import deepcopy
import importlib

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from research_agent.agent import ReportTarget, build_agent
from research_agent.hitl import terminal_review
from research_agent.main import AgentRunError, run_agent
from research_agent.memory import ResearchMemoryStore
from research_agent.subagent import RunEvidence
from test_agent_setup import ScriptedResearchModel, workspace
from test_summarization import HarnessModel, RecordingModel


def test_actual_interrupt_precedes_write_then_command_approve_resumes(workspace):
    model = ScriptedResearchModel(workspace=str(workspace))
    evidence = RunEvidence(csv_path="data/sample_experiments.csv", function_name="F1")
    target = ReportTarget(workspace / "reports/result.md")
    graph = build_agent(workspace, model=model, evidence=evidence, report_output="reports/result.md",
                        hitl=True, report_target=target)
    config = {"configurable": {"thread_id": "offline-approve"}, "recursion_limit": 60}
    states = list(graph.stream({"messages": [{"role": "user", "content": "分析并保存报告"}]}, config=config, stream_mode="values"))
    interrupt = next(i for state in states for i in state.get("__interrupt__", []))
    action = interrupt.value["action_requests"][0]
    assert interrupt.value["review_configs"][0]["allowed_decisions"] == ["approve", "edit", "reject"]
    assert action["name"] == "write_file" and action["args"]["file_path"] == "/reports/result.md"
    assert "SHA256:" in action["args"]["content"] and "| Algorithm | Seeds |" in action["args"]["content"]
    assert evidence.completed and not (workspace / "reports/result.md").exists()
    assert graph.get_state(config).next  # Actually paused at a checkpoint.
    final = graph.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config=config)
    assert (workspace / "reports/result.md").read_text() == action["args"]["content"]
    assert any(isinstance(m, ToolMessage) and m.name == "write_file" and m.status == "success" for m in final["messages"])
    assert len(model.child_requests) == 3  # Child analysis wasn't rerun on resume.


@pytest.mark.parametrize("decision", ["approve", "edit"])
def test_approval_flow_verifies_actual_approved_bytes_and_persists_correct_path(workspace, decision):
    model = ScriptedResearchModel(workspace=str(workspace))
    trace, reviewed = [], []

    def review(action):
        assert not (workspace / "reports/result.md").exists()
        assert any(e.startswith("subagent_analysis_verified ") for e in trace)
        reviewed.append(deepcopy(action))
        return {"type": decision, "file_path": "reports/human_edited.md"}

    result = run_agent(root=workspace, model=model, output="reports/result.md", hitl=True, reviewer=review,
                       max_steps=60, trace=trace.append)
    relative = "reports/result.md" if decision == "approve" else "reports/human_edited.md"
    assert result["report_path"] == str(workspace / relative)
    assert (workspace / relative).read_text() == reviewed[0]["args"]["content"]
    assert len(reviewed) == 1 and len(model.child_requests) == 3
    assert any(e.startswith("hitl_interrupt ") for e in trace)
    assert any(e.startswith("hitl_decision ") and f'"decision": "{decision}"' in e for e in trace)
    records = ResearchMemoryStore(workspace).retrieve()
    assert records[0]["reports"][0]["path"] == relative
    if decision == "edit":
        assert not (workspace / "reports/result.md").exists()
        # Locked version keeps the original AI call. Verification uses the
        # trusted review record, rather than pretending the model chose it.
        original = next(c for m in model.main_history if isinstance(m, AIMessage) for c in m.tool_calls if c["name"] == "write_file")
        assert original["args"]["file_path"] == "/reports/result.md"


def test_reject_is_resumed_as_real_error_then_stops_without_file_or_memory(workspace):
    trace = []
    with pytest.raises(AgentRunError, match="人工已拒绝"):
        run_agent(root=workspace, model=ScriptedResearchModel(workspace=str(workspace)), output="reports/result.md",
                  hitl=True, reviewer=lambda action: {"type": "reject"}, max_steps=60, trace=trace.append)
    assert any(e.startswith("hitl_decision ") and '"decision": "reject"' in e for e in trace)
    assert any("tool_result write_file" in e and "status=error" in e for e in trace)
    assert not (workspace / "reports/result.md").exists() and not (workspace / "memory/research.json").exists()


@pytest.mark.parametrize("path", ["../escape.md", "/.env", "data/evil.csv", "/reports/../escape.md", "reports/exists.md"])
def test_human_edit_cannot_bypass_path_or_existing_file_rules(workspace, path):
    existing = workspace / "reports/exists.md"
    existing.write_text("keep original")
    with pytest.raises(AgentRunError, match="审批参数无效"):
        run_agent(root=workspace, model=ScriptedResearchModel(workspace=str(workspace)), output="reports/result.md",
                  hitl=True, reviewer=lambda action: {"type": "edit", "file_path": path}, max_steps=60)
    assert existing.read_text() == "keep original"
    assert sorted(p.name for p in (workspace / "reports").iterdir()) == ["exists.md"]
    assert not (workspace / "memory/research.json").exists()


def test_backend_still_rejects_file_created_after_human_approval(workspace, monkeypatch):
    from research_agent.agent import ReportFilesystemBackend

    original = ReportFilesystemBackend.write

    def race(self, path, content):
        (workspace / path.lstrip("/")).write_text("created by another writer")
        return original(self, path, content)

    monkeypatch.setattr(ReportFilesystemBackend, "write", race)
    with pytest.raises(AgentRunError, match="拒绝覆盖"):
        run_agent(root=workspace, model=ScriptedResearchModel(workspace=str(workspace)), output="reports/result.md",
                  hitl=True, reviewer=lambda action: {"type": "edit", "file_path": "reports/approved.md"}, max_steps=60)
    assert (workspace / "reports/approved.md").read_text() == "created by another writer"
    assert not (workspace / "memory/research.json").exists()


def test_terminal_input_is_explicit_and_can_show_prepared_report(monkeypatch, capsys):
    answers = iter(["", "view", "edit", "reports/new.md"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    decision = terminal_review({"name": "write_file", "args": {"file_path": "/reports/old.md", "content": "prepared report"}})
    assert decision == {"type": "edit", "file_path": "reports/new.md"}
    text = capsys.readouterr().out
    assert "/reports/old.md" in text and "prepared report" in text and "空输入不会批准" in text


def test_hitl_with_planning_summarization_and_memory(workspace):
    trace = []
    result = run_agent(root=workspace, model=HarnessModel(workspace=str(workspace)), summary_model=RecordingModel(),
                       summary_trigger_messages=7, summary_keep_messages=2, output="reports/result.md", hitl=True,
                       reviewer=lambda action: {"type": "edit", "file_path": "reports/combined.md"},
                       max_steps=60, trace=trace.append)
    assert result["report_path"] == str(workspace / "reports/combined.md")
    assert any(e.startswith("summarization_applied ") for e in trace)
    assert any(e.startswith("plan_updated ") for e in trace)
    assert any(e.startswith("skill_load_verified ") for e in trace)
    assert len(ResearchMemoryStore(workspace).retrieve()) == 1


def test_terminal_cli_edit_and_verified_output(workspace, monkeypatch, capsys):
    cli = importlib.import_module("research_agent.main")
    original_run = run_agent
    monkeypatch.setattr(cli, "run_agent", lambda **kwargs: original_run(
        root=workspace, model=ScriptedResearchModel(workspace=str(workspace)), **kwargs))
    answers = iter(["edit", "reports/terminal_edited.md"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert cli.main(["--hitl", "--output", "reports/result.md", "--max-steps", "60"]) == 0
    output = capsys.readouterr().out
    assert "待审批操作：write_file" in output
    assert str(workspace / "reports/terminal_edited.md") in output and "已核验报告" in output
    assert not (workspace / "reports/result.md").exists()


def test_cli_rejection_never_displays_verified_success(workspace, monkeypatch, capsys):
    cli = importlib.import_module("research_agent.main")
    monkeypatch.setattr(cli, "run_agent", lambda **kwargs: run_agent(
        root=workspace, model=ScriptedResearchModel(workspace=str(workspace)), **kwargs))
    monkeypatch.setattr("builtins.input", lambda _: "reject")
    assert cli.main(["--hitl", "--output", "reports/result.md", "--max-steps", "60"]) == 1
    output = capsys.readouterr()
    assert "人工已拒绝" in output.err
    assert "已核验报告" not in output.out and "已保存研究记忆" not in output.out


def test_human_edit_symlink_is_not_authorized(workspace):
    (workspace / ".env").write_text("fake-secret")
    (workspace / "reports/link.md").symlink_to(workspace / ".env")
    with pytest.raises(AgentRunError, match="符号链接"):
        run_agent(root=workspace, model=ScriptedResearchModel(workspace=str(workspace)), output="reports/result.md",
                  hitl=True, reviewer=lambda action: {"type": "edit", "file_path": "reports/link.md"}, max_steps=60)
    assert (workspace / ".env").read_text() == "fake-secret"
    assert not (workspace / "memory/research.json").exists()


def test_analysis_failure_never_requests_approval(workspace):
    def unexpected_review(action):
        raise AssertionError("Failed analysis must not reach human review")

    with pytest.raises(AgentRunError, match="未真正成功执行"):
        run_agent(root=workspace, model=ScriptedResearchModel(workspace=str(workspace), mode="child_no_tool"),
                  output="reports/result.md", hitl=True, reviewer=unexpected_review, max_steps=60)
    assert not (workspace / "reports/result.md").exists()


def test_premature_write_cannot_request_approval_before_analysis(workspace):
    def unexpected_review(action):
        raise AssertionError("A premature write is not a reviewable completed report")

    with pytest.raises(AgentRunError, match="尚未取得可信分析结果"):
        run_agent(root=workspace, model=ScriptedResearchModel(workspace=str(workspace), mode="parallel_write"),
                  output="reports/result.md", hitl=True, reviewer=unexpected_review, max_steps=60)
    assert not (workspace / "reports/result.md").exists()
