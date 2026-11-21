"""Exercise official write_todos with real graph state, entirely offline."""

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from research_agent.agent import ReportFilesystemBackend, build_agent
from research_agent.main import AgentRunError, run_agent
from research_agent.memory import ResearchMemoryStore
from research_agent.tools.experiment import make_analyze_experiment
from memory_worker import MemoryFakeModel
from test_agent_setup import workspace


class PlanningModel(MemoryFakeModel):
    claim_only: bool = False
    invalid_status: bool = False

    def bind_tools(self, tools, **kwargs):
        names = [t.name if hasattr(t, "name") else t["name"] for t in tools]
        # Same-name replacement must expose one planner even if a model's
        # harness profile also supplies TodoListMiddleware.
        assert names.count("write_todos") == (0 if "analyze_experiment" in names else 1)
        return super().bind_tools(tools, **kwargs)

    def plan(self, statuses, call_id, failed=False):
        labels = ["委派并取得可信分析", "组织并保存报告", "整理摘要和实际路径"]
        if failed:
            labels[1] += "（写入失败，尚未完成）"
        todos = [{"content": label, "status": status} for label, status in zip(labels, statuses, strict=True)]
        return self.call("write_todos", {"todos": todos}, call_id)

    def main_reply(self, messages):
        self.main_history = list(messages)
        calls = {c["id"]: c for m in messages if isinstance(m, AIMessage) for c in m.tool_calls}
        plan_results = [m for m in messages if isinstance(m, ToolMessage)
                        and calls.get(m.tool_call_id, {}).get("name") == "write_todos"]
        if not plan_results:
            statuses = ["done", "pending", "pending"] if self.invalid_status else ["in_progress", "pending", "pending"]
            if self.claim_only:
                statuses = ["completed"] * 3
            return self.plan(statuses, "plan-start")
        if any(m.status == "error" for m in plan_results):
            return AIMessage(content="计划参数错误，未执行研究任务。")
        if self.claim_only:
            return AIMessage(content=f"任务已完成，报告已保存到 {self.workspace}/{self.report_output}（虚假声明）。")
        # Reuse the existing research script; planning adds no new statistics,
        # Skill access, report logic, or Memory behavior.
        domain_messages = [m for m in messages if not (isinstance(m, ToolMessage)
                           and calls.get(m.tool_call_id, {}).get("name") == "write_todos")]
        domain_results = [m for m in domain_messages if isinstance(m, ToolMessage)]
        if domain_results:
            latest = domain_results[-1]
            name = calls[latest.tool_call_id]["name"]
            if name == "write_file" and not any(m.tool_call_id == "plan-finish" for m in plan_results):
                failed = latest.status == "error"
                return self.plan(["completed", "in_progress", "pending"] if failed else ["completed"] * 3,
                                 "plan-finish", failed=failed)
            if name == "write_file" and latest.status == "error":
                return AIMessage(content="报告写入失败，保存与摘要步骤未完成。")
        message = super().main_reply(domain_messages)
        self.main_history = list(messages)
        if any(c["name"] == "write_file" for c in message.tool_calls):
            # Analysis is already verified. Starting the report and updating
            # its status can share a turn; completion awaits the write result.
            message.tool_calls.extend(self.plan(["completed", "in_progress", "pending"], "plan-analyzed").tool_calls)
        return message


def plan_events(trace):
    return [(e.split(" ", 1)[0], json.loads(e.split(" ", 1)[1])) for e in trace
            if e.startswith(("plan_created ", "plan_updated "))]


@pytest.mark.parametrize("with_history", [False, True])
def test_planning_creation_updates_and_existing_research_acceptance(workspace, with_history):
    if with_history:
        run_agent(root=workspace, output="reports/first.md", model=MemoryFakeModel(
            workspace=str(workspace), report_output="reports/first.md"))
    model = PlanningModel(workspace=str(workspace))
    trace = []
    result = run_agent(root=workspace, model=model, output="reports/result.md", trace=trace.append)
    events = plan_events(trace)
    assert [kind for kind, _ in events] == ["plan_created", "plan_updated", "plan_updated"]
    assert [[t["status"] for t in data["todos"]] for _, data in events] == [
        ["in_progress", "pending", "pending"], ["completed", "in_progress", "pending"], ["completed"] * 3]
    assert all(any(f"tool_result write_todos id={data['tool_call_id']} status=success" == e for e in trace)
               for _, data in events)
    assert trace.index(next(e for e in trace if e.startswith("subagent_analysis_verified "))) < trace.index(
        next(e for e in trace if e.startswith("plan_updated ")))
    assert trace.index(next(e for e in trace if e.startswith("tool_result write_file "))) < trace.index(
        next(e for e in trace if e.startswith("plan_updated ") and '"plan-finish"' in e))
    assert "write_todos" not in model.child_bound_names and "analyze_experiment" not in model.bound_names
    assert Path(result["report_path"]).is_file()
    assert result["analysis"] == make_analyze_experiment(workspace).invoke({
        "csv_path": "data/sample_experiments.csv", "function_name": "F1"})
    assert any(e.startswith("skill_load_verified ") for e in trace)
    records = ResearchMemoryStore(workspace).retrieve()
    assert len(records) == 1 and "todos" not in records[0]
    assert len(records[0]["reports"]) == (2 if with_history else 1)


def test_official_tool_actually_updates_graph_todos(workspace):
    model = PlanningModel(workspace=str(workspace))
    state = build_agent(workspace, model=model).invoke({"messages": [{"role": "user", "content": "分析并保存报告，使用计划跟踪步骤。"}]})
    assert len(state["todos"]) == 3
    assert all(t["status"] == "completed" for t in state["todos"])
    assert [m.tool_call_id for m in state["messages"] if isinstance(m, ToolMessage)
            and m.tool_call_id.startswith("plan-")] == ["plan-start", "plan-analyzed", "plan-finish"]


def test_write_failure_leaves_plan_incomplete_and_no_memory(workspace, monkeypatch):
    from deepagents.backends.protocol import WriteResult

    monkeypatch.setattr(ReportFilesystemBackend, "write", lambda *args: WriteResult(error="disk denied"))
    trace = []
    with pytest.raises(AgentRunError, match="disk denied"):
        run_agent(root=workspace, model=PlanningModel(workspace=str(workspace)), output="reports/result.md", trace=trace.append)
    final = plan_events(trace)[-1][1]["todos"]
    assert [t["status"] for t in final] == ["completed", "in_progress", "pending"]
    assert "写入失败" in final[1]["content"]
    assert not (workspace / "reports/result.md").exists() and not (workspace / "memory/research.json").exists()


def test_completed_todos_cannot_replace_real_analysis_or_report(workspace):
    trace = []
    with pytest.raises(AgentRunError, match="未成功调用 task"):
        run_agent(root=workspace, model=PlanningModel(workspace=str(workspace), claim_only=True),
                  output="reports/result.md", trace=trace.append)
    assert all(t["status"] == "completed" for t in plan_events(trace)[0][1]["todos"])
    assert not (workspace / "reports/result.md").exists() and not (workspace / "memory/research.json").exists()


def test_invalid_todo_status_is_not_traced_as_success(workspace):
    trace = []
    with pytest.raises(AgentRunError, match="未成功调用 task"):
        run_agent(root=workspace, model=PlanningModel(workspace=str(workspace), invalid_status=True),
                  output="reports/result.md", trace=trace.append)
    assert not plan_events(trace)
    assert any("tool_result write_todos" in e and "status=error" in e for e in trace)
    assert not (workspace / "memory/research.json").exists()
