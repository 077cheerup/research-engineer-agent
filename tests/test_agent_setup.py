"""Offline real-graph tests with a scripted model; never calls a remote API."""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from research_agent.agent import ConfigurationError, ReportFilesystemBackend, build_agent, load_model
from research_agent.main import AgentRunError, main, run_agent
from research_agent.subagent import AnalysisVerificationError, RunEvidence, verify_analysis
from research_agent.tools.experiment import make_analyze_experiment


class ScriptedResearchModel(BaseChatModel):
    """Script both roles, while binding distinct real tool sets per model call."""

    workspace: str
    mode: str = "normal"
    bound_names: list[str] = Field(default_factory=list)
    child_bound_names: list[str] = Field(default_factory=list)
    task_descriptions: list[str] = Field(default_factory=list)
    main_history: list[Any] = Field(default_factory=list)
    child_history: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "offline-scripted-research"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        names = [t.name if hasattr(t, "name") else t["name"] for t in tools]
        if names == ["analyze_experiment"]:
            self.child_bound_names.extend(names)
        else:
            self.bound_names.extend(names)
            self.task_descriptions.extend(t.description for t in tools if getattr(t, "name", None) == "task")
        return self.bind(offline_tools=names)

    @staticmethod
    def call(name, args, call_id):
        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])

    def child_reply(self, messages):
        self.child_history = list(messages)
        results = [m for m in messages if isinstance(m, ToolMessage)]
        if self.mode == "child_no_tool":
            return AIMessage(content='{"evidence_markdown":"claimed evidence","significance_tested":false}')
        if not results or self.mode == "child_loop":
            csv = "data/other.csv" if self.mode == "child_wrong_csv" else "data/sample_experiments.csv"
            function = "F2" if self.mode == "child_wrong_function" else "F1"
            if self.mode == "child_wrong_function":
                csv = "data/F2.csv"
            if self.mode == "child_tool_error":
                csv = "data/missing.csv"
            return self.call("analyze_experiment", {"csv_path": csv, "function_name": function}, f"child-analysis-{len(results)}")
        successful = [m for m in results if m.name == "analyze_experiment" and m.status == "success"]
        if not successful:
            return AIMessage(content="分析失败")
        if self.mode == "child_write" and len(results) == 1:
            return self.call("write_file", {"file_path": "/reports/child.md", "content": "forbidden"}, "child-write")
        if self.mode == "child_summary":
            return AIMessage(content="算法 B 更好。未做显著性检验。")
        data = json.loads(successful[-1].content)
        reply = {
            "evidence_markdown": data["evidence_markdown"],
            "conclusions": {"final_performance": "B 最终均值较低。", "stability": "B 的样本标准差较低。", "convergence": "B 在各采样点均值较低，不代表差距持续扩大。"},
            "significance_tested": False,
        }
        if self.mode == "child_missing_evidence":
            del reply["evidence_markdown"]
        if self.mode == "child_bad_evidence":
            reply["evidence_markdown"] = reply["evidence_markdown"].replace("| A | 5 |", "| A | 999 |")
        if self.mode == "child_significance":
            reply["significance_tested"] = True
        return AIMessage(content=json.dumps(reply, ensure_ascii=False))

    def main_reply(self, messages):
        self.main_history = list(messages)
        results = [m for m in messages if isinstance(m, ToolMessage)]
        if self.mode == "read_secret" and not results:
            return self.call("read_file", {"file_path": "/.env"}, "read-1")
        if self.mode in {"read_secret", "no_tools"}:
            return AIMessage(content="已保存报告（虚假声称）。")
        if not results or self.mode == "loop":
            if self.mode == "direct_analysis":
                return self.call("analyze_experiment", {"csv_path": "data/sample_experiments.csv", "function_name": "F1"}, "main-analysis")
            target = "general-purpose" if self.mode == "wrong_target" else "experiment-analyst"
            message = self.call("task", {"subagent_type": target, "description": 'csv_path="data/sample_experiments.csv", function_name="F1"; analyze simulated data'}, f"task-{len(results)}")
            if self.mode == "parallel_write":
                message.tool_calls.append({"name": "write_file", "args": {"file_path": "/reports/result.md", "content": "模拟数据：尚未分析"}, "id": "premature-write", "type": "tool_call"})
            return message
        call_names = {c["id"]: c["name"] for m in messages if isinstance(m, AIMessage) for c in m.tool_calls}
        if call_names.get(results[-1].tool_call_id) == "task" and self.mode not in {"no_write", "wrong_target"}:
            data = json.loads(results[-1].content)["analysis"]
            table = data["evidence_markdown"]
            if self.mode == "bad_numbers":
                table = table.replace("| A | 5 |", "| A | 999 |")
            if self.mode == "markdown_breaks":
                table = table.replace("\n", "  \n")
            return self.call("write_file", {
                "file_path": f"{self.workspace}/reports/result.md" if self.mode == "host_path" else "/reports/result.md",
                "content": "# 离线 Fake Model 流程验证\n\n模拟数据，非正式 CEC2013；无显著性检验。\n\n" + table,
            }, "write-1")
        return AIMessage(content=f"模拟数据摘要。报告已保存到 reports/result.md，绝对路径：{self.workspace}/reports/result.md")

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        message = self.child_reply(messages) if kwargs.get("offline_tools") == ["analyze_experiment"] else self.main_reply(messages)
        return ChatResult(generations=[ChatGeneration(message=message)])


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "reports").mkdir()
    sample = Path(__file__).resolve().parents[1] / "data" / "sample_experiments.csv"
    (tmp_path / "data" / sample.name).write_bytes(sample.read_bytes())
    (tmp_path / "data/other.csv").write_bytes(sample.read_bytes())
    (tmp_path / "data/F2.csv").write_bytes(sample.read_bytes().replace(b",F1,", b",F2,"))
    return tmp_path


@pytest.mark.parametrize("mode", ["normal", "markdown_breaks"])
def test_real_graph_registers_and_executes_tools_offline(workspace, mode):
    model = ScriptedResearchModel(workspace=str(workspace), mode=mode)
    trace = []
    result = run_agent(root=workspace, output="reports/result.md", model=model, trace=trace.append)
    assert {"task", "write_file"} <= set(model.bound_names)
    assert "analyze_experiment" not in model.bound_names
    assert set(model.child_bound_names) == {"analyze_experiment"}
    assert all("experiment-analyst" in d and "general-purpose" not in d for d in model.task_descriptions)
    assert not any(isinstance(m, ToolMessage) and m.name == "analyze_experiment" for m in model.main_history)
    assert "read_file" in model.bound_names  # Required by 0.7.40, permission denied.
    assert "edit_file" not in model.bound_names
    assert "execute" not in model.bound_names
    assert result["analysis"]["algorithms"]["A"]["n_seeds"] == 5
    assert Path(result["report_path"]).is_file()
    assert any("tool_result write_file" in line and "status=success" in line for line in trace)
    assert any("analysis_evidence" in line for line in trace)


@pytest.mark.parametrize("mode,message", [
    ("no_tools", "未成功调用 task"), ("no_write", "write_file"),
    ("bad_numbers", "数值证据"), ("loop", "步数上限"), ("parallel_write", "write_file"),
])
def test_incomplete_runs_never_pass(workspace, mode, message):
    model = ScriptedResearchModel(workspace=str(workspace), mode=mode)
    with pytest.raises(AgentRunError, match=message):
        run_agent(root=workspace, output="reports/result.md", model=model, max_steps=8)


def test_write_failure_not_reported_as_success(workspace, monkeypatch):
    from deepagents.backends.protocol import WriteResult

    monkeypatch.setattr(ReportFilesystemBackend, "write", lambda *args: WriteResult(error="disk denied"))
    model = ScriptedResearchModel(workspace=str(workspace))
    with pytest.raises(AgentRunError, match="write_file"):
        run_agent(root=workspace, output="reports/result.md", model=model)
    assert not (workspace / "reports" / "result.md").exists()


def test_host_path_failure_is_actionable_in_trace_and_final_error(workspace):
    trace = []
    model = ScriptedResearchModel(workspace=str(workspace), mode="host_path")
    with pytest.raises(AgentRunError, match="虚拟路径") as error:
        run_agent(root=workspace, output="reports/result.md", model=model, trace=trace.append)
    assert "工具错误" in str(error.value)
    assert any(line.startswith("tool_path ") and str(workspace) in line for line in trace)
    assert any("tool_error write_file:" in line and "主机绝对路径" in line for line in trace)
    assert not (workspace / "reports/result.md").exists()


def test_backend_restricts_writes_and_never_overwrites(workspace):
    backend = ReportFilesystemBackend(workspace)
    assert backend.virtual_mode is True
    assert backend.write("/reports/test.md", "first").error is None
    assert backend.write("/reports/test.md", "second").error
    assert (workspace / "reports" / "test.md").read_text() == "first"
    for path in ("/.env", "/data/evil.csv", "/reports/../evil.md", "/reports/test.txt"):
        assert backend.write(path, "bad").error
    assert asyncio.run(backend.awrite("/reports/async.md", "async")).error is None


def test_backend_rejects_symlinks(workspace):
    outside = workspace.parent / "outside"
    outside.mkdir()
    (workspace / "reports").rmdir()
    (workspace / "reports").symlink_to(outside, target_is_directory=True)
    assert ReportFilesystemBackend(workspace).write("/reports/test.md", "bad").error
    assert not (outside / "test.md").exists()


def test_existing_report_fails_before_model(workspace):
    path = workspace / "reports" / "result.md"
    path.write_text("original")
    with pytest.raises(AgentRunError, match="已存在"):
        run_agent(root=workspace, output="reports/result.md")
    assert path.read_text() == "original"


def test_actual_backend_filesystem_failure(workspace):
    # Same deterministic filesystem setup used by the optional remote script.
    (workspace / "reports").rmdir()
    (workspace / "reports").write_text("blocks mkdir")
    trace = []
    with pytest.raises(AgentRunError, match="write_file"):
        run_agent(root=workspace, output="reports/result.md", model=ScriptedResearchModel(workspace=str(workspace)), trace=trace.append)
    assert any("tool_result write_file" in line and "status=error" in line for line in trace)


def test_configuration_missing(workspace, monkeypatch):
    for name in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ConfigurationError, match="LLM_BASE_URL, LLM_API_KEY, LLM_MODEL"):
        load_model(workspace)


def test_dotenv_explicit_chatopenai_arguments(workspace, monkeypatch):
    for name in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    (workspace / ".env").write_text("LLM_BASE_URL=https://offline.invalid/v1\nLLM_API_KEY=fake-test-key\nLLM_MODEL=fake-model\n")
    monkeypatch.setenv("LLM_MODEL", "environment-model")
    model = load_model(workspace)  # Constructor only, no invoke/network.
    assert model.model_name == "environment-model"
    assert model.openai_api_base == "https://offline.invalid/v1"
    assert model.openai_api_key.get_secret_value() == "fake-test-key"
    assert model.use_responses_api is False


def test_cli_failure_has_nonzero_status(monkeypatch, capsys):
    def fail(**kwargs):
        raise AgentRunError("没有真实工具调用")
    monkeypatch.setattr("research_agent.main.run_agent", fail)
    assert main([]) == 1
    assert "没有真实工具调用" in capsys.readouterr().err


def test_required_read_tool_cannot_read_dotenv(workspace):
    secret = "fake-private-test-value"
    (workspace / ".env").write_text(secret)
    graph = build_agent(workspace, model=ScriptedResearchModel(workspace=str(workspace), mode="read_secret"))
    state = graph.invoke({"messages": [{"role": "user", "content": "offline permission test"}]})
    results = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert len(results) == 1
    assert results[0].status == "error"
    assert "permission denied" in results[0].content
    assert secret not in results[0].content


@pytest.mark.parametrize("mode,message", [
    ("child_no_tool", "未真正成功执行"),
    ("child_tool_error", "未真正成功执行"),
    ("child_summary", "不能只返回自然语言"),
    ("child_missing_evidence", "完整数值证据"),
    ("child_bad_evidence", "完整数值证据"),
    ("child_significance", "significance_tested=false"),
    ("child_wrong_csv", "CSV 或 Function"),
    ("child_wrong_function", "CSV 或 Function"),
    ("child_loop", "步数上限"),
])
def test_subagent_failures_never_produce_report(workspace, mode, message):
    parent = ScriptedResearchModel(workspace=str(workspace))
    child = ScriptedResearchModel(workspace=str(workspace), mode=mode)
    with pytest.raises(AgentRunError, match=message):
        run_agent(root=workspace, output="reports/result.md", model=parent, analyst_model=child, max_steps=8)
    assert not (workspace / "reports/result.md").exists()
    assert "analyze_experiment" not in parent.bound_names
    assert set(child.child_bound_names) == {"analyze_experiment"}


@pytest.mark.parametrize("mode", ["direct_analysis", "wrong_target"])
def test_main_cannot_analyze_directly_or_use_default_subagent(workspace, mode):
    model = ScriptedResearchModel(workspace=str(workspace), mode=mode)
    with pytest.raises(AgentRunError, match="未成功调用 task"):
        run_agent(root=workspace, output="reports/result.md", model=model)
    assert not model.child_bound_names
    assert not (workspace / "reports/result.md").exists()


def test_child_cannot_write_reports(workspace):
    parent = ScriptedResearchModel(workspace=str(workspace))
    child = ScriptedResearchModel(workspace=str(workspace), mode="child_write")
    result = run_agent(root=workspace, output="reports/result.md", model=parent, analyst_model=child)
    denied = [m for m in child.child_history if isinstance(m, ToolMessage) and m.name == "write_file"]
    assert denied and denied[0].status == "error"
    assert not (workspace / "reports/child.md").exists()
    assert Path(result["report_path"]).is_file()  # Main can still save valid analysis.


def test_task_transports_full_verified_result_without_child_messages(workspace):
    model = ScriptedResearchModel(workspace=str(workspace))
    audit = RunEvidence(csv_path="data/sample_experiments.csv", function_name="F1")
    graph = build_agent(workspace, model=model, evidence=audit)
    state = graph.invoke({"messages": [{"role": "user", "content": "offline integration"}]})
    calls = {c["id"]: c for m in state["messages"] if isinstance(m, AIMessage) for c in m.tool_calls}
    task_message = next(m for m in state["messages"] if isinstance(m, ToolMessage) and calls[m.tool_call_id]["name"] == "task")
    payload = verify_analysis(task_message.tool_call_id, task_message.text, audit, csv_path="data/sample_experiments.csv", function_name="F1")
    expected = make_analyze_experiment(workspace).invoke({"csv_path": "data/sample_experiments.csv", "function_name": "F1"})
    assert payload["analysis"] == expected
    assert payload["significance_tested"] is False
    assert set(payload["conclusions"]) == {"final_performance", "stability", "convergence"}
    assert not any(isinstance(m, ToolMessage) and m.name == "analyze_experiment" for m in state["messages"])
    assert any(isinstance(m, ToolMessage) and m.name == "analyze_experiment" for m in model.child_history)
    # A real task JSON alone is not proof in another run.
    with pytest.raises(AnalysisVerificationError, match="真实子 Agent 分析执行记录"):
        verify_analysis(task_message.tool_call_id, task_message.text, RunEvidence(), csv_path="data/sample_experiments.csv", function_name="F1")
    with pytest.raises(AnalysisVerificationError, match="CSV 或 Function"):
        verify_analysis(task_message.tool_call_id, task_message.text, audit, csv_path="data/other.csv", function_name="F1")
    forged = json.loads(task_message.text)
    forged["analysis"]["algorithms"]["A"]["mean"] = 999
    with pytest.raises(AnalysisVerificationError, match="不一致"):
        verify_analysis(task_message.tool_call_id, json.dumps(forged), audit, csv_path="data/sample_experiments.csv", function_name="F1")


def test_run_evidence_is_not_reused(workspace):
    model = ScriptedResearchModel(workspace=str(workspace))
    run_agent(root=workspace, output="reports/result.md", model=model)
    model.mode = "child_no_tool"
    with pytest.raises(AgentRunError, match="未真正成功执行"):
        run_agent(root=workspace, output="reports/second.md", model=model)
    assert not (workspace / "reports/second.md").exists()
