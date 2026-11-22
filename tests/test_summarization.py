"""Observe official automatic compaction, actual model inputs and raw evidence."""

import asyncio
import json

import pytest
from deepagents.middleware.summarization import SUMMARIZATION_EVENT_KEY
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from research_agent.agent import ReportFilesystemBackend, build_agent
from research_agent.main import AgentRunError, run_agent, verify_run
from research_agent.memory import ResearchMemoryStore
from research_agent.report import EVIDENCE_START, EVIDENCE_END
from research_agent.subagent import RunEvidence
from research_agent.summarization import HistoryArchiveBackend
from research_agent.tools.experiment import make_analyze_experiment
from memory_worker import MemoryFakeModel
from test_agent_setup import workspace


class RecordingModel(BaseChatModel):
    requests: list = Field(default_factory=list)
    rounds: int = 0

    @property
    def _llm_type(self):
        return "offline-summary-recorder"

    def bind_tools(self, tools, **kwargs):
        return self.bind()

    def _generate(self, messages, **kwargs):
        self.requests.append(list(messages))
        if len(self.requests) <= self.rounds:
            message = AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"file_path": "/reports/denied.md"},
                                                        "id": f"read-{len(self.requests)}", "type": "tool_call"}])
        else:
            message = AIMessage(content=f"FAKE_SUMMARY_{len(self.requests)}")
        return ChatResult(generations=[ChatGeneration(message=message)])


def original_messages(pairs=6):
    return [message for i in range(pairs) for message in (
        HumanMessage(content=f"ORIGINAL_USER_{i}", id=f"user-{i}"),
        AIMessage(content=f"ORIGINAL_REPLY_{i}", id=f"reply-{i}"))] + [HumanMessage(content="RECENT_REQUEST", id="recent")]


def is_summary(message):
    return isinstance(message, HumanMessage) and message.additional_kwargs.get("lc_source") == "summarization"


def final_state(graph, inputs):
    return list(graph.stream(inputs, config={"recursion_limit": 60}, stream_mode="values"))[-1]


def test_automatic_trigger_model_sees_summary_and_recent_while_state_keeps_original(workspace):
    model, summarizer = RecordingModel(), RecordingModel()
    original = original_messages()
    snapshots = [m.model_dump() for m in original]
    graph = build_agent(workspace, model=model, summary_model=summarizer,
                        summary_trigger_messages=8, summary_keep_messages=4)
    state = final_state(graph, {"messages": original})
    event = state[SUMMARIZATION_EVENT_KEY]
    assert len(summarizer.requests) == 1  # No compact tool or manual hook call.
    actual = [m for m in model.requests[0] if not isinstance(m, SystemMessage)]
    assert is_summary(actual[0]) and "FAKE_SUMMARY_1" in actual[0].text
    assert [m.model_dump() for m in actual[1:]] == snapshots[event["cutoff_index"]:]
    assert "ORIGINAL_USER_0" not in "\n".join(m.text for m in actual)
    assert [m.model_dump() for m in state["messages"][:-1]] == snapshots
    archive = HistoryArchiveBackend(workspace).archive_path(event["file_path"])
    content = archive.read_text()
    assert all(m.text in content for m in original[:event["cutoff_index"]])
    assert "RECENT_REQUEST" not in content
    assert not any(is_summary(m) for m in state["messages"])


def test_below_threshold_no_summary_and_no_archive(workspace):
    model, summarizer = RecordingModel(), RecordingModel()
    original = original_messages(pairs=1)
    state = final_state(build_agent(workspace, model=model, summary_model=summarizer,
                                   summary_trigger_messages=8, summary_keep_messages=4), {"messages": original})
    assert not summarizer.requests and not state.get(SUMMARIZATION_EVENT_KEY)
    assert [m.model_dump() for m in model.requests[0] if not isinstance(m, SystemMessage)] == [m.model_dump() for m in original]
    assert not (workspace / ".research_history").exists()


def test_repeated_compaction_appends_one_session_history_and_keeps_raw_tool_pairs(workspace):
    model, summarizer = RecordingModel(rounds=6), RecordingModel()
    state = final_state(build_agent(workspace, model=model, summary_model=summarizer,
                                   summary_trigger_messages=6, summary_keep_messages=2),
                        {"messages": [HumanMessage(content="ORIGINAL_SEED", id="seed")]})
    assert len(summarizer.requests) >= 2
    files = list((workspace / ".research_history/conversation_history").glob("*.md"))
    assert len(files) == 1
    content = files[0].read_text()
    assert content.count("## Summarized at ") == len(summarizer.requests)
    assert "ORIGINAL_SEED" in content and "permission denied" in content
    assert "FAKE_SUMMARY_" not in content  # Summary messages aren't original history.
    assert len([m for m in state["messages"] if isinstance(m, ToolMessage)]) == 6
    assert [c["id"] for m in state["messages"] if isinstance(m, AIMessage) for c in m.tool_calls] == [f"read-{i}" for i in range(1, 7)]
    event = state[SUMMARIZATION_EVENT_KEY]
    last_input = [m for m in model.requests[-1] if not isinstance(m, SystemMessage)]
    assert is_summary(last_input[0])
    assert [m.model_dump() for m in last_input[1:]] == [m.model_dump() for m in state["messages"][event["cutoff_index"]:-1]]


class HarnessModel(MemoryFakeModel):
    main_requests: list = Field(default_factory=list)

    def main_reply(self, messages):
        self.main_history = list(messages)
        self.main_requests.append(list(messages))
        tools = [m for m in messages if isinstance(m, ToolMessage)]
        if not tools:
            return self.call("write_todos", {"todos": [{"content": "分析实验", "status": "in_progress"},
                                                       {"content": "保存报告", "status": "pending"}]}, "plan-start")
        latest = tools[-1].tool_call_id
        if latest == "plan-start":
            return self.call("task", {"subagent_type": "experiment-analyst",
                                      "description": f'csv_path="{self.csv_input}", function_name="F1"'}, "task-analysis")
        if latest == "task-analysis":
            return self.call("write_todos", {"todos": [{"content": "分析实验", "status": "completed"},
                                                       {"content": "保存报告", "status": "in_progress"}]}, "plan-ready")
        if latest == "plan-ready":
            return self.call("write_file", {"file_path": "/" + self.report_output,
                "content": "# 摘要后组织的报告\n\n模拟数据，未做显著性检验。\n\n" + EVIDENCE_START + "\n" + EVIDENCE_END}, "write-report")
        if latest == "write-report":
            return self.call("write_todos", {"todos": [{"content": "分析实验", "status": "completed"},
                                                       {"content": "保存报告", "status": "completed"}]}, "plan-finish")
        assert latest == "plan-finish"
        return AIMessage(content=f"本次模拟数据分析完成：{self.report_output}；{self.workspace}/{self.report_output}")


def test_full_harness_preserves_evidence_when_task_result_is_absent_from_model_context(workspace):
    # Seed real persistent Memory, then start a fresh run with related history.
    run_agent(root=workspace, model=MemoryFakeModel(workspace=str(workspace)), output="reports/result.md")
    model = HarnessModel(workspace=str(workspace), report_output="reports/compacted.md")
    summarizer = RecordingModel()
    trace = []
    result = run_agent(root=workspace, model=model, summary_model=summarizer,
                       summary_trigger_messages=7, summary_keep_messages=2,
                       output="reports/compacted.md", max_steps=60, trace=trace.append)
    assert any(e.startswith("memory_retrieved matched=1 ") for e in trace)
    assert any(e.startswith("plan_created ") for e in trace) and any(e.startswith("plan_updated ") for e in trace)
    assert any(e.startswith("skill_load_verified ") for e in trace)
    assert any(e.startswith("summarization_applied ") for e in trace)
    write_input = next(request for request in model.main_requests
                       if any(isinstance(m, ToolMessage) and m.tool_call_id == "plan-ready" for m in request))
    assert any(is_summary(m) for m in write_input)
    assert not any(isinstance(m, ToolMessage) and m.tool_call_id == "task-analysis" for m in write_input)
    expected = make_analyze_experiment(workspace).invoke({"csv_path": "data/sample_experiments.csv", "function_name": "F1"})
    assert result["analysis"] == expected
    report = (workspace / "reports/compacted.md").read_text()
    assert expected["evidence_markdown"] in report
    records = ResearchMemoryStore(workspace).retrieve()
    assert len(records) == 1 and len(records[0]["reports"]) == 2
    assert records[0]["facts"]["final_statistics"] == expected["algorithms"]


def test_raw_task_message_and_run_evidence_survive_lossy_summary(workspace):
    model = HarnessModel(workspace=str(workspace))
    evidence = RunEvidence(csv_path="data/sample_experiments.csv", function_name="F1")
    state = final_state(build_agent(workspace, model=model, evidence=evidence, summary_model=RecordingModel(),
                                   summary_trigger_messages=7, summary_keep_messages=2),
                        {"messages": [HumanMessage(content="分析并保存模拟实验报告。", id="original-task")]})
    task = next(m for m in state["messages"] if isinstance(m, ToolMessage) and m.tool_call_id == "task-analysis")
    assert json.loads(task.text) == evidence.completed[task.tool_call_id]["payload"]
    assert state["todos"] and all(t["status"] == "completed" for t in state["todos"])
    assert state["messages"][0].id == "original-task"
    result = verify_run(state["messages"], root=workspace, csv=workspace / "data/sample_experiments.csv", function="F1",
                        destination=workspace / "reports/result.md", run_evidence=evidence)
    archive = HistoryArchiveBackend(workspace).archive_path(state[SUMMARIZATION_EVENT_KEY]["file_path"])
    assert result["analysis"]["source"]["sha256"] in archive.read_text()
    assert task.text in archive.read_text()  # Complete original task JSON, not just its hash.
    # A summary containing only a marker never replaces the trusted task JSON.
    assert "FAKE_SUMMARY_" not in task.text


@pytest.mark.parametrize("path", ["/.env", "/data/sample_experiments.csv", "/reports/result.md", "/etc/passwd",
                                  "/conversation_history/../.env", "/conversation_history/session_bad.md"])
def test_archive_backend_whitelist_and_main_permissions(workspace, path):
    backend = HistoryArchiveBackend(workspace)
    assert backend.download_files([path])[0].error
    assert backend.write(path, "forbidden").error
    assert backend.edit(path, "old", "old new").error
    assert backend.read(path).error
    assert ReportFilesystemBackend(workspace).read(path).error


def test_archive_append_only_and_symlink_rejection(workspace):
    backend = HistoryArchiveBackend(workspace)
    name = "/conversation_history/session_" + "a" * 32 + ".md"
    assert backend.write(name, "first").error is None
    assert backend.write(name, "overwrite").error
    assert backend.edit(name, "first", "replacement").error
    assert backend.edit(name, "wrong", "wrong append").error
    assert backend.edit(name, "first", "first second").error is None
    assert asyncio.run(backend.adownload_files([name]))[0].content == b"first second"
    assert ReportFilesystemBackend(workspace).write(name, "forbidden").error
    path = backend.archive_path(name)
    path.unlink()
    (workspace / ".env").write_text("fake-secret")
    path.symlink_to(workspace / ".env")
    assert backend.download_files([name])[0].error and backend.write(name, "bad").error
    assert backend.edit(name, "fake-secret", "fake-secret extra").error
    assert (workspace / ".env").read_text() == "fake-secret"


def test_archive_failure_does_not_report_success_or_write_memory(workspace, monkeypatch):
    from deepagents.backends.protocol import WriteResult

    monkeypatch.setattr(HistoryArchiveBackend, "write", lambda *args: WriteResult(error="disk denied"))
    with pytest.warns(UserWarning, match="Older messages will not be recoverable"):
        with pytest.raises(AgentRunError, match="历史归档失败"):
            run_agent(root=workspace, model=HarnessModel(workspace=str(workspace)), summary_model=RecordingModel(),
                       summary_trigger_messages=7, summary_keep_messages=2, output="reports/result.md", max_steps=60)
    assert not (workspace / "reports/result.md").exists() and not (workspace / "memory/research.json").exists()
