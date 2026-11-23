"""Offline callback fixtures and actual graph integration; usage is synthetic."""

import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langgraph.types import Command

from research_agent import evaluation
from research_agent.evaluation_model import EvaluationModel
from research_agent.observability import PerformanceRecorder, reported_usage, token_summary


def response(usage=None, legacy=None):
    return LLMResult(generations=[[ChatGeneration(message=AIMessage(content="", usage_metadata=usage))]],
                     llm_output={"token_usage": legacy} if legacy is not None else None)


def test_nested_spans_exact_clock_context_counts_and_no_payload_logging():
    recorder = PerformanceRecorder("fixture", clock=iter([10, 11, 12, 13, 14, 15]).__next__)
    chain, task, child, model = (uuid4() for _ in range(4))
    recorder.on_chain_start({}, {"secret": "unlogged-input"}, run_id=chain)
    recorder.on_tool_start({"name": "task"}, "unlogged-tool-body", run_id=task, parent_run_id=chain)
    recorder.on_chain_start({}, {}, run_id=child, parent_run_id=task)
    recorder.on_chat_model_start({"name": "FixtureModel", "kwargs": {"api_key": "unlogged-api-key"}},
        [[HumanMessage(content="unlogged-prompt")]], run_id=model, parent_run_id=child)
    recorder.on_llm_end(response({"input_tokens": 11, "output_tokens": 4, "total_tokens": 15}), run_id=model)
    recorder.on_tool_end("unlogged-tool-output", run_id=task)
    spans, summary = recorder.finish()
    assert len(spans) == 3
    assert recorder.spans[str(model)]["parent_span_id"] == str(task)
    assert recorder.spans[str(task)]["parent_span_id"] == recorder.root_id
    assert recorder.spans[str(model)]["input_messages"] == 1
    assert recorder.spans[str(model)]["input_text_chars"] == len("unlogged-prompt")
    assert recorder.spans[str(model)]["duration_ms"] == 1000
    assert recorder.spans[str(task)]["duration_ms"] == 3000
    assert summary["duration_ms"] == 5000  # Do not sum nested span durations.
    assert summary["token_usage"]["total_tokens"] == 15
    assert summary["token_usage"]["status"] == "complete"
    assert "unlogged" not in json.dumps(spans)


@pytest.mark.parametrize("source", ["normalized", "legacy", "both"])
def test_usage_sources_are_normalized_and_not_double_counted(source):
    normalized = {"input_tokens": 11, "output_tokens": 4, "total_tokens": 15}
    legacy = {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}
    result = response(normalized if source != "legacy" else None, legacy if source != "normalized" else None)
    usage = reported_usage(result)
    assert usage["total_tokens"] == 15
    assert usage["source"] == ("llm_output.token_usage" if source == "legacy" else "usage_metadata")


def test_alternative_generations_are_not_extra_token_usage():
    result = response({"input_tokens": 11, "output_tokens": 4, "total_tokens": 15})
    result.generations[0].append(result.generations[0][0])
    assert reported_usage(result)["total_tokens"] == 15


def test_unknown_zero_partial_and_not_called_usage_are_distinct():
    assert reported_usage(response()) is None
    assert token_summary([])["status"] == "not_called"
    unknown = {"token_usage": None}
    zero = {"token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}}
    assert token_summary([unknown])["status"] == "unavailable"
    assert token_summary([unknown])["total_tokens"] is None
    assert token_summary([zero])["status"] == "complete"
    assert token_summary([zero])["total_tokens"] == 0
    partial = token_summary([zero, unknown])
    assert partial["status"] == "partial" and partial["calls_without_usage"] == 1
    # Missing fields, negative counts or non-integers are unknown, never estimates.
    for invalid in ({"input_tokens": 5}, {"input_tokens": -1, "output_tokens": 1, "total_tokens": 0},
                    {"input_tokens": True, "output_tokens": 1, "total_tokens": 2}):
        fixture = SimpleNamespace(generations=[[SimpleNamespace(message=SimpleNamespace(usage_metadata=invalid))]], llm_output=None)
        assert reported_usage(fixture) is None


def test_errors_and_interrupted_spans_end_without_logging_error_body():
    recorder = PerformanceRecorder(clock=iter(range(9)).__next__)
    model, tool, interrupted = (uuid4() for _ in range(3))
    recorder.on_llm_start({"name": "FailureModel"}, ["unlogged-prompt"], run_id=model)
    recorder.on_llm_error(RuntimeError("unlogged-provider-error"), run_id=model)
    recorder.on_tool_start({"name": "write_file"}, "", run_id=tool)
    recorder.on_tool_error(ValueError("unlogged-tool-error"), run_id=tool)
    recorder.on_chat_model_start({"name": "Interrupted"}, [[]], run_id=interrupted)
    spans, summary = recorder.finish(RuntimeError("unlogged-root-error"))
    assert summary["model_errors"] == 1 and summary["tool_errors"] == 1
    assert summary["incomplete_spans"] == 1
    assert all(s["duration_ms"] is not None and s["status"] != "running" for s in spans)
    assert recorder.spans[str(model)]["error_type"] == "RuntimeError"
    assert "unlogged" not in json.dumps(spans)


@pytest.mark.parametrize("command", [False, True])
def test_error_tool_message_is_not_a_successful_span(command):
    recorder = PerformanceRecorder()
    tool = uuid4()
    recorder.on_tool_start({"name": "write_file"}, "", run_id=tool)
    message = ToolMessage(content="blocked", status="error", tool_call_id="tool")
    recorder.on_tool_end(Command(update={"messages": [message]}) if command else message, run_id=tool)
    _, summary = recorder.finish()
    assert summary["tool_errors"] == 1


def test_real_graph_spans_match_tool_events_and_preserve_evaluation(tmp_path):
    path, report = evaluation.run_suite(output_dir=tmp_path)
    overall = report["performance"]
    assert report["summary"]["passed_cases"] == 5
    assert overall["model_calls"] > 0 and overall["tool_calls"] > 0
    assert overall["incomplete_spans"] == 0
    assert overall["token_usage"]["status"] == "unavailable"
    assert overall["token_usage"]["total_tokens"] is None
    assert overall["model_calls"] == sum(c["performance"]["model_calls"] for c in report["cases"])
    assert overall["tool_calls"] == sum(c["performance"]["tool_calls"] for c in report["cases"])
    for case in report["cases"]:
        captured = json.loads((path.parent / case["artifacts"]["case_directory"] / "trace.json").read_text())
        spans = captured["spans"]
        tool_spans = {s["span_id"]: s for s in spans if s["kind"] == "tool"}
        events = {e["run_id"]: e for e in captured["tool_events"] if e["event"] == "tool_end"}
        assert tool_spans.keys() == events.keys()
        assert all(span["status"] == events[key]["status"] for key, span in tool_spans.items())
        assert all(s["duration_ms"] >= 0 and s["start_ms"] >= 0 for s in spans)
        assert all(s["parent_span_id"] is None or s["parent_span_id"] in {p["span_id"] for p in spans} for s in spans)
        if case["id"] == "normal_report":
            task = next(s for s in spans if s["name"] == "task")
            assert any(s["kind"] == "model" and s["parent_span_id"] == task["span_id"] for s in spans)
            assert next(s for s in spans if s["name"] == "analyze_experiment")["parent_span_id"] == task["span_id"]
            assert case["performance"]["by_model"][0]["max_input_text_chars"] > 0
        if case["id"] == "csv_missing":
            assert len(spans) == 1 and case["performance"]["model_calls"] == 0
            assert case["performance"]["token_usage"]["status"] == "not_called"


def test_reported_synthetic_usage_flows_through_actual_model_callbacks(tmp_path, monkeypatch):
    class UsageFixtureModel(EvaluationModel):
        def _generate(self, messages, **kwargs):
            result = super()._generate(messages, **kwargs)
            result.generations[0].message.usage_metadata = {"input_tokens": 11, "output_tokens": 4, "total_tokens": 15}
            result.llm_output = {"token_usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}}
            return result

    monkeypatch.setattr(evaluation, "EvaluationModel", UsageFixtureModel)
    _, report = evaluation.run_suite(output_dir=tmp_path)
    perf = report["performance"]
    assert report["mode"] == "offline_fake" and report["summary"]["passed_cases"] == 5
    assert perf["token_usage"]["status"] == "complete"
    assert perf["token_usage"]["input_tokens"] == 11 * perf["model_calls"]
    assert perf["token_usage"]["total_tokens"] == 15 * perf["model_calls"]


def test_actual_model_failure_records_span_and_cli_displays_performance(tmp_path, monkeypatch, capsys):
    class FailingFixtureModel(EvaluationModel):
        def _generate(self, messages, **kwargs):
            raise RuntimeError("unlogged-provider-error")

    monkeypatch.setattr(evaluation, "EvaluationModel", FailingFixtureModel)
    assert evaluation.main(["--output-dir", str(tmp_path)]) == 1
    output = capsys.readouterr().out
    assert "性能摘要：" in output and "model_calls=" in output
    report = json.loads(next(tmp_path.glob("*/report.json")).read_text())
    assert report["performance"]["model_errors"] == 4
    assert report["performance"]["incomplete_spans"] == 0
    assert "unlogged" not in json.dumps(report)
