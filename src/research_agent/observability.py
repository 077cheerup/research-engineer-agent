"""Local model/tool spans; no prompt logging, token estimation or exporter."""

from __future__ import annotations

from time import perf_counter
from uuid import uuid4

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import ToolMessage
from langgraph.types import Command


def reported_usage(response):
    """Prefer normalized message usage; never add duplicate provider totals."""
    keys = ("input_tokens", "output_tokens", "total_tokens")

    def valid(data):
        return isinstance(data, dict) and all(type(data.get(k)) is int and data[k] >= 0 for k in keys)

    # Candidates within one generation are alternatives, not extra requests.
    usages = [getattr(group[0].message, "usage_metadata", None)
              for group in response.generations if group and hasattr(group[0], "message")]
    if usages and len(usages) == len(response.generations) and all(valid(u) for u in usages):
        return {"source": "usage_metadata", **{k: sum(u[k] for u in usages) for k in keys}}
    data = (response.llm_output or {}).get("token_usage")
    if isinstance(data, dict):
        normalized = dict(zip(keys, (data.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens"))))
        if valid(normalized):
            return {"source": "llm_output.token_usage", **normalized}
    return None


def token_summary(models):
    known = [s["token_usage"] for s in models if s.get("token_usage") is not None]
    return {
        "status": "not_called" if not models else "unavailable" if not known else
                  "complete" if len(known) == len(models) else "partial",
        "calls_with_usage": len(known), "calls_without_usage": len(models) - len(known),
        "input_tokens": sum(u["input_tokens"] for u in known) if known else None,
        "output_tokens": sum(u["output_tokens"] for u in known) if known else None,
        "total_tokens": sum(u["total_tokens"] for u in known) if known else None,
    }


def performance_summary(spans, *, duration_ms):
    """Inclusive span durations overlap; only duration_ms is wall-clock time."""
    models = [s for s in spans if s["kind"] == "model"]
    tools = [s for s in spans if s["kind"] == "tool"]

    def grouped(items):
        groups = []
        for name in sorted({s["name"] for s in items}):
            selected = [s for s in items if s["name"] == name]
            group = {"name": name, "calls": len(selected),
                     "errors": sum(s["status"] == "error" for s in selected),
                     "inclusive_total_ms": round(sum(s["duration_ms"] for s in selected), 3),
                     "max_ms": max(s["duration_ms"] for s in selected)}
            if selected[0]["kind"] == "model":
                group.update(token_usage=token_summary(selected),
                             max_input_messages=max(s.get("input_messages", 0) for s in selected),
                             max_input_text_chars=max(s.get("input_text_chars", 0) for s in selected))
            groups.append(group)
        return groups

    return {"duration_ms": round(duration_ms, 3), "model_calls": len(models), "tool_calls": len(tools),
            "model_errors": sum(s["status"] == "error" for s in models),
            "tool_errors": sum(s["status"] == "error" for s in tools),
            "incomplete_spans": sum(s["status"] == "incomplete" for s in spans),
            "token_usage": token_summary(models), "by_model": grouped(models), "by_tool": grouped(tools),
            "slowest_spans": [{k: s[k] for k in ("span_id", "kind", "name", "duration_ms", "status")}
                              for s in sorted(models + tools, key=lambda s: s["duration_ms"], reverse=True)[:5]]}


class PerformanceRecorder(BaseCallbackHandler):
    """One instance per run; graph callbacks retain ancestry without graph payloads."""

    run_inline = True

    def __init__(self, name="agent", *, clock=perf_counter):
        self.clock = clock
        self.origin = clock()
        self.root_id = uuid4().hex
        self.parents = {}
        self.spans = {self.root_id: {"span_id": self.root_id, "parent_span_id": None, "kind": "run",
                                   "name": name, "start_ms": 0.0, "duration_ms": None, "status": "running"}}
        self.started = {self.root_id: self.origin}

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
        self.parents[str(run_id)] = str(parent_run_id) if parent_run_id else None

    def _start(self, kind, serialized, run_id, parent_run_id, **fields):
        key = str(run_id)
        self.parents[key] = str(parent_run_id) if parent_run_id else None
        parent = self.parents[key]
        while parent is not None and parent not in self.spans:
            parent = self.parents.get(parent)
        now = self.clock()
        name = (serialized or {}).get("name") or ((serialized or {}).get("id") or [kind])[-1]
        self.started[key] = now
        self.spans[key] = {"span_id": key, "parent_span_id": parent or self.root_id,
                           "kind": kind, "name": name, "start_ms": round((now - self.origin) * 1000, 3),
                           "duration_ms": None, "status": "running", **fields}

    def _end(self, run_id, status="success", **fields):
        key = str(run_id)
        span = self.spans.get(key)
        if span is not None and span["status"] == "running":
            span.update(duration_ms=round((self.clock() - self.started[key]) * 1000, 3), status=status, **fields)

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kwargs):
        metadata = kwargs.get("metadata") or {}
        if isinstance(metadata.get("ls_model_name"), str):
            serialized = {"name": metadata["ls_model_name"]}
        self._start("model", serialized, run_id, parent_run_id,
                    input_messages=sum(len(batch) for batch in messages),
                    input_text_chars=sum(len(m.text) for batch in messages for m in batch), token_usage=None)

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, **kwargs):
        metadata = kwargs.get("metadata") or {}
        if isinstance(metadata.get("ls_model_name"), str):
            serialized = {"name": metadata["ls_model_name"]}
        self._start("model", serialized, run_id, parent_run_id,
                    input_messages=len(prompts), input_text_chars=sum(map(len, prompts)), token_usage=None)

    def on_llm_end(self, response, *, run_id, **kwargs):
        self._end(run_id, token_usage=reported_usage(response))

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._end(run_id, "error", error_type=type(error).__name__)

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, **kwargs):
        self._start("tool", serialized, run_id, parent_run_id)

    def on_tool_end(self, output, *, run_id, **kwargs):
        messages = [output] if isinstance(output, ToolMessage) else []
        if isinstance(output, Command) and isinstance(output.update, dict):
            messages = output.update.get("messages", [])
        failed = any(isinstance(m, ToolMessage) and m.status == "error" for m in messages)
        self._end(run_id, "error" if failed else "success")

    def on_tool_error(self, error, *, run_id, **kwargs):
        self._end(run_id, "error", error_type=type(error).__name__)

    def finish(self, error=None):
        for key, span in self.spans.items():
            if key != self.root_id and span["status"] == "running":
                self._end(key, "incomplete")
        self._end(self.root_id, "error" if error is not None else "success",
                  **({"error_type": type(error).__name__} if error is not None else {}))
        spans = list(self.spans.values())
        return spans, performance_summary(spans, duration_ms=self.spans[self.root_id]["duration_ms"])
