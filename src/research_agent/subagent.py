"""One isolated analyst, with execution evidence bound to each real task ID."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from deepagents.middleware.subagents import CompiledSubAgent
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.skills import SkillsMiddleware
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda

from research_agent.prompts import ANALYST_PROMPT
from research_agent.tools.experiment import make_analyze_experiment
from research_agent.skill_backend import SKILL_NAME, SKILL_PATH, SkillReadBackend

ANALYST_NAME = "experiment-analyst"


class AnalysisVerificationError(ValueError):
    """A child result lacks real tool execution or complete evidence."""


def normalize_evidence(text: str) -> str:
    """Ignore Markdown hard-break spaces and platform line endings only."""
    return "\n".join(line.rstrip() for line in text.splitlines())


@dataclass
class RunEvidence:
    """Fresh per run; never put in model state, prompts, or module globals.

    Only the compiled analyst wrapper publishes here, after checking a real
    tool callback and the child's output. Keys are stamped by DeepAgents'
    task runtime, not supplied in the model's description text.
    """

    csv_path: str | None = None
    function_name: str | None = None
    completed: dict[str, dict[str, Any]] = field(default_factory=dict)


def _parse_child_reply(message: Any) -> dict:
    if not isinstance(message, AIMessage) or message.tool_calls:
        raise AnalysisVerificationError("子 Agent 缺少最终结构化分析结果")
    text = message.text.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[8:-3].strip()
    try:
        reply = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise AnalysisVerificationError("子 Agent 必须返回 JSON 及数据来源引用，不能只返回自然语言摘要") from exc
    if not isinstance(reply, dict):
        raise AnalysisVerificationError("子 Agent 输出必须是 JSON 对象")
    return reply


def make_analyst(root: Path, model: BaseChatModel, evidence: RunEvidence, *, max_steps: int = 24) -> CompiledSubAgent:
    """Official CompiledSubAgent with CSV analysis and one read-only Skill.

    Every task invocation creates its own callback list and analysis graph.
    The wrapper passes only a verified structured_response to task, retaining
    neither internal ToolMessages nor internal state in Main's context.
    """
    def invoke(state: dict, config: RunnableConfig) -> dict:
        task_id = config.get("metadata", {}).get("lc_subagent_tool_call_id")
        if not task_id:
            raise AnalysisVerificationError("缺少 DeepAgents task 调用 ID，无法绑定可信分析记录")
        records = []
        skill_backend = SkillReadBackend(root)

        def record(csv_path: str, function_name: str, result: dict) -> None:
            # Called only inside the deterministic tool, after success.
            records.append(deepcopy({"csv_path": csv_path, "function_name": function_name, "analysis": result}))

        child = create_agent(
            model=model,
            tools=[make_analyze_experiment(root, on_success=record)],
            system_prompt=ANALYST_PROMPT,
            middleware=[
                FilesystemMiddleware(backend=skill_backend, tools=["read_file"]),
                SkillsMiddleware(backend=skill_backend, sources=["/skills/"]),
            ],
            name=ANALYST_NAME,
        )
        # values exposes the middleware's private discovery state to this
        # wrapper; invoke's public output deliberately omits skills_metadata.
        for result in child.stream({"messages": state["messages"]}, config={**config, "recursion_limit": max_steps}, stream_mode="values"):
            if result.get("skills_metadata") is not None and not any(
                item["name"] == SKILL_NAME and item["path"] == SKILL_PATH
                for item in result["skills_metadata"]
            ):
                raise AnalysisVerificationError("Skill 发现失败：experiment-analysis 缺失或 YAML frontmatter 无效")
        if not records:
            raise AnalysisVerificationError("子 Agent 未真正成功执行 analyze_experiment")
        read_calls = {
            call["id"]: call for message in result["messages"] if isinstance(message, AIMessage)
            for call in message.tool_calls if call["name"] == "read_file"
        }
        loaded = any(
            isinstance(message, ToolMessage) and message.status == "success"
            and message.tool_call_id in read_calls
            and read_calls[message.tool_call_id]["args"].get("file_path") == SKILL_PATH
            for message in result["messages"]
        )
        if not loaded or not skill_backend.loaded:
            raise AnalysisVerificationError("子 Agent 未实际完整读取 experiment-analysis Skill，不能声称加载成功；请 read_file(limit=1000)")
        skill = skill_backend.loaded[-1]
        reply = _parse_child_reply(result["messages"][-1])
        trusted = records[-1]
        if ((evidence.csv_path is not None and trusted["csv_path"] != evidence.csv_path)
                or (evidence.function_name is not None and trusted["function_name"] != evidence.function_name)):
            raise AnalysisVerificationError("子 Agent 实际分析的 CSV 或 Function 与本次任务不符")
        if not isinstance(reply.get("source_sha256"), str):
            raise AnalysisVerificationError("子 Agent 缺少数据来源引用 source_sha256，不能只返回摘要")
        if reply["source_sha256"] != trusted["analysis"]["source"]["sha256"]:
            raise AnalysisVerificationError("子 Agent 的 source_sha256 与真实工具记录不一致")
        conclusions = reply.get("conclusions")
        if not isinstance(conclusions, dict) or any(
            not isinstance(conclusions.get(key), str) or not conclusions[key].strip()
            for key in ("final_performance", "stability", "convergence")
        ):
            raise AnalysisVerificationError("子 Agent 缺少最终性能、稳定性或收敛趋势的自然语言结论")
        if reply.get("significance_tested") is not False:
            raise AnalysisVerificationError("子 Agent 必须明确声明 significance_tested=false")
        payload = {"analysis": trusted["analysis"], "conclusions": conclusions, "significance_tested": False, "skill": skill}
        # Statistics AND Markdown evidence are transported from the real tool,
        # rather than asking a model to regenerate either representation.
        json.dumps(payload, allow_nan=False)
        evidence.completed[task_id] = deepcopy({**trusted, "payload": payload})
        return {"messages": [], "structured_response": payload}

    return {
        "name": ANALYST_NAME,
        "description": "分析 data/ 中的模拟实验 CSV。任务必须给出 csv_path 和 function_name；返回经过程序验证的完整统计证据、收敛曲线和中文解释，不写报告。",
        "runnable": RunnableLambda(invoke),
    }


def verify_analysis(task_id: str, returned_text: str, evidence: RunEvidence, *, csv_path: str, function_name: str) -> dict:
    """Layer 1: match Main's task result to trusted run-local execution."""
    record = evidence.completed.get(task_id)
    if record is None:
        raise AnalysisVerificationError("task 没有绑定的真实子 Agent 分析执行记录")
    if record["csv_path"] != csv_path or record["function_name"] != function_name:
        raise AnalysisVerificationError("子 Agent 实际分析的 CSV 或 Function 与本次任务不符")
    try:
        returned = json.loads(returned_text)
    except (TypeError, ValueError) as exc:
        raise AnalysisVerificationError("task 返回结果不是结构化 JSON，数值证据无法验证") from exc
    if returned != record["payload"]:
        raise AnalysisVerificationError("task 返回结果与可信分析记录不一致，证据丢失或被修改")
    return deepcopy(record["payload"])
