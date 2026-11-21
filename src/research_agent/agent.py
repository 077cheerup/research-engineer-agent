"""Construct the real DeepAgents graph and its narrowly scoped filesystem."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import NotRequired

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.backends.protocol import ReadResult, WriteResult
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.subagents import SubAgentMiddleware
from dotenv import dotenv_values
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from research_agent.prompts import SYSTEM_PROMPT
from research_agent.report import assemble_report
from research_agent.tools.experiment import project_root
from research_agent.subagent import ANALYST_NAME, RunEvidence, make_analyst, verify_analysis


class ConfigurationError(ValueError):
    """Missing model configuration; values and credentials are never logged."""


class ReportState(AgentState):
    report_reminder_sent: NotRequired[bool]


class ReportCompletionMiddleware(AgentMiddleware):
    """Assemble trusted evidence before writing; remind once if writing is omitted.

    Does not retry failed writes or validate wording. State is scoped to one
    invocation and the normal graph recursion limit also bounds the reminder.
    """

    state_schema = ReportState

    def __init__(self, evidence: RunEvidence, relative_output: str | None):
        self.evidence = evidence
        self.relative_output = relative_output

    def verified_payload(self, messages):
        calls = {call["id"]: call for m in messages if isinstance(m, AIMessage) for call in m.tool_calls}
        for message in reversed(messages):
            if not isinstance(message, ToolMessage) or message.status != "success":
                continue
            call = calls.get(message.tool_call_id, {})
            if (call.get("name") != "task"
                    or call["args"].get("subagent_type") != ANALYST_NAME):
                continue
            record = self.evidence.completed.get(message.tool_call_id, {})
            return verify_analysis(
                message.tool_call_id, message.text, self.evidence,
                csv_path=self.evidence.csv_path or record.get("csv_path"),
                function_name=self.evidence.function_name or record.get("function_name"),
            )
        return None

    @hook_config(can_jump_to=["model"])
    def after_model(self, state, runtime):
        messages = state["messages"]
        final = messages[-1]
        if not isinstance(final, AIMessage):
            return None
        if any(call["name"] == "write_file" for call in final.tool_calls):
            payload = self.verified_payload(messages[:-1])
            if payload is not None:
                # Replace the AIMessage by its ID before the real tool node.
                # The effective submitted arguments therefore remain visible
                # to verify_run, which still compares them exactly to disk.
                submitted = final.model_copy(deep=True)
                for call in submitted.tool_calls:
                    if call["name"] == "write_file":
                        call["args"]["content"] = assemble_report(call["args"]["content"], payload["analysis"])
                return {"messages": [submitted]}
            return None
        if (self.relative_output is None or state.get("report_reminder_sent")
                or final.tool_calls or final.invalid_tool_calls
                or final.response_metadata.get("finish_reason") not in (None, "stop")):
            return None
        calls = {call["id"]: call for m in messages if isinstance(m, AIMessage) for call in m.tool_calls}
        # A write attempt, even a failed or premature one, goes to the existing
        # verifier. Never overwrite a file or mask a real write failure.
        if any(call["name"] == "write_file" for call in calls.values()):
            return None
        if self.verified_payload(messages) is not None:
            return {
                "report_reminder_sent": True,
                "messages": [HumanMessage(content=(
                    "本次任务尚未完成：当前分析已核验，但你尚未调用 write_file。"
                    "历史报告仅供参考，不能替代本次输出。请使用刚才 task 的可信 analysis 和 conclusions，"
                    f"调用 write_file(file_path=\"/{self.relative_output}\", content=完整 Markdown 报告)，"
                    "保留空的可信证据标记区域，由程序填入完整数值证据；保留模拟数据说明，无需重新分析。"
                    "等待写入成功后再返回摘要、本次报告相对路径及主机绝对路径。"
                ))],
                "jump_to": "model",
            }
        return None


def report_path(output: str, root: Path) -> Path:
    """Use one simple, stable virtual namespace: /reports/<safe-name>.md."""
    value = output.removeprefix("/")
    if not re.fullmatch(r"reports/[A-Za-z0-9][A-Za-z0-9_.-]*\.md", value):
        raise ValueError("报告路径必须是 reports/<安全文件名>.md（字母、数字、_、-、.）")
    root = root.resolve()
    candidate = root / value
    if (root / "reports").resolve() != root / "reports" or candidate.resolve() != candidate:
        raise ValueError("报告路径不能经过符号链接")
    return candidate


class ReportFilesystemBackend(FilesystemBackend):
    """Keep the built-in write_file, but limit writes and never overwrite.

    DeepAgents 0.7.40's base write() truncates existing files. Exclusive
    creation implements the task's stronger no-overwrite requirement.
    No third-party sources are modified.
    """

    def __init__(self, root: Path):
        self.workspace = root.resolve()
        super().__init__(root_dir=self.workspace, virtual_mode=True)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        # The CSV tool owns reads. This also covers dotfiles: the framework's
        # /** permission glob does not match .env in version 0.7.40.
        return ReadResult(error="permission denied: delegate CSV analysis to experiment-analyst")

    def write(self, file_path: str, content: str) -> WriteResult:
        try:
            destination = report_path(file_path, self.workspace)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("x", encoding="utf-8", newline="") as handle:
                handle.write(content)
            return WriteResult(path="/" + destination.relative_to(self.workspace).as_posix())
        except FileExistsError:
            return WriteResult(error="报告已存在，拒绝覆盖；请指定新的 reports/*.md 文件名")
        except ValueError as exc:
            return WriteResult(error=f"报告路径错误：{exc}；write_file.file_path 请使用 /reports/<name>.md 虚拟路径，不要使用主机绝对路径")
        except OSError as exc:
            return WriteResult(error=f"报告写入失败 ({type(exc).__name__}, errno={exc.errno})；请检查 reports/ 目录是否可写")


def load_model(root: Path) -> ChatOpenAI:
    """Read project .env, with explicit LLM_* environment overrides."""
    file_config = dotenv_values(root / ".env", interpolate=False)
    names = ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")
    values = {name: os.environ.get(name, file_config.get(name) or "").strip() for name in names}
    missing = [name for name in names if not values[name]]
    if missing:
        raise ConfigurationError("缺少模型配置：" + ", ".join(missing) + "；请设置项目 .env")
    return ChatOpenAI(
        model=values["LLM_MODEL"],
        api_key=values["LLM_API_KEY"],
        base_url=values["LLM_BASE_URL"],
        use_responses_api=False,
        max_retries=0,
        timeout=60,
    )


def build_agent(
    root: Path | str | None = None, *, model: BaseChatModel | None = None,
    analyst_model: BaseChatModel | None = None, evidence: RunEvidence | None = None,
    max_steps: int = 24,
    report_output: str | None = None,
):
    """Use the real create_deep_agent; injectable models enable offline tests."""
    workspace = Path(root).resolve() if root is not None else project_root()
    backend = ReportFilesystemBackend(workspace)
    main_model = model if model is not None else load_model(workspace)
    run_evidence = evidence if evidence is not None else RunEvidence()
    analyst = make_analyst(
        workspace, analyst_model if analyst_model is not None else main_model,
        run_evidence, max_steps=max_steps,
    )
    relative_output = (report_path(report_output, workspace).relative_to(workspace).as_posix()
                       if report_output is not None else None)
    return create_deep_agent(
        model=main_model,
        tools=[],
        backend=backend,
        system_prompt=SYSTEM_PROMPT,
        # Public same-name replacement removes the default general-purpose
        # agent from the final task tool without a global profile registration.
        middleware=[ReportCompletionMiddleware(run_evidence, relative_output), SubAgentMiddleware(
            backend=backend, subagents=[analyst],
            task_description="委派实验分析任务。description 中必须包含 csv_path 和 function_name。只可使用以下子 Agent：\n{available_agents}\n返回经过程序核验的完整统计 JSON 和中文结论。",
        ), FilesystemMiddleware(
            # 0.7.40 requires read_file; the backend denies every read.
            backend=backend, tools=["read_file", "write_file"],
            custom_tool_descriptions={
                "read_file": "此项目不提供通用文件读取，本工具总是拒绝。分析 CSV 请委派 experiment-analyst；新建报告前无需读取目标文件。",
                "write_file": "新建 Markdown 报告。file_path 必须是 /reports/<name>.md 虚拟路径，例如 /reports/F1_comparison.md；禁止传主机绝对路径。无需先调用 read_file；目标存在时会拒绝覆盖。content 是报告草稿，必须含一组 <!-- BEGIN_ANALYSIS_EVIDENCE --> 与 <!-- END_ANALYSIS_EVIDENCE --> 空标记区域；程序从本次可信分析填入完整来源、SHA256 和统计证据。不要自行复制证据。",
            },
        )],
        name="mini-research-engineer",
    )
