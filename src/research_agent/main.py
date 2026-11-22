"""CLI and observable acceptance checks for the analyze -> write loop."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

from deepagents.middleware.summarization import SUMMARIZATION_EVENT_KEY
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from openai import APIConnectionError, APIStatusError, APITimeoutError

from research_agent.agent import ConfigurationError, build_agent, report_path
from research_agent.tools.experiment import project_root, resolve_csv_path
from research_agent.subagent import ANALYST_NAME, AnalysisVerificationError, RunEvidence, normalize_evidence, verify_analysis
from research_agent.memory import ResearchMemoryError, ResearchMemoryStore, history_summary
from research_agent.report import ReportAssemblyError, verified_summary
from research_agent.summarization import HistoryArchiveBackend, HistoryArchiveError


class AgentRunError(RuntimeError):
    """The graph ran without satisfying the observable research contract."""


def verify_report_evidence(saved: str, analysis: dict) -> None:
    """Allow separate report sections; keep each trusted evidence block intact.

    The deterministic formatter emits provenance, final statistics and mean
    convergence as three blank-line-separated blocks. This is not a check
    for scattered numbers: complete source text and both full tables must match.
    """
    report = normalize_evidence(saved)
    blocks = normalize_evidence(analysis["evidence_markdown"]).split("\n\n")
    names = ("来源信息", "最终统计表", "平均收敛表")
    missing = [name for name, block in zip(names, blocks, strict=True) if block not in report]
    if missing:
        raise AgentRunError(
            "报告数值证据缺失或被修改：" + "、".join(missing)
            + "（允许分章节放置，需保留完整证据块；仅忽略行尾空白）"
        )


def verify_run(
    messages: list, *, root: Path, csv: Path, function: str, destination: Path,
    run_evidence: RunEvidence,
) -> dict[str, Any]:
    """Require real ToolMessages, correct ordering, evidence, and disk content.

    Exact evidence is checked. Natural-language correctness requires human
    review and does not block this technical acceptance check.
    """
    calls = {}
    payload = None
    report_payload = None
    written_content = None
    last_write_error = None
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                calls[call["id"]] = (call, payload)
        if isinstance(message, ToolMessage) and message.name == "write_file" and message.status == "error":
            last_write_error = message.text
        if not isinstance(message, ToolMessage) or message.status != "success":
            continue
        call, preceding_payload = calls.get(message.tool_call_id, ({}, None))
        if call.get("name") == "task":
            args = call["args"]
            if args.get("subagent_type") != ANALYST_NAME:
                continue
            payload = verify_analysis(
                message.tool_call_id, message.text, run_evidence,
                csv_path=csv.relative_to(root).as_posix(), function_name=function,
            )
        if message.name == "write_file" and call.get("name") == message.name and preceding_payload:
            args = call["args"]
            if report_path(args["file_path"], root) == destination:
                written_content = args["content"]
                report_payload = preceding_payload
    if payload is None:
        raise AgentRunError("Main 未成功调用 task 委派 experiment-analyst 并取得可信分析结果")
    if written_content is None:
        detail = f"；工具错误：{last_write_error}" if last_write_error else ""
        final = messages[-1] if messages else None
        if not last_write_error and isinstance(final, AIMessage) and not final.tool_calls:
            detail += (f"；Main 已结束回答（finish_reason={final.response_metadata.get('finish_reason', 'unknown')}，"
                       f"回答字符数={len(final.text)}，无法解析的工具调用数={len(final.invalid_tool_calls)}）")
        raise AgentRunError("未观察到分析成功之后的 write_file 成功调用；报告未验证保存" + detail)
    evidence = report_payload["analysis"]
    if not destination.is_file():
        raise AgentRunError("write_file 返回后预期报告文件不存在")
    saved = destination.read_text(encoding="utf-8")
    if saved != written_content:
        raise AgentRunError("磁盘报告与实际 write_file 调用内容不一致")
    verify_report_evidence(saved, evidence)
    if "模拟数据" not in saved:
        raise AgentRunError("报告缺少模拟数据声明")
    final = messages[-1] if messages else None
    if not isinstance(final, AIMessage) or final.tool_calls or not final.text.strip():
        raise AgentRunError("缺少 Agent 最终回答")
    relative = destination.relative_to(root).as_posix()
    return {"summary": verified_summary(report_payload["conclusions"], relative, str(destination)),
            "model_summary": final.text, "report_path": str(destination), "analysis": evidence,
            "conclusions": report_payload["conclusions"], "skill": report_payload["skill"]}


def run_agent(
    *, csv_path: str = "data/sample_experiments.csv", function: str = "F1",
    output: str = "reports/F1_comparison.md", root: Path | str | None = None,
    model: BaseChatModel | None = None, analyst_model: BaseChatModel | None = None, max_steps: int = 24,
    trace: Callable[[str], None] | None = None,
    memory_limit: int = 3, memory_max_chars: int = 4000,
    summary_model: BaseChatModel | None = None,
    summary_trigger_messages: int | None = None, summary_keep_messages: int = 6,
) -> dict[str, Any]:
    """Invoke a bounded real graph; no deterministic substitute for the agent."""
    workspace = Path(root).resolve() if root is not None else project_root()
    csv = resolve_csv_path(csv_path, workspace)
    destination = report_path(output, workspace)
    if destination.exists():
        raise AgentRunError("目标报告已存在，拒绝覆盖；请通过 --output 指定新文件名")
    if max_steps < 1:
        raise ValueError("max_steps 必须为正整数")
    if memory_max_chars < 0:
        raise ValueError("memory_max_chars 不能为负")
    try:
        memory = ResearchMemoryStore(workspace)
        # The current CLI task explicitly compares A/B. Retrieval by both
        # algorithm names keeps unrelated function/algorithm histories out.
        historical = memory.retrieve(function_name=function, algorithms=["A", "B"], limit=memory_limit)
        historical_text = history_summary(historical, max_chars=memory_max_chars)
    except ResearchMemoryError as exc:
        raise AgentRunError(f"历史 Memory 读取或验证失败：{exc}") from exc
    if trace:
        trace(f"memory_retrieved matched={len(historical)} injected_chars={len(historical_text)}")
    run_evidence = RunEvidence(csv_path=csv.relative_to(workspace).as_posix(), function_name=function)
    graph = build_agent(workspace, model=model, analyst_model=analyst_model, evidence=run_evidence,
                        max_steps=max_steps, report_output=destination.relative_to(workspace).as_posix(),
                        summary_model=summary_model, summary_trigger_messages=summary_trigger_messages,
                        summary_keep_messages=summary_keep_messages)
    request = (
        f"分析算法 A 和算法 B 在 CEC2013 {function} 上的模拟实验结果，比较最终性能、稳定性和收敛趋势，"
        f"生成中文 Markdown 报告。CSV 参数：{json.dumps(csv.relative_to(workspace).as_posix())}；"
        f"function_name 参数：{json.dumps(function)}。本多步骤任务先用 write_todos 创建计划，再用 task 委派 experiment-analyst，"
        "description 中须包含以上两个精确参数。取得完整可信分析结果后再写入 "
        f"/{destination.relative_to(workspace).as_posix()}。write_file.file_path 必须精确使用 "
        f"{json.dumps('/' + destination.relative_to(workspace).as_posix())}，无需先读取目标。"
        f"实际主机绝对路径为 {destination}，仅供程序在验收后展示，禁止作为工具的 file_path 参数。"
        "最终回复只需给出简短摘要；实际保存路径由 Python 在验收后提供，不要另选文件名。"
    )
    seen = set()
    call_names = {}
    plan_created = False
    messages = []
    reminder_seen = False
    last_summary = None
    inputs = []
    if historical_text:
        inputs.append({"role": "user", "content": historical_text})
    inputs.append({"role": "user", "content": request})
    try:
        for state in graph.stream(
            {"messages": inputs},
            config={"recursion_limit": max_steps}, stream_mode="values",
        ):
            messages = state["messages"]
            event = state.get(SUMMARIZATION_EVENT_KEY)
            if event:
                if event.get("file_path") is None:
                    raise HistoryArchiveError("摘要已触发，但原始历史归档失败；本次运行不能验收或保存 Memory")
                signature = (event["cutoff_index"], event.get("created_at_index"), event["file_path"])
                if trace and signature != last_summary:
                    trace("summarization_applied " + json.dumps({
                        "cutoff_index": event["cutoff_index"], "created_at_index": event.get("created_at_index"),
                        "history_path": event["file_path"],
                        "archive_path": str(HistoryArchiveBackend(workspace).archive_path(event["file_path"])),
                    }, ensure_ascii=False))
                last_summary = signature
            if trace:
                if state.get("report_reminder_sent") and not reminder_seen:
                    trace("report_completion_reminder attempt=1")
                    reminder_seen = True
                for message in messages:
                    if message.id in seen:
                        continue
                    seen.add(message.id)
                    if isinstance(message, AIMessage):
                        if not message.tool_calls:
                            trace("main_reply " + json.dumps({
                                "finish_reason": message.response_metadata.get("finish_reason", "unknown"),
                                "text_chars": len(message.text), "preview": message.text[:240],
                                "invalid_tool_calls": len(message.invalid_tool_calls),
                            }, ensure_ascii=False))
                        for call in message.tool_calls:
                            call_names[call['id']] = call['name']
                            trace(f"tool_call {call['name']} id={call['id']}")
                            if call['name'] == "task":
                                trace("task_target " + str(call['args'].get('subagent_type')))
                            if call['name'] in {"read_file", "write_file"}:
                                trace("tool_path " + json.dumps({"file_path": call['args'].get('file_path')}, ensure_ascii=False))
                    elif isinstance(message, ToolMessage):
                        name = message.name or call_names.get(message.tool_call_id, "unknown")
                        trace(f"tool_result {name} id={message.tool_call_id} status={message.status}")
                        if name == "write_todos" and message.status == "success" and isinstance(state.get("todos"), list):
                            event = "plan_updated" if plan_created else "plan_created"
                            trace(event + " " + json.dumps({"tool_call_id": message.tool_call_id, "todos": state["todos"]}, ensure_ascii=False))
                            plan_created = True
                        if message.status == "error" and name in {"read_file", "write_file", "write_todos"}:
                            trace(f"tool_error {name}: {message.text}")
                        if name == "task" and message.tool_call_id in run_evidence.completed:
                            trace(f"subagent_analysis_verified subagent={ANALYST_NAME} task_id={message.tool_call_id}")
                            skill = run_evidence.completed[message.tool_call_id]["payload"]["skill"]
                            trace("skill_load_verified " + json.dumps(skill, ensure_ascii=False))
                            trace("analysis_conclusions " + json.dumps(run_evidence.completed[message.tool_call_id]["payload"]["conclusions"], ensure_ascii=False))
                            trace("analysis_evidence " + json.dumps(run_evidence.completed[message.tool_call_id]["analysis"], ensure_ascii=False, allow_nan=False))
    except GraphRecursionError as exc:
        raise AgentRunError("达到运行步数上限；可能存在重复 Tool Calling，报告未验收") from exc
    except (APIConnectionError, APITimeoutError) as exc:
        raise AgentRunError(f"模型服务连接或超时失败 ({type(exc).__name__})；请检查 LLM_BASE_URL 和网络") from exc
    except APIStatusError as exc:
        raise AgentRunError(
            f"模型服务返回 HTTP {exc.status_code}；检查 LLM_* 配置及 OpenAI Chat Completions "
            "的 tools/tool_calls 支持。未退化为伪 Agent，报告未验收。"
        ) from exc
    except NotImplementedError as exc:
        raise AgentRunError("模型未实现 Tool Calling/bind_tools；请选择支持工具调用的服务") from exc
    except AnalysisVerificationError as exc:
        raise AgentRunError(f"子 Agent 分析验证失败：{exc}") from exc
    except ReportAssemblyError as exc:
        raise AgentRunError(f"报告可信证据组装失败：{exc}；未确认保存，未写入 Memory") from exc
    except HistoryArchiveError as exc:
        raise AgentRunError(f"历史归档失败：{exc}") from exc
    try:
        verified = verify_run(messages, root=workspace, csv=csv, function=function, destination=destination, run_evidence=run_evidence)
    except AnalysisVerificationError as exc:
        raise AgentRunError(f"子 Agent 分析验证失败：{exc}") from exc
    try:
        verified["memory"] = memory.save_verified(verified)
    except ResearchMemoryError as exc:
        raise AgentRunError(f"报告验收已通过，但 Memory 保存失败：{exc}") from exc
    if trace:
        trace("memory_saved " + json.dumps(verified["memory"], ensure_ascii=False))
    return verified


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mini Research Engineer Agent — Phase 2.3")
    parser.add_argument("--csv", default="data/sample_experiments.csv")
    parser.add_argument("--function", default="F1")
    parser.add_argument("--output", default="reports/F1_comparison.md")
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--summary-trigger-messages", type=int, help="验证用摘要消息阈值；默认使用框架模型感知阈值")
    parser.add_argument("--summary-keep-messages", type=int, default=6, help="指定消息阈值时保留的近期消息数，默认 6")
    parser.add_argument("--trace", action="store_true", help="显示工具、计划、摘要归档及统计事件，不输出模型配置或密钥")
    args = parser.parse_args(argv)
    try:
        result = run_agent(
            csv_path=args.csv, function=args.function, output=args.output,
            max_steps=args.max_steps, trace=print if args.trace else None,
            summary_trigger_messages=args.summary_trigger_messages, summary_keep_messages=args.summary_keep_messages,
        )
    except (ConfigurationError, AgentRunError, ValueError, OSError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        # Some compatible providers return malformed responses. Do not print
        # their raw bodies, which may echo credentials or request headers.
        print(f"错误：Agent 执行异常 ({type(exc).__name__})；请检查服务的 tools/tool_calls 协议兼容性，报告未验收", file=sys.stderr)
        return 1
    print(result["summary"])
    print(f"已核验报告：{result['report_path']}")
    print(f"已保存研究记忆：{result['memory']['experiment_id']} → {result['memory']['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
