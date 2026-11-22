"""Terminal review of one prepared write_file, using real interrupt/resume."""

from copy import deepcopy
import json

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from research_agent.agent import ReportTarget, report_path


class ApprovalError(RuntimeError):
    """Review was rejected or could not be applied safely."""


def terminal_review(action: dict) -> dict:
    """Never approve on empty input/EOF. Editing is restricted to file_path."""
    print(f"\n待审批操作：{action['name']}（新建 Markdown 报告）")
    print(f"文件路径：{action['args']['file_path']}")
    print(f"报告字符数：{len(action['args']['content'])}；输入 view 可查看完整内容")
    while True:
        try:
            choice = input("请选择 approve / edit / reject：").strip().lower()
            if choice == "view":
                print(action["args"]["content"])
                continue
            if choice == "edit":
                return {"type": "edit", "file_path": input("新路径（reports/<安全文件名>.md）：").strip()}
            if choice in {"approve", "reject"}:
                return {"type": choice}
        except (EOFError, KeyboardInterrupt):
            print("\n审批输入已结束，拒绝写入。")
            return {"type": "reject"}
        print("请输入 approve、edit、reject 或 view；空输入不会批准。")


def review_write(graph, config, interrupt, *, root, target: ReportTarget, reviewer=None, trace=None):
    """Bind the human decision to the paused real call, never to model claims."""
    request = interrupt.value
    actions = request.get("action_requests", [])
    if len(actions) != 1 or actions[0].get("name") != "write_file":
        raise ApprovalError("本实验一次只支持审批一个 write_file；未执行写入")
    action = actions[0]
    snapshot = graph.get_state(config).values
    last_ai = next(m for m in reversed(snapshot["messages"]) if isinstance(m, AIMessage))
    calls = [c for c in last_ai.tool_calls if c["name"] == "write_file"]
    if len(calls) != 1 or calls[0]["args"] != action["args"]:
        raise ApprovalError("审批内容与实际挂起调用不一致")
    call = calls[0]
    if trace:
        trace("hitl_interrupt " + json.dumps({"tool_call_id": call["id"], "operation": "write_file",
              "file_path": action["args"]["file_path"], "content_chars": len(action["args"]["content"])}, ensure_ascii=False))
    while True:
        decision = (reviewer or terminal_review)(deepcopy(action))
        try:
            kind = decision.get("type")
            if kind not in {"approve", "edit", "reject"}:
                raise ApprovalError("只允许 approve / edit / reject；不能伪造工具响应")
            executed = deepcopy(call)
            if kind == "edit":
                path = report_path(decision["file_path"], root)
                if path.exists():
                    raise ApprovalError("编辑后的报告已存在，拒绝覆盖")
                executed["args"]["file_path"] = "/" + path.relative_to(root).as_posix()
                target.destination = path
                response = {"type": "edit", "edited_action": {"name": "write_file", "args": executed["args"]}}
            elif kind == "reject":
                target.rejected = True
                target.rejected_call_id = call["id"]
                response = {"type": "reject", "message": "人工拒绝本轮报告写入；停止执行，不要重试或改名。"}
            else:
                response = {"type": "approve"}
        except (ValueError, KeyError, ApprovalError) as exc:
            if reviewer is not None:
                raise ApprovalError(f"审批参数无效：{exc}") from exc
            print(f"审批未提交：{exc}；请重新选择。")
            continue
        break
    # This record is not a success receipt. verify_run still requires the real
    # tool's success message, exact submitted bytes, disk file and evidence.
    target.effective_calls[call["id"]] = executed
    if trace:
        trace("hitl_decision " + json.dumps({"tool_call_id": call["id"], "decision": kind,
              "effective_path": executed["args"]["file_path"]}, ensure_ascii=False))
    return Command(resume={"decisions": [response]})


def stream_with_approval(graph, inputs, config, *, root, target, enabled=False, reviewer=None, trace=None):
    """Keep one graph/checkpoint/thread and RunEvidence across all resumes."""
    next_input = inputs
    while True:
        pending = []
        for state in graph.stream(next_input, config=config, stream_mode="values"):
            if "__interrupt__" in state:
                pending.extend(state["__interrupt__"])
            if "messages" in state:
                yield state
                if target.rejected and any(
                    isinstance(m, ToolMessage) and m.status == "error"
                    and m.tool_call_id == target.rejected_call_id for m in state["messages"]
                ):
                    raise ApprovalError("人工已拒绝本轮报告写入；未验收报告，未保存 Memory")
        if not pending:
            return
        if not enabled or len(pending) != 1:
            raise ApprovalError("遇到未配置或不支持的审批中断；未恢复执行")
        next_input = review_write(graph, config, pending[0], root=root, target=target, reviewer=reviewer, trace=trace)
