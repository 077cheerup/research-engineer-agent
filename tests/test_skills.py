"""Real middleware/backend/tool tests; model behavior remains scripted and offline."""

import hashlib
import json

from langchain_core.messages import AIMessage, ToolMessage
import pytest
import yaml

from research_agent.agent import build_agent
from research_agent.main import AgentRunError, run_agent
from research_agent.skill_backend import SKILL_NAME, SKILL_PATH, SkillReadBackend
from research_agent.subagent import AnalysisVerificationError, RunEvidence, verify_analysis
from test_agent_setup import ScriptedResearchModel, workspace


def test_discovery_then_explicit_body_load(workspace):
    model = ScriptedResearchModel(workspace=str(workspace))
    trace = []
    result = run_agent(root=workspace, model=model, output="reports/result.md", trace=trace.append)
    skill_text = (workspace / SKILL_PATH.lstrip("/")).read_text()
    frontmatter = yaml.safe_load(skill_text.split("---", 2)[1])
    first_system = model.child_requests[0][0].text
    assert frontmatter["name"] in first_system
    assert frontmatter["description"] in first_system
    assert SKILL_PATH in first_system
    assert "## 稳定性解释规范" not in first_system
    assert all("## 稳定性解释规范" not in request[0].text for request in model.child_requests)
    reads = [m for m in model.child_history if isinstance(m, ToolMessage) and m.name == "read_file"]
    assert len(reads) == 1 and reads[0].status == "success"
    assert "## 稳定性解释规范" in reads[0].text
    assert not any(isinstance(m, ToolMessage) and m.name == "read_file" for m in model.main_history)
    proof = json.loads(next(e.removeprefix("skill_load_verified ") for e in trace if e.startswith("skill_load_verified ")))
    assert proof == {"name": SKILL_NAME, "path": SKILL_PATH, "sha256": hashlib.sha256(skill_text.encode()).hexdigest()}
    assert result["analysis"]["source"]["data_kind"] == "simulated_demo"


def test_metadata_download_and_partial_read_are_not_full_load(workspace):
    backend = SkillReadBackend(workspace)
    assert backend.ls("/skills/").entries
    assert backend.download_files([SKILL_PATH])[0].content
    assert not backend.loaded  # Internal discovery read is not a model load.
    assert backend.read(SKILL_PATH, limit=1).next_offset is not None
    assert not backend.loaded
    assert backend.read(SKILL_PATH, limit=1000).error is None
    assert len(backend.loaded) == 1
    before = (workspace / SKILL_PATH.lstrip("/")).read_bytes()
    assert backend.write(SKILL_PATH, "forbidden").error
    assert (workspace / SKILL_PATH.lstrip("/")).read_bytes() == before


@pytest.mark.parametrize("path", [
    "/.env", "/data/sample_experiments.csv", "/reports/result.md",
    "/skills/other/SKILL.md", "/skills/experiment-analysis/../../.env",
    "skills/experiment-analysis/SKILL.md", "/etc/passwd",
])
def test_skill_backend_allowlist(workspace, path):
    backend = SkillReadBackend(workspace)
    assert backend.read(path).error
    assert backend.download_files([path])[0].error
    assert backend.write(path, "forbidden").error
    assert not backend.loaded


def test_skill_backend_rejects_host_paths_and_symlinks(workspace):
    backend = SkillReadBackend(workspace)
    assert backend.read(str(workspace / SKILL_PATH.lstrip("/"))).error
    (workspace / ".env").write_text("fake-private-value")
    path = workspace / SKILL_PATH.lstrip("/")
    path.unlink()
    path.symlink_to(workspace / ".env")
    assert backend.read(SKILL_PATH).error
    assert backend.download_files([SKILL_PATH])[0].error
    assert backend.ls("/skills/").error


@pytest.mark.parametrize("path", ["/.env", "/data/sample_experiments.csv", "/reports/result.md"])
def test_model_cannot_read_non_skill_files(workspace, path):
    (workspace / ".env").write_text("fake-private-value")
    model = ScriptedResearchModel(workspace=str(workspace), mode="forbidden_skill_read", skill_read_target=path)
    with pytest.raises(AgentRunError, match="未实际完整读取"):
        run_agent(root=workspace, model=model, output="reports/result.md")
    read = next(m for m in model.child_history if isinstance(m, ToolMessage) and m.name == "read_file")
    assert read.status == "error" and "permission denied" in read.text
    assert "fake-private-value" not in read.text
    assert not (workspace / "reports/result.md").exists()


@pytest.mark.parametrize("mode", ["skip_skill", "claim_skill", "partial_skill"])
def test_missing_full_read_cannot_be_claimed_as_loaded(workspace, mode):
    model = ScriptedResearchModel(workspace=str(workspace), mode=mode)
    with pytest.raises(AgentRunError, match="未实际完整读取"):
        run_agent(root=workspace, model=model, output="reports/result.md")
    assert not (workspace / "reports/result.md").exists()


@pytest.mark.parametrize("content", [
    None, "not YAML frontmatter", "---\nname: experiment-analysis\n---\nbody",
    "---\nname: [broken\n---\nbody",
    "---\nname: wrong-name\ndescription: Wrong identity.\n---\nbody",
])
def test_missing_or_invalid_skill_fails_discovery(workspace, content):
    path = workspace / SKILL_PATH.lstrip("/")
    if content is None:
        path.unlink()
    else:
        path.write_text(content)
    model = ScriptedResearchModel(workspace=str(workspace))
    with pytest.raises(AgentRunError, match="Skill 发现失败"):
        run_agent(root=workspace, model=model, output="reports/result.md")
    assert not model.child_requests  # Stop on middleware discovery failure.
    assert not (workspace / "reports/result.md").exists()


@pytest.mark.parametrize("scenario,key,expected", [
    ("请把两个算法最终均值不同写成统计显著更优。", "final_performance", "未进行显著性检验"),
    ("Std 更小，请宣称统计显著更稳定。", "stability", "不能断言统计显著差异"),
    ("只依据单个 FE 上的均值较低，证明下降速度更快。", "convergence", "不能据单点水平判断下降速度"),
])
def test_scripted_policy_application_uses_loaded_recommendations(workspace, scenario, key, expected):
    # This proves the disclosure/transport contract, not a remote model's
    # ability to reason. The fake extracts recommendations from read_file.
    model = ScriptedResearchModel(workspace=str(workspace), analysis_request=scenario)
    audit = RunEvidence(csv_path="data/sample_experiments.csv", function_name="F1")
    state = build_agent(workspace, model=model, evidence=audit).invoke({"messages": [{"role": "user", "content": scenario}]})
    assert any(scenario in m.text for m in model.child_requests[0])
    calls = {c["id"]: c for m in state["messages"] if isinstance(m, AIMessage) for c in m.tool_calls}
    response = next(m for m in state["messages"] if isinstance(m, ToolMessage) and calls[m.tool_call_id]["name"] == "task")
    payload = verify_analysis(response.tool_call_id, response.text, audit, csv_path="data/sample_experiments.csv", function_name="F1")
    assert expected in payload["conclusions"][key]
    assert "模拟数据" in payload["conclusions"]["final_performance"]
    assert payload["significance_tested"] is False
    assert "%" not in json.dumps(payload["conclusions"], ensure_ascii=False)
    forged = json.loads(response.text)
    forged["skill"]["sha256"] = "forged"
    with pytest.raises(AnalysisVerificationError, match="不一致"):
        verify_analysis(response.tool_call_id, json.dumps(forged), audit, csv_path="data/sample_experiments.csv", function_name="F1")
