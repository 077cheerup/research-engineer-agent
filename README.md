# Mini Research Engineer Agent — Phase 2.1

基于 **DeepAgents 0.7.40** 的增量科研分析闭环：Main 用内置 `task` 委派 `experiment-analyst`，子 Agent 调用确定性分析 Tool；经过程序验证后返回完整统计与解释，Main 用内置 `write_file` 保存报告。Main 仍由 `create_deep_agent()` 创建，子图通过官方 `CompiledSubAgent` 接口接入。

本项目是 DeepAgents / Agent Harness 学习路线 Level 7 Phase 2.1，在 Phase 1 上增量集成一个分析子 Agent，统计公式和 CSV 格式不变。这里研究的内容只有算法 A/B 在 F1 上的模拟结果；不运行真实 CEC2013 函数，不代表正式 CEC2013 实验，不做显著性检验。

## 安装、配置与运行

需要 Python 3.11+ 和 uv。依赖和具体版本固定在 `uv.lock`；`.python-version` 指定 3.11。

```bash
uv sync
cp .env.example .env
# 编辑 .env，填入自己服务的三个配置
uv run python -m research_agent.main
```

`.env` 使用固定变量名：

```dotenv
LLM_BASE_URL=https://your-provider.example/v1
LLM_API_KEY=replace-with-your-key
LLM_MODEL=your-tool-calling-model
```

从项目根目录的 `.env` 加载，已有同名环境变量优先。不读取 shell 当前目录的 `.env`。`ChatOpenAI` 显式接收 `model`、`api_key`、`base_url`，使用 Chat Completions（`use_responses_api=False`），超时 60 秒，SDK 不自动重试。不主动启用 JSON mode、强制 tool_choice、并行工具调用或其他服务可能不支持的参数。

`.env`、环境变体、生成报告、虚拟环境均被 Git 忽略。请勿把真实密钥写进 README、测试、命令参数、日志或 Git。`--trace` 只显示工具事件和统计数据，不输出模型配置或密钥。

```bash
uv run python -m research_agent.main \
  --csv data/sample_experiments.csv \
  --function F1 \
  --output reports/F1_comparison.md \
  --trace
```

默认输出已存在时明确失败；更换 `--output reports/F1_comparison_2.md` 后再运行。输出仅允许 `reports/<文件名>.md`，文件名使用字母、数字、`_`、`-`、`.`，不允许嵌套目录。工具虚拟路径为 `/reports/...`，实际文件保存在项目的 `reports/`。最终回答要求同时包含项目相对路径和主机绝对路径。

CLI 默认 `--max-steps 24`，Main 与每次子图运行分别使用这一上限（LangGraph 图步数，不是 token 数或整个嵌套执行的总预算），防止各自无限 Tool Calling。缺失配置、服务错误、漏调工具、文件不存在、数值证据不一致时退出码为 1；只有验收通过才输出“已核验报告”。如果失败前已经生成文件，该文件是**未通过验收的产物**，不会被自动删除或标为成功；再次运行需换文件名。

当前项目根目录通过源码位置定位，与 shell 的当前目录无关。源码/uv editable 安装是支持的使用方式；也可显式调用 `build_agent(root=...)` / `run_agent(root=...)` 绑定工作区。此版本不面向脱离源码目录的 wheel 分发。

## 数据格式与模拟数据

```csv
algorithm,function,seed,fe,best_fitness
A,F1,1,100,150.0
A,F1,1,200,120.0
B,F1,1,100,140.0
B,F1,1,200,110.0
```

这是格式片段；完整分析每算法至少要有两个种子，才能定义样本标准差。字段要求：

| 字段 | 含义与约束 |
| --- | --- |
| algorithm | 非空算法标签；不能包含 Markdown 表格分隔符、换行、反引号 |
| function | 非空函数标签，例如 F1 |
| seed | 正整数种子编号 |
| fe | 正整数函数评估次数；最大 `2**53-1` |
| best_fitness | 有限数值；按 FE 排序的每条 best-so-far 轨迹非递增 |

拒绝缺列、缺失值、非有限值、非法数据类型、重复 `(algorithm,function,seed,fe)`、不一致 FE 网格。每个函数内各算法必须有相同种子集合，各种子必须有完全相同 FE 网格，以保持预算和重复次数可比。所有行先验证，再选择函数；排序不依赖 CSV 行序。非递增验证默认绝对容差 `1e-10`，可在纯分析函数中配置。

`data/sample_experiments.csv` **全部是模拟数据**，不是 CEC2013 的真实函数值、误差或官方实验。本阶段继续把用户输入视作模拟演示数据，不适用于正式 benchmark 结果的直接发表。

- A、B 两算法；F1 一个函数；每算法种子 1–5。
- 每轨迹 100、200、…、1000 FE，共 10 个采样点、100 行记录。
- **演示预算 1,000 FE**；未调用任何真实 benchmark。
- 固定 NumPy 随机种子 `20261119`，指数下降曲线加噪声后使用累计最小值。

重新生成：

```bash
uv run python scripts/generate_sample.py
```

确定性重算的最终统计示例（仅描述性比较）：

| 算法 | 最终 Mean | Sample Std（ddof=1） |
| --- | --- | --- |
| A | 20.738583304 | 0.934708603623 |
| B | 13.60833829 | 0.592108552027 |

本模拟样本中 B 的最终均值和跨种子离散程度较低；这不支持统计显著性或真实 CEC2013 泛化结论。平均曲线仅支持采样点上的趋势比较，不能推断采样点之间的轨迹。

## 架构和职责

```text
用户 -> Main（create_deep_agent）
          -> task(subagent_type="experiment-analyst", description="CSV + Function")
             -> 隔离的 Analysis Subagent（create_agent）
                -> analyze_experiment -> 原有确定性统计
                -> JSON 证据片段 + 中文结论
             -> 程序核对真实工具记录、证据及参数
             -> task ToolMessage：完整可信 analysis + conclusions
          -> 内置 write_file -> ReportFilesystemBackend -> reports/*.md
CLI -> 两层验证 -> 摘要和真实路径
```

| 权限 | Main Agent | experiment-analyst |
| --- | --- | --- |
| `task` | 仅可委派 experiment-analyst | 无 |
| `analyze_experiment` | **不注册** | 唯一可用 Tool，只读 data/ CSV |
| `write_file` | 内置 Tool，仅新建 reports/*.md | **不注册** |
| `read_file` | 框架要求注册，但 Backend 一律拒绝 | 不注册 |
| 其他文件工具 / execute | 不注册 | 不注册 |

`Tool` 负责确定性计算；两角色的 `LLM` 分别负责编排与解释；官方 `SubAgentMiddleware` 处理委派与上下文隔离，`FilesystemMiddleware` 提供 Main 的内置写工具；原有 Backend 维持路径限制和禁止覆盖；CLI 验收实际执行证据和文件。

`agent.py:build_agent()` 的最终组装（省略工具描述文本）：

```python
run_evidence = RunEvidence()  # run_agent 每次运行创建，并绑定期望 CSV / Function
analyst = make_analyst(workspace, analyst_model, run_evidence, max_steps=24)
backend = ReportFilesystemBackend(workspace)  # virtual_mode=True
agent = create_deep_agent(
    model=main_model,
    tools=[],  # Main 不注册 analyze_experiment
    backend=backend,
    system_prompt=SYSTEM_PROMPT,
    middleware=[
        SubAgentMiddleware(backend=backend, subagents=[analyst]),
        FilesystemMiddleware(backend=backend, tools=["read_file", "write_file"]),
    ],
    name="mini-research-engineer",
)
```

检查安装源码后采用的 **0.7.40 兼容方式**：

- 普通声明式 `SubAgent` 会获得文件中间件，因此使用官方 `CompiledSubAgent`，包装仅注册分析 Tool 的 `create_agent()` 子图。子图没有 Backend、文件工具或 `task`，不能绕过 Main 写报告；没有 fork Main 的历史。
- 工厂默认添加 `general-purpose`。通过官方的同名 Middleware 替换机制，用只含 `experiment-analyst` 的 `SubAgentMiddleware` 替换默认实例。工厂临时构造的默认栈不会进入最终 Main 的工具集合；最终仅一个 `task`，仅一个可用子 Agent。测试同时检查工具描述和实际拒绝 general-purpose 调用。无需修改第三方源码或进程级全局 Profile 注册。
- 内置 `task` 支持 CompiledSubAgent 的 `structured_response`：返回对象被序列化为一个 task ToolMessage，而非复制子 Agent 全部消息。当前源码还通过 `config.metadata.lc_subagent_tool_call_id` 传递真实父调用 ID，记录器据此绑定证据。
- `FilesystemBackend.write()` 默认覆盖文件，因此继续使用原来的 `open('x')` 排他新建。`FilesystemMiddleware` 要求保留 `read_file`，Backend 仍拒绝所有通用读取。写入安全策略没有放宽。

未新增 Skills、Memory、Planning、HITL、MCP 或自定义 Summarization。`virtual_mode` 是路径限制，不是操作系统沙箱；这是本地可信演示。

接口参考：[create_deep_agent](https://reference.langchain.com/python/deepagents/graph/create_deep_agent)、[SubAgentMiddleware](https://reference.langchain.com/python/deepagents/middleware/subagents/SubAgentMiddleware)。具体行为以 `uv.lock` 和安装源码为准。

## 子 Agent 输出与两层验证

子模型最终返回普通 JSON 文本，不要求第三方服务支持 JSON mode 或额外结构化输出工具：

```json
{
  "evidence_markdown": "完整复制 analyze_experiment 的证据片段",
  "conclusions": {
    "final_performance": "中文解释",
    "stability": "中文解释",
    "convergence": "中文解释"
  },
  "significance_tested": false
}
```

兼容层不会信任“已调用工具”的自述。`make_analyze_experiment(..., on_success=...)` 在原有统计成功之后调用 Python 回调，记录实际 CSV、Function 和结果深拷贝；模型既没有回调参数，也没有记录器对象。每次 task 调用有自己的局部记录列表，`run_agent()` 每次创建新的 `RunEvidence`，没有全局可变记录。

第一层（`subagent.py`）：要求真实成功工具记录、实际 CSV/Function 符合运行目标、模型返回完整匹配的证据片段、三类非空结论、`significance_tested=false`。只有摘要、伪造调用、错误参数、丢失或修改证据均失败。数值有效性由原有 CSV 校验和统计函数保证，JSON 另要求不含 NaN/Infinity。

核验成功后，程序从**真实工具记录**补入完整精度原始统计，不要求模型重新抄写浮点数。Main 收到的 task JSON 为：

```text
{
  analysis: {完整 analyze_experiment 结果，Schema 见下文},
  conclusions: {final_performance: string, stability: string, convergence: string},
  significance_tested: false
}
```

`analysis` 保留函数、优化方向、路径/SHA256、每个 seed 最终值、每算法最终统计、平均曲线和完整 `evidence_markdown`。这不是用自然语言摘要替代数据：子模型必须保留证据，完整数字由可信工具记录传递。

第二层（`main.py:verify_run()`）：核对 Main 的真实 `task` 目标及调用 ID；`verify_analysis()` 将 task JSON 与该 ID 绑定的记录逐项比较；只有取得可信 task 结果之后的 `write_file` 才有效。再验证磁盘文件、实际写入内容、完整数值证据、模拟数据声明和最终路径。同一批同时调用 task 与 write_file 不能冒充先分析后写入。Main 的消息不包含子图的内部 ToolMessage。

Main 与每次子图运行分别限步；第一次分析验证失败会终止本次 CLI 验收。自然语言结论仍可能出错：例如“每个采样点较低”不能证明“差距持续扩大”。本阶段不增加语义判定器或新的分析公式，仍需人工审阅解释。

## 核心函数与 Tool Schema

- `analysis.py:analyze_dataframe()`：可独立测试的确定性分析，不修改输入 DataFrame。
- `analysis.py:analyze_csv()`：读取同一份 CSV 字节并计算 SHA256，再调用分析函数。
- `analysis.py:evidence_markdown()`：只格式化数值证据表，既不解释结论，也不生成完整报告。
- `tools/experiment.py:make_analyze_experiment(root)`：以闭包绑定工作区，避免依赖 cwd。
- `agent.py:load_model()` / `build_agent()`：加载配置、创建模型、Backend 和真实 Agent。
- `subagent.py:make_analyst()` / `verify_analysis()` / `RunEvidence`：隔离分析子图、核对输出、保存运行级可信证据。
- `main.py:run_agent()` / `verify_run()`：执行图并验收委派、写入及真实文件。

Tool 输入为 JSON 对象，两个字段都必填：

```json
{
  "csv_path": "data/sample_experiments.csv",
  "function_name": "F1"
}
```

`csv_path: string` 是项目相对路径，也可传入解析后位于本项目 `data/` 内的主机绝对路径。拒绝 `..`、`~`、项目外路径、越界符号链接、符号链接 data 目录、非 CSV 和不存在的文件。它不是 Backend 的虚拟路径，不能把 `/data/...` 当成主机上的项目目录。

成功输出为可 `json.dumps(..., allow_nan=False)` 的字典：

```text
{
  function: string,
  objective: "minimize",
  fe_grid: integer[],
  monotonic_atol: number,
  algorithms: {
    <算法名>: {
      n_seeds: integer,
      final_by_seed: [{seed: integer, fe: integer, fitness: number}],
      mean: number, sample_std: number, median: number,
      best: number, worst: number
    }
  },
  mean_convergence: {<算法名>: [{fe: integer, mean_fitness: number}]},
  comparison: {
    ranking_by_final_mean: string[],
    lowest_mean_algorithms: string[],
    significance_tested: false
  },
  source: {csv_path: string, sha256: string, data_kind: "simulated_demo", note: string},
  evidence_markdown: string
}
```

所有统计基于每个种子**最后一个 FE 的 fitness**。标准差使用 `ddof=1`；`Best=min`、`Worst=max`；平均收敛曲线按相同 FE 跨 seed 平均。均值并列时在 `lowest_mean_algorithms` 中保留所有并列算法；排名列表以名称打破排序歧义，不代表并列算法有优劣。未做 Wilcoxon/Friedman 等检验。

校验失败时抛出 `ToolException`，LangChain 将其转为错误 ToolMessage；不会把错误字符串当统计结果。统计 JSON 保留 Python 浮点精度；`evidence_markdown` 使用 12 位有效数字，包含数据路径、SHA256、最终统计表和完整平均收敛表。LLM 原样嵌入该证据片段，并独立生成解释段落。

## 离线测试

```bash
uv sync
uv run pytest -q
```

整个 pytest 套件禁止 HTTP 和 socket 连接，即使环境里有真实 API Key 也不能调用付费 API。测试分为：

- 确定性分析、`ddof=1`、最终 FE 提取、收敛聚合、输入错误与路径限制。
- 用脚本化 `BaseChatModel` 驱动**真实 create_deep_agent 图与工具节点**；验证工具注册、真实 CSV 分析、真实内置 `write_file` 写临时文件、异常不误报成功、拒绝覆盖、读取限制和步数上限。
- `ChatOpenAI` 配置测试只构造对象、核对显式参数，不调用模型。

这比单纯 patch 工厂函数的结构检查更完整，但依然是 **Fake Model 测试**，不能证明远程服务会产生正确工具调用或可靠科研解释。

## 如何证明 Agent 确实调用了 Tool

使用 `--trace` 后应观察到匹配 ID 的事件：

```text
tool_call task id=...
task_target experiment-analyst
tool_result task id=... status=success
subagent_analysis_verified subagent=experiment-analyst task_id=...
analysis_evidence { ...来自子图真实工具执行记录的统计... }
tool_call write_file id=...
tool_path {"file_path": "/reports/F1_comparison.md"}
tool_result write_file id=... status=success
```

CLI 读取的是真实 Main AIMessage/ToolMessage。子 Agent 的内部分析 ToolMessage **不在 Main 的历史中**，`subagent_analysis_verified` 来自绑定 task ID 的程序记录，而非模型自述。证据比较仅忽略行尾空白和换行符差异，允许 Markdown 双空格换行；数值、哈希、行内内容仍须匹配。仅模型声称“已保存”不算成功。

文件工具失败时，`--trace` 还会显示 `tool_error <name>: <具体原因>`；最终验收错误保留最后一次 `write_file` 错误，避免只显示泛化的“报告未保存”。不会输出写入报告的完整 content 参数或原始 API 错误体。

如果看到 `read_file status=error`，这是 Backend 禁止通用读取的结果；创建新报告无需先读取目标。若 `write_file` 提示“虚拟路径”，检查其 `tool_path`：工具参数必须为 `/reports/F1_comparison.md`，不能传 `/Users/.../reports/F1_comparison.md`。主机绝对路径只用于最终回答。System Prompt、用户请求和工具描述都明确区分这两种路径；Backend 继续拒绝非法路径，不自动纠正或放宽限制。

如果服务拒绝 `tools` 或 `tool_calls` 格式，会报告 HTTP 状态和兼容性诊断；连接失败、超时、未实现 `bind_tools` 或一直没有工具调用也会失败。不会改用普通 Python 直接生成报告来伪装 Agent 成功。为了不泄露服务可能回显的密钥，不打印原始 API 错误体。

## 手动远程集成测试（需要真实 API，可能付费）

配置 `.env` 后明确执行：

```bash
uv run python scripts/manual_integration.py
```

这不是 pytest 项目，也不会自动运行。脚本进行两次真实 Agent 运行：

1. 正常案例：Main 实际调用 task 委派 experiment-analyst；子模型调用分析 Tool，并返回通过运行记录核验的完整结果；Main 调用内置 write_file 保存唯一新报告。检查 task 目标、可信记录、独立重算结果、报告数值和真实路径。Main 不直接调用分析 Tool。
2. 写入失败案例：在临时工作区用名为 `reports` 的普通文件阻止创建报告目录。要求观察到真实 `write_file` 错误 ToolMessage、无报告文件、验收函数拒绝返回成功。这个案例也调用真实模型，不使用 Fake Model。

任一条件未满足则脚本失败；缺少 API 配置也不会记为通过。脚本成功之后，**人工审阅报告文字**是否基于数值解释最终性能、标准差和曲线，是否错误声称统计显著性。程序验证数值核心，不能证明每一句自由文本语义都正确。只读独立复算：

```bash
uv run python - <<'PY'
import json
from research_agent.tools.experiment import analyze_experiment
r = analyze_experiment.invoke({"csv_path": "data/sample_experiments.csv", "function_name": "F1"})
print(json.dumps(r, ensure_ascii=False, indent=2, allow_nan=False))
PY
```

## 实际验证记录

2026-11-20（Asia/Shanghai）：

| 验证 | 实际结果 |
| --- | --- |
| Phase 2.1 修改前基线 | **57 passed**，含先前的路径诊断和 Markdown 行尾空白修复 |
| Phase 2.1 全套离线测试 | **71 passed**；原 57 项回归保留/适配委派角色，新增 14 个参数化测试用例 |
| 统计公式和 CSV 数据 | `analysis.py` 与 sample CSV 无改动；依赖锁定仍为 DeepAgents 0.7.40 |
| 新测试覆盖 | 工具隔离、上下文隔离、完整 task JSON、证据丢失/篡改、无工具执行、错误参数、子图越权写入、记录隔离、写入失败及限步 |
| Phase 1 用户本地真实运行 | 用户提供的 trace 已观察到端到端分析、写入和 CLI 验收成功；不等于解释语义完全正确 |
| Phase 2.1 真实远程模型 | **本次未执行、未验证**；需显式运行手动脚本，不能沿用 Phase 1 成功记录 |

整个 pytest 套件禁止 HTTP 和 socket。真实 API 测试不被 pytest 收集，也不会因缺密钥而记为通过。本机使用可写缓存：`UV_CACHE_DIR=/private/tmp/research-agent-uv-cache`；正常开发环境可直接用 `uv sync`、`uv run pytest -q`。

## Phase 2.1 范围与限制

仅增加一个负责实验分析的子 Agent，验证委派后的工具权限、数据证据和上下文隔离。不增加统计公式、数据库、前端、容器、MCP、Skills、Memory、Planning、HITL、昂贵实验或自动改算法。

模型须支持普通 tools/tool_calls，并能返回包含完整证据的 JSON 文本；格式不合约会明确失败，不回退成伪成功。解释仍可能出错，完整证据不等于语义正确。记录器是进程内、单次运行的验证辅助，不是持久审计系统；版本依赖点是 CompiledSubAgent structured_response、task metadata ID 和同名 Middleware 替换，升级需重跑回归。

## 文件结构

```text
research-engineer-agent/
├── pyproject.toml
├── uv.lock
├── .python-version
├── .env.example
├── .gitignore
├── README.md
├── src/research_agent/
│   ├── __init__.py
│   ├── agent.py
│   ├── main.py
│   ├── prompts.py
│   ├── analysis.py
│   ├── subagent.py
│   └── tools/
│       ├── __init__.py
│       └── experiment.py
├── scripts/
│   ├── generate_sample.py
│   └── manual_integration.py
├── data/sample_experiments.csv
├── reports/.gitkeep
└── tests/
    ├── conftest.py
    ├── test_analysis.py
    ├── test_experiment_tool.py
    └── test_agent_setup.py
```

GitHub 仓库：[077cheerup/research-engineer-agent](https://github.com/077cheerup/research-engineer-agent)。按完成并验证的阶段提交和推送，密钥与运行报告不上传。
