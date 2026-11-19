# Mini Research Engineer Agent — Phase 1

基于 **DeepAgents 0.7.40** 的最小科研分析闭环：确定性 Python Tool 计算模拟实验统计，LLM 基于结果解释，DeepAgents 内置 `write_file` 保存 Markdown，CLI 验证真实工具调用和磁盘文件。使用 `create_deep_agent()`，没有用普通函数冒充 Agent。

本项目是 DeepAgents / Agent Harness 学习路线 Level 7 Phase 1。这里研究的内容只有算法 A/B 在 F1 上的模拟结果；不运行真实 CEC2013 函数，不代表正式 CEC2013 实验，不做显著性检验。

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

CLI 默认 `--max-steps 24`（LangGraph 图步数，不是 token 数或精确模型调用次数），防止无限 Tool Calling。缺失配置、服务错误、漏调工具、文件不存在、数值证据不一致时退出码为 1；只有验收通过才输出“已核验报告”。如果失败前已经生成文件，该文件是**未通过验收的产物**，不会被自动删除或标为成功；再次运行需换文件名。

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

`data/sample_experiments.csv` **全部是模拟数据**，不是 CEC2013 的真实函数值、误差或官方实验。Phase 1 同样把用户输入视作模拟演示数据，不适用于正式 benchmark 结果的直接发表。

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
CLI -> ChatOpenAI / create_deep_agent (真实 LangGraph 图)
        -> analyze_experiment(csv_path, function_name)
           -> 路径限制 -> 读取一次 CSV / SHA256 -> 校验 / 统计 -> JSON
        -> LLM 解释结构化结果并组装报告
        -> DeepAgents 内置 write_file(file_path, content)
           -> ReportFilesystemBackend -> reports/<name>.md
CLI <- ToolMessages + 最终回答 -> 验证调用顺序、数值证据、文件与路径
```

| 组件 | 职责 |
| --- | --- |
| Tool | `@tool analyze_experiment` 只读取 `data/` 的 CSV，调用确定性分析；不调用 LLM，不生成完整报告 |
| LLM | 选择工具、解释最终性能/稳定性/收敛趋势、撰写报告；不能自行编造统计数字 |
| Middleware | 使用 DeepAgents 官方 `FilesystemMiddleware` 暴露内置文件工具；默认 harness 处理图的工具循环等能力 |
| Backend | `FilesystemBackend` 子类，根目录为项目，显式 `virtual_mode=True`；新建报告、禁止覆盖和通用读取 |
| CLI | 限制图步数，检查真实 ToolMessage、成功写入、磁盘内容、数字证据和最终路径 |

具体组装在 `agent.py:build_agent()`：

```python
backend = ReportFilesystemBackend(workspace)  # super(..., virtual_mode=True)
agent = create_deep_agent(
    model=model,  # 实际为显式配置的 ChatOpenAI；离线测试注入 Fake Model
    tools=[make_analyze_experiment(workspace)],
    backend=backend,
    system_prompt=SYSTEM_PROMPT,
    middleware=[FilesystemMiddleware(
        backend=backend, tools=["read_file", "write_file"],
    )],
    name="mini-research-engineer",
)
```

两个经过实际测试确认的 **0.7.40 接口细节**：

1. 原始 `FilesystemBackend.write()` 会覆盖已有文件。因此小型子类只在 `reports/*.md` 下用 `open('x')` 排他新建，不调用原来的截断写入；仍由原生 `write_file` 工具实际调用 Backend，未修改第三方源码。
2. `FilesystemMiddleware` 工具列表必须包含 `read_file`；`/**` 权限模式不匹配 `.env`。因此保留所需的 `read_file` 注册，但 Backend 的 `read()` 一律返回拒绝，CSV 读取只由分析 Tool 负责。测试直接检查模型无法通过该工具读取 `.env`。

没有额外实现 Subagent、Skills、Memory、Todo、HITL、自定义 Summarization 或 MCP。DeepAgents 自身可能仍默认注册 `task` 等工具/中间件；System Prompt 明确要求只使用分析和写文件两种工具，不依赖高级功能。`virtual_mode` 是路径语义与限制，不是操作系统沙箱；这是本地可信演示项目。

接口参考：[create_deep_agent](https://reference.langchain.com/python/deepagents/graph/create_deep_agent)、[FilesystemBackend](https://reference.langchain.com/python/deepagents/backends/filesystem/FilesystemBackend)、[FilesystemMiddleware](https://reference.langchain.com/python/deepagents/middleware/filesystem/FilesystemMiddleware)。具体行为以 `uv.lock` 锁定版本及本地运行测试为准。

## 核心函数与 Tool Schema

- `analysis.py:analyze_dataframe()`：可独立测试的确定性分析，不修改输入 DataFrame。
- `analysis.py:analyze_csv()`：读取同一份 CSV 字节并计算 SHA256，再调用分析函数。
- `analysis.py:evidence_markdown()`：只格式化数值证据表，既不解释结论，也不生成完整报告。
- `tools/experiment.py:make_analyze_experiment(root)`：以闭包绑定工作区，避免依赖 cwd。
- `agent.py:load_model()` / `build_agent()`：加载配置、创建模型、Backend 和真实 Agent。
- `main.py:run_agent()` / `verify_run()`：执行图并验收真实工具链。

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
tool_call analyze_experiment id=...
tool_result analyze_experiment id=... status=success
analysis_evidence { ...真实结构化统计... }
tool_call write_file id=...
tool_path {"file_path": "/reports/F1_comparison.md"}
tool_result write_file id=... status=success
```

CLI 读取的是真实 AIMessage/ToolMessage，并将 Tool Call 的参数与指定 CSV、函数和目标文件匹配。只有先收到成功分析结果，再调用成功的 `write_file` 才验收；同一批并发发起分析和写文件也不能冒充先分析后写报告。CLI 还检查文件内容等于写入参数、证据表完全包含、模拟数据声明存在、最终回答包含真实相对和绝对路径。仅模型文字声称“已保存”不算成功。

文件工具失败时，`--trace` 还会显示 `tool_error <name>: <具体原因>`；最终验收错误保留最后一次 `write_file` 错误，避免只显示泛化的“报告未保存”。不会输出写入报告的完整 content 参数或原始 API 错误体。

如果看到 `read_file status=error`，这是 Backend 禁止通用读取的结果；创建新报告无需先读取目标。若 `write_file` 提示“虚拟路径”，检查其 `tool_path`：工具参数必须为 `/reports/F1_comparison.md`，不能传 `/Users/.../reports/F1_comparison.md`。主机绝对路径只用于最终回答。System Prompt、用户请求和工具描述都明确区分这两种路径；Backend 继续拒绝非法路径，不自动纠正或放宽限制。

如果服务拒绝 `tools` 或 `tool_calls` 格式，会报告 HTTP 状态和兼容性诊断；连接失败、超时、未实现 `bind_tools` 或一直没有工具调用也会失败。不会改用普通 Python 直接生成报告来伪装 Agent 成功。为了不泄露服务可能回显的密钥，不打印原始 API 错误体。

## 手动远程集成测试（需要真实 API，可能付费）

配置 `.env` 后明确执行：

```bash
uv run python scripts/manual_integration.py
```

这不是 pytest 项目，也不会自动运行。脚本进行两次真实 Agent 运行：

1. 正常案例：真实模型生成分析 Tool Call、Python Tool 执行、LLM 使用结果、真实内置 `write_file` 保存唯一新报告。重算 Tool JSON 核对结果，检查报告证据、磁盘文件、最终路径。
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

2026-11-19（Asia/Shanghai）：

| 验证 | 实际结果 |
| --- | --- |
| `uv sync --python 3.11`，随后 `uv sync` | 均成功；CPython 3.11.16、DeepAgents 0.7.40、LangChain 1.5.2、langchain-openai 1.8.3 |
| PyPI 稳定版核对 | 官方 PyPI JSON 返回最新版本 0.7.40，与安装版本一致 |
| `uv run pytest -q` | **55 passed in 0.56s**，完全离线 |
| 模拟 CSV 重新生成 | SHA256 字节一致，可重复 |
| 移除 LLM 配置后的实际 CLI 运行 | 明确列出缺失配置，退出码 1，未调用远程模型 |
| 文件工具错误诊断修复后 `uv run pytest -q` | **56 passed**；新增真实图中误传主机绝对路径的回归测试 |
| 远程模型成功/写入失败集成 | **未执行、未验证**；仅提供显式手动脚本 |

本机默认 uv 缓存目录不在可写范围内，因此实际命令使用了 `UV_CACHE_DIR=/private/tmp/research-agent-uv-cache`；第一次沙箱内安装因 DNS 失败，授权网络访问后同步成功。正常开发环境无需这一缓存覆盖。不把环境限制或缺密钥的远程测试记为通过。

## Phase 1 范围与后续

Phase 1 只验证“读取 CSV → 确定性统计 → LLM 解释 → 内置工具保存 → 文件验收”这一条主假设。不提供数据库、前端、Docker、MCP、自定义多 Agent、自动改算法、昂贵实验、论文复现、大规模评测或显著性检验。

局限包括演示数据、单函数交互、同预算同采样点要求，以及自由文本解释仍需审阅。优先在 Phase 2 用真实模型和人工核验的小批报告验证解释可靠性，再根据证据决定是否增加能力；不预先堆叠模块。

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
