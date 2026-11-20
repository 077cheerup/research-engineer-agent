# Mini Research Engineer Agent — Phase 2.2

基于 **DeepAgents 0.7.40** 的增量科研分析闭环：Main 用内置 `task` 委派 `experiment-analyst`，子 Agent 通过 `SkillsMiddleware` 发现专业规范、按需读取白名单 Skill，再调用确定性分析 Tool；验证真实 Skill 加载和工具执行后返回完整统计与解释，Main 用内置 `write_file` 保存报告。Main 仍由 `create_deep_agent()` 创建，子图通过官方 `CompiledSubAgent` 接口接入。

本项目是 DeepAgents / Agent Harness 学习路线 Level 7 Phase 2.2。在 `1af7f5c` 的 Phase 2.1 架构上继续增量开发，保留当前 HEAD 的可信证据直接传输与分章节核验修复；不回退代码，统计公式和 CSV 格式不变。这里研究的内容只有算法 A/B 在 F1 上的模拟结果；不运行真实 CEC2013 函数，不代表正式 CEC2013 实验，不做显著性检验。

## 安装、配置与运行

需要 Python 3.11+ 和 uv。依赖和具体版本固定在 `uv.lock`；`.python-version` 指定 3.11。

```bash
uv sync
cp .env.example .env
# 编辑 .env，填入自己服务的三个配置
uv run python -m research_agent.main --output reports/F1_phase22.md --trace
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
                -> SkillsMiddleware 发现名称/描述/路径
                -> read_file：只读 experiment-analysis/SKILL.md
                -> analyze_experiment -> 原有确定性统计
                -> JSON 来源引用 + 中文结论
             -> 程序核对真实工具记录、证据及参数
             -> task ToolMessage：完整可信 analysis + conclusions + skill 加载凭据
          -> 内置 write_file -> ReportFilesystemBackend -> reports/*.md
CLI -> 两层验证 -> 摘要和真实路径
```

| 权限 | Main Agent | experiment-analyst |
| --- | --- | --- |
| `task` | 仅可委派 experiment-analyst | 无 |
| `analyze_experiment` | **不注册** | 保留，只读 data/ CSV |
| `write_file` | 内置 Tool，仅新建 reports/*.md | **不注册** |
| `read_file` | 框架要求注册，但 Backend 一律拒绝 | **新增**，只读指定 Skill 的虚拟路径 |
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

- 普通声明式 `SubAgent` 会获得文件中间件，因此使用官方 `CompiledSubAgent`，包装独立的 `create_agent()` 子图。Phase 2.2 子图只增加 Skill 专用只读 Backend 与 `read_file`，不继承 Main 的报告 Backend 或其他文件工具，不能绕过 Main 写报告；没有 fork Main 的历史。
- 工厂默认添加 `general-purpose`。通过官方的同名 Middleware 替换机制，用只含 `experiment-analyst` 的 `SubAgentMiddleware` 替换默认实例。工厂临时构造的默认栈不会进入最终 Main 的工具集合；最终仅一个 `task`，仅一个可用子 Agent。测试同时检查工具描述和实际拒绝 general-purpose 调用。无需修改第三方源码或进程级全局 Profile 注册。
- 内置 `task` 支持 CompiledSubAgent 的 `structured_response`：返回对象被序列化为一个 task ToolMessage，而非复制子 Agent 全部消息。当前源码还通过 `config.metadata.lc_subagent_tool_call_id` 传递真实父调用 ID，记录器据此绑定证据。
- `FilesystemBackend.write()` 默认覆盖文件，因此继续使用原来的 `open('x')` 排他新建。`FilesystemMiddleware` 要求保留 `read_file`，Backend 仍拒绝所有通用读取。写入安全策略没有放宽。

本阶段只增加一个专业 Skill；未新增子 Agent、Memory、Planning、HITL、MCP 或自定义 Summarization。`virtual_mode` 是路径限制，不是操作系统沙箱；这是本地可信演示。

接口参考：[create_deep_agent](https://reference.langchain.com/python/deepagents/graph/create_deep_agent)、[SubAgentMiddleware](https://reference.langchain.com/python/deepagents/middleware/subagents/SubAgentMiddleware)。具体行为以 `uv.lock` 和安装源码为准。

## experiment-analysis Skill 与渐进加载

```text
skills/
└── experiment-analysis/
    └── SKILL.md
```

`SKILL.md` 有有效的 YAML `name`、`description`，正文涵盖适用任务、步骤、Mean/Median/Std 解释、平均收敛趋势、显著性边界、模拟数据声明及错误/推荐表述。它不计算统计，不运行 Benchmark。专业规范只在 Skill 文件中维护，未整篇复制到 System Prompt；Prompt 仅规定角色、工具调用和输出协议。

配置位于 `subagent.py:make_analyst()`，只用于 Analysis Subagent：

```python
skill_backend = SkillReadBackend(root)
child = create_agent(
    model=model,
    tools=[make_analyze_experiment(root, on_success=record)],
    system_prompt=ANALYST_PROMPT,
    middleware=[
        FilesystemMiddleware(backend=skill_backend, tools=["read_file"]),
        SkillsMiddleware(backend=skill_backend, sources=["/skills/"]),
    ],
    name="experiment-analyst",
)
```

已检查锁定版本 **DeepAgents 0.7.40 / LangChain 1.5.2 / langchain-core 1.8.3 / langchain-openai 1.8.3** 的安装源码：

1. **发现**：`SkillsMiddleware.before_agent()` 用 Backend 的 `ls()` 与 `download_files()` 读取并解析 frontmatter，然后把名称、描述、加载路径加入系统消息。虽然发现过程在程序内部读取文件字节，模型此时只看到元信息，**没有 Skill 正文**。
2. **加载**：模型调用内置 `read_file(file_path="/skills/experiment-analysis/SKILL.md", limit=1000)`，实际成功的 ToolMessage 才把正文送入子模型上下文。不是创建 Agent 时拼接文件，也没有 pinned Skill 预加载。
3. **应用**：子模型基于加载的规范解释统计结果；Main 使用结论和可信数值证据写报告。Skill 不计算数字。

`SkillReadBackend` 是最小 `BackendProtocol` 实现：只枚举上述目录，只能下载或读取这一个虚拟路径；路径穿越、主机绝对路径、符号链接、`.env`、CSV、报告和其他 Skill 都被拒绝。写入一律拒绝，其他文件工具根本不注册。原有 Main Backend 与 CSV 安全解析未放宽。

`make_analyst()` 检查中间件真实发现的名称和路径，并同时要求：

- 对应 `read_file` 的真实 AI Tool Call 与成功 ToolMessage；
- Backend 的全文读取凭据（记录路径和文件 SHA256）。发现用的 `download_files()` 或只读一行不会生成全文凭据。

子模型声称 `skill_loaded=true` 不产生凭据。未加载、无效或缺失 Skill 会在发布可信 task 结果前失败。加载凭据与原有统计记录一样绑定到本次真实 task ID；不使用全局可变记录。

本版本的 Skills 元数据在普通 `invoke()` 输出中被隐藏，因此 wrapper 用官方 `stream_mode="values"` 观察内部发现状态；只将验证后的结构化 task 结果返回 Main，不传播子图内部消息或 Skill 全文。

## 子 Agent 输出与两层验证

子模型最终返回普通 JSON 文本，不要求第三方服务支持 JSON mode 或额外结构化输出工具：

```json
{
  "source_sha256": "工具结果 source.sha256 的完整字符串",
  "conclusions": {
    "final_performance": "中文解释",
    "stability": "中文解释",
    "convergence": "中文解释"
  },
  "significance_tested": false
}
```

兼容层不会信任“已调用工具”的自述。`make_analyze_experiment(..., on_success=...)` 在原有统计成功之后调用 Python 回调，记录实际 CSV、Function 和结果深拷贝；模型既没有回调参数，也没有记录器对象。每次 task 调用有自己的局部记录列表，`run_agent()` 每次创建新的 `RunEvidence`，没有全局可变记录。

第一层（`subagent.py`）：要求真实成功工具记录、实际 CSV/Function 符合运行目标、子模型返回的 `source_sha256` 与该工具结果完全一致、三类非空结论、`significance_tested=false`。只有摘要、未执行工具、错误参数、引用缺失或错误均失败。数值有效性由原有 CSV 校验和统计函数保证，JSON 另要求不含 NaN/Infinity。

核验成功后，程序从**真实工具记录**直接附入完整统计与 `evidence_markdown`，不要求子模型重新抄写浮点数或 Markdown 表格。先前的复制契约会因模型漏行或重排而失败；现在删除这一重复搬运步骤，来源校验、实际执行证明与 task 完整性检查仍保留。Main 收到的 task JSON 为：

```text
{
  analysis: {完整 analyze_experiment 结果，Schema 见下文},
  conclusions: {final_performance: string, stability: string, convergence: string},
  significance_tested: false,
  skill: {name: "experiment-analysis", path: "/skills/experiment-analysis/SKILL.md", sha256: string}
}
```

`analysis` 保留函数、优化方向、路径/SHA256、每个 seed 最终值、每算法最终统计、平均曲线和完整 `evidence_markdown`。这不是用自然语言摘要替代数据：`task` 结果必须保留完整可信证据，程序将实际统计与子模型结论合并后返回；`verify_analysis()` 会拒绝丢失或篡改任一数据字段的 task 结果。

第二层（`main.py:verify_run()`）：核对 Main 的真实 `task` 目标及调用 ID；`verify_analysis()` 将 task JSON 与该 ID 绑定的记录逐项比较；只有取得可信 task 结果之后的 `write_file` 才有效。再验证磁盘文件、实际写入内容、完整数值证据、模拟数据声明和最终路径。同一批同时调用 task 与 write_file 不能冒充先分析后写入。Main 的消息不包含子图的内部 ToolMessage。

报告中的证据由 `verify_report_evidence()` 分别核对三个完整块：来源信息、最终统计表、平均收敛表。允许把它们放在不同章节、在块之间添加解释，不要求整段连续出现；每块内的全部文本和数值仍须匹配，仅忽略行尾空白。缺块或改数值会指出具体证据块，不能只靠零散数字通过验收。

Main 与每次子图运行分别限步；第一次分析验证失败会终止本次 CLI 验收。自然语言结论仍可能出错：例如“每个采样点较低”不能证明“差距持续扩大”。本次只增加下述有限语言检查，不增加通用语义判定器或新的统计公式，仍需人工审阅解释。

## 统计语言核验（Phase 2.2 修复）

真实运行已证明 Skill 成功读取不等于输出语义正确。`reports/F1_phase22.md` 中出现了不受当前模拟样本支持的“全程领先”等概括；用户提供的 Main 最终回答还有“显著更优”与“未进行检验”的矛盾。原验证只检查统计证据和执行链，没有审查这些文字。历史 task conclusions 未保存，原 trace 也不包含它们，因此不能断言措辞最初来自子模型还是 Main 的改写。

本次只增加 `statistical_language.py:verify_statistical_language()`，复用于三个输出位置：

- 子 Agent 的三个 `conclusions` 字段，在发布可信 task 结果前检查；
- `verify_run()` 检查实际保存的 Markdown 正文；
- `verify_run()` 检查实际 Main 最终回答，全部检查通过之后 CLI 才打印“已核验报告”。

显著性状态取自可信 Python 工具的 `analysis.comparison.significance_tested`，不以模型自己的声明为准。未检验时拒绝常见的“显著更优/低于/更稳定”“存在统计学显著差异”等正向断言。另检查采样点被强化为“全程/每个阶段领先”、未经计算的速度优势，以及未限定样本的种子/稳定性概括或真实 CEC2013 优势。

按标点与转折划分局部子句，允许“不能据此宣称显著更优”“较小 Std 不等于显著更稳定”等否定；“未做检验，但 B 显著更优”中的后半句仍会失败。不是简单禁止“显著”一词，也不把一个段落开头的免责声明当作后文所有断言的豁免。推荐使用“当前模拟样本中均值较低”“样本标准差较小”“已记录 FE 采样点的均值较低”等具体表述。

Main Prompt 仅补充报告与最终回答必须保持同一描述性边界。现有 Analyst Prompt、Skill 内容、渐进加载、Backend 和统计公式不改。trace 新增 `analysis_conclusions`，显示程序已经接收的子结论，便于下一次对照 Main 的报告与最终回答，不新增持久日志系统。

**范围与局限**：这是针对当前项目常见中文措辞的有限规则，不是完整语法或统计推理。复杂引用、跨句限定、双重否定、同义改写可能漏检或误报；它不验证任意派生数字，也不能证明每句结论正确。检查在验收阶段运行；文件若已写入，会作为未验收产物保留，不自动删除或覆盖。原有精确数值证据核验继续执行。

## 核心函数与 Tool Schema

- `analysis.py:analyze_dataframe()`：可独立测试的确定性分析，不修改输入 DataFrame。
- `analysis.py:analyze_csv()`：读取同一份 CSV 字节并计算 SHA256，再调用分析函数。
- `analysis.py:evidence_markdown()`：只格式化数值证据表，既不解释结论，也不生成完整报告。
- `tools/experiment.py:make_analyze_experiment(root)`：以闭包绑定工作区，避免依赖 cwd。
- `agent.py:load_model()` / `build_agent()`：加载配置、创建模型、Backend 和真实 Agent。
- `subagent.py:make_analyst()` / `verify_analysis()` / `RunEvidence`：隔离分析子图、核对输出、保存运行级可信证据。
- `skill_backend.py:SkillReadBackend`：Skill 发现/加载的单文件白名单和全文读取凭据。
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

校验失败时抛出 `ToolException`，LangChain 将其转为错误 ToolMessage；不会把错误字符串当统计结果。统计 JSON 保留 Python 浮点精度；`evidence_markdown` 使用 12 位有效数字，包含数据路径、SHA256、最终统计表和完整平均收敛表。Main 可将三个完整证据块放在不同章节，并独立生成解释段落。

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
skill_load_verified {"name": "experiment-analysis", "path": "/skills/experiment-analysis/SKILL.md", "sha256": "..."}
analysis_conclusions { ...已接收的子 Agent 结论... }
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

1. 正常案例：Main 实际调用 task 委派 experiment-analyst；子模型真实读取白名单 Skill、调用分析 Tool，并返回通过运行记录核验的完整结果；Main 调用内置 write_file 保存唯一新报告。检查 task 目标、真实全文读取凭据及 Skill SHA256、分析记录、独立重算结果、报告数值和真实路径。Main 不直接调用分析 Tool。
2. 写入失败案例：临时工作区包含同一 Skill 和样例 CSV，用名为 `reports` 的普通文件阻止创建报告目录。要求观察到真实 `write_file` 错误 ToolMessage、无报告文件、验收函数拒绝返回成功。这个案例也调用真实模型，不使用 Fake Model。

任一条件未满足则脚本失败；缺少 API 配置也不会记为通过。脚本成功之后，**人工审阅报告文字**是否基于数值解释最终性能、标准差和曲线，是否错误声称均值或稳定性差异显著、是否从单点均值推断速度，以及是否编造派生数字。程序验证真实加载与数值核心，不能证明每一句自由文本语义都正确。只读独立复算：

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
| 本阶段修改前基线 | **75 项离线测试**；已保留 Phase 2.1 后续证据传输和报告核验修复 |
| Phase 2.2 全部离线测试 | **99 passed**；75 项回归通过，新增 24 个 Skill 测试用例 |
| Skill frontmatter / 命名校验 | `skill-creator/scripts/quick_validate.py` 返回 `Skill is valid!` |
| 发现与加载 | 实际 SkillsMiddleware、read_file、白名单 Backend；发现时无正文，加载时有全文及可信凭据 |
| 权限与失败 | 非白名单拒绝、缺失/无效 Skill、仅读一行、跳过读取、伪造已加载声明均覆盖 |
| 使用阶段 | Fake Model 从真实读取的 Skill 表格提取推荐表述，覆盖均值显著性、Std 显著性和单点速度误推；保持模拟声明、不新增百分比等统计 |
| Phase 2.1 真实模型 | 之前正常链路已通过技术验收；不等于本阶段 Skill 的远程效果已验证 |
| Phase 2.2 用户真实模型运行 | 用户报告 task、Skill 加载、write_file 和最终验收均通过，但 Main 文字仍有不严谨断言；本次针对这一已发生问题修复 |
| 统计语言修复后 | **130 项离线测试通过**；保留原 99 项，新增 31 项规则与三输出位置/CLI 回归用例 |
| 修复后真实模型 | **尚未重跑**；使用新报告名或显式手动入口验证，不能将 Fake Model 通过视为远程通过 |

整个 pytest 套件禁止 HTTP 和 socket。Fake Model 测试验证调用协议、信息暴露和证据传递，**不能证明真实模型理解或遵守专业规范**。真实 API 测试不被 pytest 收集，不会因缺密钥记为通过。本机使用可写缓存 `UV_CACHE_DIR=/private/tmp/research-agent-uv-cache`；普通环境可直接 `uv run pytest -q`。

## Phase 2.2 范围与限制

仅增加一个专业分析 Skill 与受限读取，未增加子 Agent、统计公式、数据库、前端、MCP、Memory、Planning、HITL 或自动实验。当前 CLI 只处理实验分析，因此这一类任务必须实际加载适用 Skill 才验收。

专业规范没有复制进 System Prompt；仅在输出边界增加上述有限语言规则，没有通用语义判定器。模型仍可能通过未覆盖的改写产生错误推断，Main 也可能在整理时改变限定范围；真实模型效果与全文语义仍需手动审阅。记录器是进程内单次运行的验证辅助，不是持久审计系统。

版本依赖点包括 SkillsMiddleware 的 ls/download_files 发现接口、内部 Skill 元数据状态、FilesystemMiddleware 的 read_file，以及原有 task metadata ID、structured_response 和同名 Middleware 替换。依赖未升级；升级时应重跑回归。源码工作区运行仍是支持的使用方式，wheel 不单独打包 skills/data 资源。

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
│   ├── skill_backend.py
│   ├── statistical_language.py
│   └── tools/
│       ├── __init__.py
│       └── experiment.py
├── scripts/
│   ├── generate_sample.py
│   └── manual_integration.py
├── skills/experiment-analysis/SKILL.md
├── data/sample_experiments.csv
├── reports/.gitkeep
└── tests/
    ├── conftest.py
    ├── test_analysis.py
    ├── test_experiment_tool.py
    ├── test_agent_setup.py
    ├── test_skills.py
    └── test_statistical_language.py
```

GitHub 仓库：[077cheerup/research-engineer-agent](https://github.com/077cheerup/research-engineer-agent)。按完成并验证的阶段提交和推送，密钥与运行报告不上传。
