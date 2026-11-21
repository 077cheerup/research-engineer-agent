# Mini Research Engineer Agent — Phase 2.3

基于 **DeepAgents 0.7.40** 的增量科研分析闭环：Main 用内置 `task` 委派 `experiment-analyst`，子 Agent 通过 `SkillsMiddleware` 发现专业规范、按需读取白名单 Skill，再调用确定性分析 Tool；验证真实 Skill 加载和工具执行后返回完整统计与解释，Main 用内置 `write_file` 保存报告。Main 仍由 `create_deep_agent()` 创建，子图通过官方 `CompiledSubAgent` 接口接入。

本项目是 DeepAgents / Agent Harness 学习路线 Level 7 Phase 2.3。基于 Phase 2.2 及统计语言修复 `a2817cb` 增量增加本地 JSON Research Memory，保留可信证据直接传输、分章节核验、Skill 和语言边界；统计公式与 CSV 格式不变。这里研究的内容只有算法 A/B 的模拟结果；不运行真实 CEC2013 函数，不代表正式 CEC2013 实验，不做显著性检验。

## 安装、配置与运行

需要 Python 3.11+ 和 uv。依赖和具体版本固定在 `uv.lock`；`.python-version` 指定 3.11。

```bash
uv sync
cp .env.example .env
# 编辑 .env，填入自己服务的三个配置
uv run python -m research_agent.main --output reports/F1_phase23_first.md --trace
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
用户 -> Python 按函数/算法检索并验证 Memory -> 有长度限制的历史事实参考
      -> Main（create_deep_agent）
          -> task(subagent_type="experiment-analyst", description="CSV + Function")
             -> 隔离的 Analysis Subagent（create_agent）
                -> SkillsMiddleware 发现名称/描述/路径
                -> read_file：只读 experiment-analysis/SKILL.md
                -> analyze_experiment -> 原有确定性统计
                -> JSON 来源引用 + 中文结论
             -> 程序核对真实工具记录、证据及参数
             -> task ToolMessage：完整可信 analysis + conclusions + skill 加载凭据
          -> 内置 write_file -> ReportFilesystemBackend -> reports/*.md
CLI -> 两层验证及统计语言核验 -> 保存可信 Research Memory -> 摘要和真实路径
```

| 权限 | Main Agent | experiment-analyst |
| --- | --- | --- |
| `task` | 仅可委派 experiment-analyst | 无 |
| `analyze_experiment` | **不注册** | 保留，只读 data/ CSV |
| `write_file` | 内置 Tool，仅新建 reports/*.md | **不注册** |
| `read_file` | 框架要求注册，但 Backend 一律拒绝 | **新增**，只读指定 Skill 的虚拟路径 |
| Memory 读写工具 | 不注册；由 Python 编排检索与验收后保存 | 不注册 |
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

沿用 Phase 2.2 的专业 Skill；本阶段只增加程序管理的 JSON Memory，未新增子 Agent、MemoryMiddleware、Planning、HITL、MCP 或自定义 Summarization。`virtual_mode` 是路径限制，不是操作系统沙箱；这是本地可信演示。

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

显著性状态取自可信 Python 工具的 `analysis.comparison.significance_tested`，不以模型自己的声明为准。未检验时拒绝“显著更优/更稳定”“在统计意义上显著低于”“存在统计学显著差异”等正向统计断言。按 2026-11-21 的使用偏好，单纯“显著低于／显著更低”作为程度描述暂不拦截；它不表示已证明统计显著性。另检查采样点被强化为“全程/每个阶段领先”、未经计算的速度优势，以及未限定样本的种子/稳定性概括或真实 CEC2013 优势。

按标点与转折划分局部子句，允许“不能据此宣称显著更优”“较小 Std 不等于显著更稳定”等否定；“未做检验，但 B 显著更优”中的后半句仍会失败。不是简单禁止“显著”一词，也不把一个段落开头的免责声明当作后文所有断言的豁免。推荐使用“当前模拟样本中均值较低”“样本标准差较小”“已记录 FE 采样点的均值较低”等具体表述。

Main Prompt 要求报告与最终回答保持同一描述性边界；Analyst Prompt 与 Skill 补充“优先使用当前样本均值较低，程度描述不等于统计显著”的说明。渐进加载、Backend 和统计公式不改。trace 新增 `analysis_conclusions`，显示程序已经接收的子结论，便于下一次对照 Main 的报告与最终回答，不新增持久日志系统。

**范围与局限**：这是针对当前项目常见中文措辞的有限规则，不是完整语法或统计推理。复杂引用、跨句限定、双重否定、同义改写可能漏检或误报；它不验证任意派生数字，也不能证明每句结论正确。检查在验收阶段运行；文件若已写入，会作为未验收产物保留，不自动删除或覆盖。原有精确数值证据核验继续执行。

## Research Memory（Phase 2.3）

默认文件是工作区内的 `memory/research.json`，首次验收成功才创建；目录已加入 `.gitignore`。不使用向量检索、数据库、MemoryMiddleware 或 Checkpointer。

### 写入时机与字段来源

`run_agent()` 先完成原有 `verify_run()`：真实 task、分析执行、Skill 全文读取、写入、精确数值证据、报告和最终回答的统计语言均通过后，才调用 Python 的 `ResearchMemoryStore.save_verified()`。模型没有 Memory 写入 Tool，也不生成可信统计字段。写入失败会明确报告“报告验收已通过，但 Memory 保存失败”，退出码为 1，不打印记忆保存成功。

文件结构（省略实际数值）：

```text
{
  version: 1,
  checksum: SHA256(version + records 的规范 JSON),
  records: [{
    experiment_id: 稳定 SHA256,
    facts: {
      function_name, algorithms, source: {csv_path, sha256, data_kind, note},
      data_kind,
      configuration: {objective, fe_grid, observed_seeds},
      analysis_settings: {monotonic_atol, sample_std_ddof},
      final_statistics: {算法名: {n_seeds, final_by_seed, mean, sample_std, median, best, worst}},
      significance_tested
    },
    interpretation: {
      status: "model_interpretation_not_facts",
      conclusions: {final_performance, stability, convergence}
    },
    unverified: {status: "unverified", hypotheses: [], next_experiments: []},
    reports: [{path: "reports/<name>.md", sha256}]
  }]
}
```

`facts` 只取自通过验证的 `analysis`，保存前独立重算比对。配置只含实际观察到的 FE/种子和真实分析设置，不补造算法超参数、维度或正式 CEC2013 预算。`interpretation` 保存与报告所用 task 对应的子 Agent 结论，并明确标为模型解释；没有把它转成事实。当前输出协议没有提出新假设或下一步实验，所以相应数组为空，不自动编造研究计划。Memory 不保存 Messages、最终回答全文、Skill 正文或完整报告。

### 幂等性与持久化

`experiment_id` 来自数据 SHA256、函数、算法、数据性质和真实配置/分析设置，独立于模型措辞、报告文件名或相同 CSV 的路径别名。同一实验重跑只合并新的报告引用，保留首条来源和解释；同一报告重复保存不重写文件。不同数据、函数或设置分别记录。

JSON 采用同目录临时文件、flush/fsync 后原子替换；失败不破坏原文件。支持不同进程**顺序**运行，不提供并发写锁或多写者合并。

### 检索与上下文注入

```python
store = ResearchMemoryStore(workspace)
related = store.retrieve(function_name="F1", algorithms=["A", "B"], limit=3)
reference = history_summary(related, max_chars=4000)
```

函数精确匹配；算法筛选要求包含全部指定算法；按实验插入顺序从新到旧返回。`function_name=None` 可单独按算法查询。空文件库正常返回空列表，不创建假历史。JSON 文件会整体解析并校验，但只有筛选后的记录进入上下文。

当前 CLI 的任务仍是比较 A/B，检索条件使用本次函数与 A/B，默认最多 3 条、总计 4000 个 Unicode 字符。摘要只保留完整记录能放下的项，不截断 SHA256 或 JSON；包含数据性质、来源、样本数、观察到的最后 FE、Mean/Std/Median 和报告引用。没有相关记录或长度预算不足时不注入。

历史摘要作为当前任务之前的独立 **Human 参考资料消息**，不拼接进 System Prompt；静态 System 规则明确历史是数据而非指令，当前任务必须重新计算。自由文本解释、假设、建议不注入，只保存在明确分类的 JSON 中，因此模型解释不能变成新的历史指令或可信数字。Main/子 Agent 的原有工具权限不变，当前统计仍由原有验证链保证。

trace 会出现：

```text
memory_retrieved matched=1 injected_chars=...
...task、Skill、统计、write_file 与完整验收...
memory_saved {"experiment_id": "...", "created": false, "path": ".../memory/research.json"}
```

### 无效文件与信任边界

读取和保存时验证版本、结构、全文件 checksum、experiment_id，并使用保留的原 CSV 独立重算 `facts`、核对每个报告 SHA256。即使有人重算了 JSON checksum，伪造 Mean 仍会被拒绝。损坏、字段类型错误、来源丢失或内容变化会明确失败，不静默忽略或自动清空；请检查文件并恢复正确的来源/报告。

**应为不同实验保留独立 CSV 和报告，不覆盖历史来源。** checksum 不是数字签名，不能证明恶意用户同时改写全部文件后的真实性；它不提供身份认证。模型解释始终不是可信统计事实，统计语言规则也不保证所有语义正确。JSON 存储用于本地可信工作区，不是生产审计服务。

### 跨进程验收演示（完全离线）

```bash
uv run pytest -q -s tests/test_memory.py::test_memory_persists_across_fresh_processes_with_unrelated_control
```

该测试创建临时工作区并依次启动三个独立 Python 进程，运行真实 Agent 图和 Fake Model：首次 F1 保存 Memory；无关 F2 单独保存；全新进程再做 F1。检查实际 Main 输入含第一份 F1 的来源/报告，却不含 F2；最终 JSON 仅两条实验，F1 的两份报告合并。各子进程额外禁止 HTTP/socket，不能借用真实 API。

本次实际输出：

```json
{"process_pids": [89782, 89783, 89784], "first_history_count": 0, "second_F1_history_count": 1, "unrelated_F2_excluded": true, "persisted_record_count": 2}
```

也可在已配置模型的终端里分两次执行（真实 API，可能付费）：

```bash
uv run python -m research_agent.main --output reports/F1_phase23_first.md --trace
uv run python -m research_agent.main --output reports/F1_phase23_second.md --trace
```

首次使用空库时命中 0；第二次是新 CLI 进程，应出现相关历史命中，且统计仍重新执行。两次都要完整验收才保存 Memory；目标报告已有时需换新文件名。

## 核心函数与 Tool Schema

- `analysis.py:analyze_dataframe()`：可独立测试的确定性分析，不修改输入 DataFrame。
- `analysis.py:analyze_csv()`：读取同一份 CSV 字节并计算 SHA256，再调用分析函数。
- `analysis.py:evidence_markdown()`：只格式化数值证据表，既不解释结论，也不生成完整报告。
- `tools/experiment.py:make_analyze_experiment(root)`：以闭包绑定工作区，避免依赖 cwd。
- `agent.py:load_model()` / `build_agent()`：加载配置、创建模型、Backend 和真实 Agent。
- `subagent.py:make_analyst()` / `verify_analysis()` / `RunEvidence`：隔离分析子图、核对输出、保存运行级可信证据。
- `skill_backend.py:SkillReadBackend`：Skill 发现/加载的单文件白名单和全文读取凭据。
- `memory.py:ResearchMemoryStore` / `history_summary()`：可信记录保存、校验、精确筛选和有界历史摘要。
- `main.py:run_agent()` / `verify_run()`：检索历史、执行图、完整验收，最后持久化 Memory。

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
   正常案例还会新建 Store，从磁盘读取并检查实际保存的 Memory。

2. 写入失败案例：临时工作区包含同一 Skill 和样例 CSV，用名为 `reports` 的普通文件阻止创建报告目录。要求观察到真实 `write_file` 错误 ToolMessage、无报告文件、验收函数拒绝返回成功，并且没有写入 Memory。这个案例也调用真实模型，不使用 Fake Model。

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
| Phase 2.3 修改前基线 | **130 passed**，含 Skill、工具隔离和统计语言回归 |
| Phase 2.3 全套离线测试 | **153 passed**；原 130 项保留，新增 23 项 Memory 用例 |
| 持久化与幂等性 | 验收后保存；相同实验单记录合并报告；不同实验分开；完全重复保存文件字节不变 |
| 检索与注入 | 按函数、算法和数量；无历史正常；长度限制；模型自由文本不注入 |
| 可信性与失败 | 真实分析字段一致；伪造统计/重算 checksum 仍拒绝；无效 JSON/类型/来源或报告变化明确失败；写入失败不报成功，原子替换失败保留旧库 |
| 跨进程演示 | **1 passed**；三个不同 PID，F1 第二次命中，F2 对照未注入 |
| 2026-11-21 程度措辞调整 | **158 项离线测试通过**；单纯“显著低于/更低”放行，显式统计显著性断言仍拒绝；覆盖子结论、报告、最终回答与验收后 Memory 保存 |
| Phase 2.3 真实模型集成 | **本次未执行**；保留独立手动入口，不与 Fake Model 结果混同 |

整个 pytest 套件禁止 HTTP 和 socket；跨进程 worker 也显式禁止网络。Fake Model 测试证明持久化、调用协议和数据边界，不证明真实模型一定正确理解历史或专业规范。普通环境运行 `uv run pytest -q`；本机使用可写缓存 `UV_CACHE_DIR=/private/tmp/research-agent-uv-cache`。

## Phase 2.3 范围与限制

只增加本地 JSON Memory 和 Python 编排，未增加子 Agent、Memory 写入 Tool、MemoryMiddleware、Checkpointer、向量数据库、RAG、MCP、Planning、HITL 或前端。Main 和 Analysis Subagent 的工具列表、Skill 白名单及统计公式不变。

仅支持本地单写者、顺序多进程运行；原始 CSV 与报告需保留。有限统计语言检查与 SHA256 完整性校验不等于通用语义正确性或密码学认证。真实模型如何使用相关历史仍需手动验证。

版本依赖未升级：DeepAgents 0.7.40 与 LangChain 1.5.2，原有 Skills/task 兼容点继续由回归测试覆盖。支持源码工作区运行；wheel 不单独打包 skills/data 资源。

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
│   ├── memory.py
│   └── tools/
│       ├── __init__.py
│       └── experiment.py
├── scripts/
│   ├── generate_sample.py
│   └── manual_integration.py
├── skills/experiment-analysis/SKILL.md
├── data/sample_experiments.csv
├── reports/.gitkeep
├── memory/research.json  # 验收后生成，不提交 Git
└── tests/
    ├── conftest.py
    ├── test_analysis.py
    ├── test_experiment_tool.py
    ├── test_agent_setup.py
    ├── test_skills.py
    ├── test_statistical_language.py
    ├── test_memory.py
    └── memory_worker.py
```

GitHub 仓库：[077cheerup/research-engineer-agent](https://github.com/077cheerup/research-engineer-agent)。按完成并验证的阶段提交和推送，密钥、运行报告与研究记忆不上传。
