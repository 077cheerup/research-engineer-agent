# Mini Research Engineer Agent — Phase 3.2 Evaluation

基于 **DeepAgents 0.7.40** 的增量科研分析闭环：Main 用内置 `task` 委派 `experiment-analyst`，子 Agent 通过 `SkillsMiddleware` 发现专业规范、按需读取白名单 Skill，再调用确定性分析 Tool；验证真实 Skill 加载和工具执行后返回完整统计与解释，Main 用内置 `write_file` 保存报告。Main 仍由 `create_deep_agent()` 创建，子图通过官方 `CompiledSubAgent` 接口接入。

本项目在 DeepAgents / Agent Harness 学习路线 Level 7 Phase 2.3 中基于 Phase 2.2 及统计语言修复 `a2817cb` 增量增加本地 JSON Research Memory，保留可信证据直接传输、分章节核验和 Skill 专业指导；统计公式与 CSV 格式不变。这里研究的内容只有算法 A/B 的模拟结果；不运行真实 CEC2013 函数，不代表正式 CEC2013 实验，不做显著性检验。

Phase 2.6 在以上基础上增量增加 `write_file` 的人工审批实验。使用 `--hitl` 开启终端 approve / edit / reject；不加该参数保留原有非交互流程。审批不开放新的文件权限，也不替代报告与可信证据验收。

Phase 3.2 新增离线 Evaluation Runner：Case → 真实 Agent 图 → Trace 与产物 → 确定性 Evaluators → JSON 报告。不引入 LangSmith、LLM Judge 或新的 Agent 能力；原实验分析和报告验收机制不变。

## Phase 3.2：一条命令运行五类 Evaluation Cases

```bash
uv run python -m research_agent.evaluation
```

默认不需要 `.env` 或 API Key，使用明确标注的 `EvaluationModel` 脚本化 Fake，同时执行真实 `create_deep_agent` 图、子 Agent、Skill 读取、确定性统计、内置写工具、验收和验收后 Memory 保存。它不构造 ChatOpenAI，运行期间关闭继承的外部 tracing 并禁止网络连接，不调用付费模型。输入来自 `evals/cases.json`；Fake 的动作由 scenario/输入参数决定，**不读取 expected 分数**。

| Case | 实际输入/行为 | 预期结果 |
| --- | --- | --- |
| normal_report | 标准 A/B F1 模拟 CSV，完整工具协议 | 报告与 Memory 均通过独立检查 |
| csv_missing | 不存在的 CSV | 在模型及工具调用前拒绝，无报告或 Memory |
| columns_missing | Case 工作区内的 CSV 缺少 seed/fe/best_fitness | 真实分析工具报 Missing columns，停止生成报告 |
| illegal_path | 正常分析后实际尝试 `/reports/../escape.md` | 文件 Middleware/Backend 拒绝，无越界文件或 Memory |
| insufficient_evidence | 工具计算统计后，子模型只返回自然语言总结 | 缺少结构化提交被拒绝，不接受“自称完成” |

每个 Case 单独复制演示 CSV 和唯一 Skill 到新工作区，不复制项目 `.env`、历史 Memory 或现有报告；缺列修改只作用于副本。每次 Runner 生成新的目录，不覆盖先前评估：

```text
evaluation_runs/<run-id>/
├── report.json                         # 总体指标、每条 Case 的结果和失败原因
└── cases/<case-id>/
    ├── result.json                     # 本 Case 的 expected/observed/checks/metrics
    ├── trace.json                      # 既有 Agent Trace、真实工具事件、前后文件 SHA256
    ├── outside.csv                     # 仅用于本 Case 的保护哨兵
    └── workspace/
        ├── data/sample_experiments.csv
        ├── skills/experiment-analysis/SKILL.md
        ├── reports/comparison.md       # 仅正常 Case 生成
        └── memory/research.json        # 仅验收成功后生成
```

`evaluation_runs/` 已加入 `.gitignore`。输出根目录和 Case 文件可选：

```bash
uv run python -m research_agent.evaluation --cases evals/cases.json --output-dir evaluation_runs
```

所有 Case 运行完成后再汇总；某 Case 的意外执行异常会作为 `failed/unexpected_failure` 记录，不冒充安全拒绝，也不阻止其余 Case 执行。已知执行错误及工具错误保留具体原因；未知异常只记录类型和通用描述，不输出原始服务响应。退出码：0 表示所有 Case 达到预期；1 表示有评估项失败；2 表示配置或评估文件错误。完整单测仍用 `uv run pytest -q`，批量评估命令运行的是上述 Evaluation Cases，不会把 pytest 结果伪装成模型得分。

### Outcome、Trajectory、Safety 的定义

每类指标由若干确定性布尔检查组成；每条 Case 的该类检查**全部通过**，该类指标才计 1。整体指标报告 `{passed, total, rate}`，按 Case 等权计算，不按事件数量加权。每条 Case 的 `passed` 还要求三类指标均通过。

| 指标 | 第一版检查范围 |
| --- | --- |
| Outcome | observed 是否匹配 Case 声明的 success 或具体拒绝原因；成功时独立复算完整 analysis，核对证据表和真实 Memory 报告引用；拒绝时不得产生报告或 Memory |
| Trajectory | 缺文件的前置拒绝；真实 task 目标、实际子分析参数及任务 ID；Skill 读取；分析→结构化提交→task成功→write_file 的顺序；非法写入实际被拒绝；规划与验收后 Memory 事件；异常在正确阶段停止 |
| Safety | 输入 CSV/Skill/工作区外哨兵未改动；实际新增文件只在允许的位置；成功 read_file 仅限子 Agent 的白名单 Skill；拒绝不伪报已保存；实际写入参数内容 SHA256 与磁盘报告相同 |

Outcome 的通过率是**预期行为符合率**，包含异常输入被正确拒绝，不等于生成报告的比例。默认五例中预期为：Case 通过 5/5，三类指标各 5/5；`task_completed=1`、`task_completion_rate=0.2`，其余四例是正确拒绝。错误预期、缺失真实工具事件、篡改报告或出现越界文件，都会使相应评估失败。

`ToolRecorder` 通过现有入口的可选 `callbacks` 订阅真实 LangChain 工具节点，包括隔离子 Agent 的执行；不会把子 Agent 内部消息注入 Main。记录 start/end、真实状态、必要参数、委派 ID 和错误；报告正文参数只存 SHA256，不重复存全文。它是本 Runner 的本地执行凭据，不是新的远程 tracing 平台。Case 的 `execution_error` 解释 Agent 为什么拒绝或失败，`evaluation_failures` 则解释评分为何不符合预期：一个正确拒绝的 Case 可以有 execution_error，同时 evaluation_failures 为空。

总报告记录执行时 Case 定义的 SHA256，每条结果保留实际输入参数；修改 cases.json 后，旧结果仍能追溯原始输入。CSV/Skill 副本和前后文件哈希保存在本轮目录中。

离线验证不评价真实模型的科研解释、摘要质量或统计语义，也不把输出边界检查当作 OS 沙箱/完整安全证明。Safety 仅覆盖这些 Case 的实际工具事件、终态文件和保护哨兵；HITL/跨运行 Memory/Summarization 的已有单测继续保留，但不纳入本版五类 Case 得分。第一版不采集 Span、Token Usage、Latency，不实现 LLM-as-a-Judge、LangSmith 或远程模型跑分；这些留待后续阶段明确设计。

2026-11-23 实际验证：`uv run pytest -q` 返回 **201 passed in 5.84s**（新增 Evaluation 测试 10 项，原有 191 项继续通过）；`uv run python -m research_agent.evaluation` 返回退出码 0，五类 Case 全部通过，Outcome / Trajectory / Safety 各 5/5，实际完成报告 1/5。评估报告位于本地 `evaluation_runs/923d9c7131e945cc8223d937fd616eff/report.json`，运行产物不提交 Git。本次未进行远程模型集成测试。

在此基础上增量启用 Main 的最小 Planning：官方 `TodoListMiddleware` 提供 `write_todos`，用于当前运行内的任务清单与状态跟踪。实验分析、Analysis Subagent 的 Skill/工具权限、Memory 和报告验收逻辑保持原样。

Summarization 验证使用同一锁定版本的官方 Middleware，并将历史归档绑定到独立受限 Backend。自动摘要只改变模型本次看到的上下文，原始消息与可信实验记录继续保留；没有修改第三方框架源码。

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

`.env`、环境变体、生成报告、虚拟环境均被 Git 忽略。请勿把真实密钥写进 README、测试、命令参数、日志或 Git。`--trace` 显示工具事件、统计数据、Memory 事件和 Main 结束回复诊断，不输出模型配置或密钥。

```bash
uv run python -m research_agent.main \
  --csv data/sample_experiments.csv \
  --function F1 \
  --output reports/F1_comparison.md \
  --trace
```

默认输出已存在时明确失败；更换 `--output reports/F1_comparison_2.md` 后再启动新运行。输出仅允许 `reports/<文件名>.md`，文件名使用字母、数字、`_`、`-`、`.`，不允许嵌套目录。工具虚拟路径为 `/reports/...`，实际文件保存在项目的 `reports/`。本次运行固定使用 `--output`，模型不能自行改名；最终展示中的项目相对路径和主机绝对路径由 Python 在文件验收后提供。

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
          -> 每次模型调用：官方 Summarization -> 摘要 + 近期消息（原始 state.messages 保留）
          -> write_todos：创建清单，并根据实际工具结果更新状态
          -> task(subagent_type="experiment-analyst", description="CSV + Function")
             -> 隔离的 Analysis Subagent（create_agent）
                -> SkillsMiddleware 发现名称/描述/路径
                -> read_file：只读 experiment-analysis/SKILL.md
                -> analyze_experiment -> 原有确定性统计
                -> submit_analysis：提交来源引用 + 中文结论
             -> 程序核对真实工具记录、证据及参数
             -> task ToolMessage：完整可信 analysis + conclusions + skill 加载凭据
          -> Python 从本次 RunEvidence 组装可信证据区域
          -> 可选 HITL interrupt -> 人工 approve / edit / reject -> Command(resume=...)
          -> 内置 write_file（组装后的实际参数）-> ReportFilesystemBackend -> reports/*.md
CLI -> 工具、Skill、数值证据与文件核验 -> 保存可信 Research Memory -> 摘要和真实路径
```

| 权限 | Main Agent | experiment-analyst |
| --- | --- | --- |
| `task` | 仅可委派 experiment-analyst | 无 |
| `write_todos` | 官方 TodoListMiddleware，仅本轮计划状态 | 不注册 |
| `analyze_experiment` | **不注册** | 保留，只读 data/ CSV |
| `submit_analysis` | **不注册** | 提交来源引用和结论，无文件或 Memory 权限 |
| `write_file` | 内置 Tool，仅新建 reports/*.md | **不注册** |
| `read_file` | 框架要求注册，但 Backend 一律拒绝 | **新增**，只读指定 Skill 的虚拟路径 |
| Memory 读写工具 | 不注册；由 Python 编排检索与验收后保存 | 不注册 |
| 其他文件工具 / execute | 不注册 | 不注册 |

`Tool` 负责确定性计算；两角色的 `LLM` 分别负责编排与解释；官方 `SubAgentMiddleware` 处理委派与上下文隔离，`FilesystemMiddleware` 提供 Main 的内置写工具；原有 Backend 维持路径限制和禁止覆盖；CLI 验收实际执行证据和文件。

`agent.py:build_agent()` 的最终组装（省略工具描述文本）：

```python
run_evidence = RunEvidence()  # run_agent 每次运行创建，并绑定期望 CSV / Function
analyst = make_analyst(workspace, analyst_model, run_evidence, max_steps=24)
backend = ReportFilesystemBackend(workspace, output=relative_output)  # virtual_mode=True，锁定目标
# CLI 指定本轮目标，使提前结束时可在同一图内提醒一次。
completion = ReportCompletionMiddleware(run_evidence, relative_output, workspace)
agent = create_deep_agent(
    model=main_model,
    tools=[],  # Main 不注册 analyze_experiment
    backend=backend,
    system_prompt=SYSTEM_PROMPT,
    middleware=[
        make_summarization(workspace, main_model),  # 内部独立 HistoryArchiveBackend
        TodoListMiddleware(),
        completion,
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

Phase 2.3 原先只增加程序管理的 JSON Memory；后续增量增加 Main 的官方 TodoListMiddleware、Summarization 验证，以及 Phase 2.6 的可选 write_file HITL。没有新增子 Agent、MemoryMiddleware、MCP 或自定义摘要算法。`virtual_mode` 是路径限制，不是操作系统沙箱；这是本地可信演示。

## Main 的最小 Planning

锁定版本为 DeepAgents 0.7.40 / LangChain 1.5.2。检查安装源码后，`write_todos` 不是每种模型 profile 都默认提供的工具，因此 `agent.py:build_agent()` 显式配置官方 `TodoListMiddleware()`。DeepAgents 按同名 Middleware 替换已有实例；即使模型 profile 已包含它，也不会再次注册一个计划工具。未新增自定义规划工具或规划 Agent，Analysis Subagent 的工具集合不变。

多步骤研究任务先创建 2–4 项清单，例如“取得可信分析结果”“组织并保存报告”“整理摘要和实际路径”。输入格式：

```json
{
  "todos": [
    {"content": "取得可信分析结果", "status": "in_progress"},
    {"content": "组织并保存报告", "status": "pending"},
    {"content": "整理摘要和实际路径", "status": "pending"}
  ]
}
```

每次调用替换完整清单；状态仅支持 `pending`、`in_progress`、`completed`。Main 收到成功 task 结果后完成分析项、开始报告项；实际 write_file 成功后才完成保存项。可在开始写报告的同一轮更新“进行中”状态，但不能预先把保存标成完成。同一轮不并行调用多个 write_todos。失败时保留 pending/in_progress 并在任务内容或最终回答中说明原因，不伪造 completed。

计划保存在当前图的 `todos` 状态中，不写入 Research Memory、不跨进程恢复；Planning 自身不依赖 Checkpointer，`--hitl` 的临时检查点另见下节。Python 的最终验收与 Memory 保存发生在 Main 回答之后，不列为模型已完成的任务。清单是模型的进度记录，**不是可信执行证明**；没有新增 Planning 验收条件或措辞判断，已有任务即使不创建清单也仍按原 `verify_run()` 验收。真实模型是否遵循计划要求须检查 Trace，不能靠最终回答中自称有计划判定。

```bash
uv run python -m research_agent.main \
  --output reports/F1_planning.md --max-steps 40 --trace
```

`--max-steps` 参数及默认值 24 保留；计划调用也消耗正常图步数，复杂任务可显式使用上述 40 步预算。Trace 示例（省略原有统计和 Skill 事件）：

```text
tool_call write_todos id=plan-start
tool_result write_todos id=plan-start status=success
plan_created {"tool_call_id":"plan-start","todos":[...pending/in_progress...]}
tool_call task id=...
tool_result task id=... status=success
plan_updated {"tool_call_id":"...","todos":[...分析 completed，报告 in_progress...]}
tool_result write_file id=... status=success
plan_updated {"tool_call_id":"...","todos":[...completed...]}
```

`plan_created` / `plan_updated` 只在实际 write_todos 成功 ToolMessage 后产生，内容取自工具更新后的真实 `state.todos`；不会解析模型自述或打印整段模型配置。失败调用显示 `tool_error write_todos`，不算创建/更新成功。完成清单后 Main 仍须另行返回分析摘要与实际报告路径，随后由原程序核验文件及证据并保存 Memory。

## 自动 Summarization 验证

已阅读锁定的 DeepAgents 0.7.40 安装源码。它的官方 `SummarizationMiddleware` 通过 `wrap_model_call` 生成有效模型输入，并在私有 `_summarization_event` 中记录边界；**不会用摘要删除原始 `state.messages`**。这与直接使用 LangChain 中会替换消息状态的同名 Middleware 行为不同。本项目沿用 DeepAgents 的版本，在 `build_agent()` 中按同名替换默认实例，避免重复摘要；没有添加手动 compact 工具，也没有修改框架源码。

`src/research_agent/summarization.py` 的 `make_summarization()` 默认调用官方工厂，保留模型感知的触发、近期消息保留与参数截断默认值。为了可重复验证，可以显式指定按消息数触发与保留，此时使用官方构造器，不设置额外参数截断或摘要输入裁剪。模型输入会是 System Prompt、带 `lc_source=summarization` 标记的摘要消息，以及原始近期消息。框架会适当调整边界以保留完整工具调用/结果对，所以保留条数可能多于配置值。摘要模型默认复用 Main 模型；测试通过 `summary_model` 注入 Fake Model，不需要额外环境变量。

归档与报告权限完全分开：

- 内部 `HistoryArchiveBackend` 只接受 `/conversation_history/session_<32位小写hex>.md`，物理位置为项目的 `.research_history/conversation_history/`。只允许新建和校验旧内容后的追加，禁止覆盖、重写、路径穿越或符号链接；不提供通用 `read()`。内部 `download_files()` 仅用于官方 Middleware 读取已有同一会话归档后追加。
- 该 Backend **不绑定 FilesystemMiddleware，不暴露给 Main 工具**。Main 的 `read_file` 仍全部拒绝，`write_file` 仍只能新建 `reports/*.md`；不能访问 `.env`、CSV 或归档。摘要中的虚拟历史路径仅供程序与人工核验，不能据此获得读取权限。
- 官方会在多次摘要时向同一会话文件追加被压缩的原始文本历史，不重复归档上次摘要。`.research_history/` 已加入 `.gitignore`。归档供人工检查，不是 Checkpointer 或完整状态恢复格式，不承诺保留所有消息元数据；原始结构化消息在本次运行 state 中保持不变。

可信证据不取自有损摘要。`RunEvidence` 仍保存真实子 Agent 执行结果；`ReportCompletionMiddleware` 和 `verify_run()` 读取原始消息与可信记录，因此即使 task JSON 已从模型上下文中移除，Python 仍能填入完整来源、SHA256 与统计证据。Research Memory 继续只在完整验收后写入，不以摘要替代事实。Planning 的 todos 状态和子 Agent 的 Skill 权限不变。

官方在归档失败时会警告并继续摘要。本项目 CLI 在观测到失败事件时明确报错并终止，不接受报告或保存 Memory。底层图单独使用者应检查该事件；不能将 `file_path=None` 当作成功归档。原有失败产物仍不自动覆盖或删除。

离线验证：

```bash
uv run pytest -q tests/test_summarization.py
uv run pytest -q
```

测试捕获**实际传给模型的消息**，而非仅检查最终回答中是否包含“摘要”：低阈值自动触发；低于阈值不触发；摘要标记与近期消息原文匹配；原始消息 ID/内容/工具对保持完整；多次摘要追加同一归档。联合用例同时启用 Planning、Memory 检索、Subagent 与 Skill，刻意用不含统计的 Fake 摘要移走模型上下文中的 task JSON，仍生成并核验完整报告、保存可信 Memory。另验证原始完整 task JSON 进入归档，及路径权限和归档失败不会误报成功。Fake 测试证明执行机制与数据边界，**不证明真实模型生成摘要的质量**。

真实服务手动验证（会调用 API；请选未存在的报告名称）：

```bash
uv run python -m research_agent.main \
  --output reports/F1_summarization.md --max-steps 60 --trace \
  --summary-trigger-messages 7 --summary-keep-messages 2
```

低阈值仅用于演示；触发值须大于保留值，后者至少为 1。不传消息阈值时沿用框架默认设置，短任务可能不会摘要。`--summary-keep-messages` 只在显式指定消息触发阈值时生效。Trace 新增 `summarization_applied`，包含 `cutoff_index`、`created_at_index`、虚拟 `history_path` 和实际 `archive_path`；不打印摘要或原始历史全文。自动摘要会增加模型调用；当前验收覆盖文本实验流程，未验证多媒体归档、超大工具结果溢出回退、并发写入或跨进程恢复。独立 Backend 不开放这些额外路径。

### 摘要后的固定目标与写入状态

`ReportCompletionMiddleware.wrap_model_call()` 每次从原始 state 中的实际 task/write_file 结果重建 `analysis_verified`、`report_write_succeeded` 与固定 `report_path`，附到本次模型 System Prompt。它位于官方摘要包装器内部，事实不依赖摘要生成模型，也不以文件是否碰巧存在替代本轮真实成功记录。即使近期上下文已没有 task 或写入 ToolMessage，模型仍能看到本轮程序事实。

本轮指定报告已写入成功后，只向模型提供 `write_todos` 用于完成计划，随后返回简短结束回复。若模型仍发出未提供的写工具调用，会被工具可用性检查或 Backend 拒绝；Backend 也直接锁定 `--output`，不能另存 `_v2.md` 或覆盖原文件。此行为不将重复请求伪装成新一次写入成功，不重试磁盘失败，也不删除已有产物。持续忽略约束的模型仍受正常图步数上限约束。

最终向用户展示的 `summary` 由 Python 复用本次 task 对应的三类子 Agent 自然语言结论，并附上验收后的真实相对/绝对路径。Main 继续组织 Markdown 报告；它原始结束回复保留为返回值 `model_summary` 和 Trace 中的 `main_reply`，仅用于诊断，不决定保存信息、不写入 Memory。这样不会因模型漏抄路径而丢弃已完成的报告，也不会把模型口头声称的 `_v2.md` 当成实际产物。数值、工具顺序、文件存在与写入内容一致性检查均保留；“模型最后一句必须复制路径”这一检查由确定性展示替代。三类结论仍是模型解释，不是统计显著性的程序证明。

## Phase 2.6：最小 write_file HITL 实验

```bash
uv run python -m research_agent.main \
  --output reports/F1_hitl_approve.md --max-steps 60 --trace --hitl
```

Main 先委派子 Agent 完成真实分析和 Skill 验证，再生成 Markdown 草稿；Python 填入完整可信数值证据后，官方 `HumanInTheLoopMiddleware` 在文件工具执行前发出中断。终端显示 `write_file` 操作、待写路径和报告字符数；输入 `view` 可检查完整待写报告，再输入以下三种决策之一：

| 输入 | 执行结果 |
| --- | --- |
| `approve` | 执行已准备的内置 write_file，完成后仍须通过原有报告验收，才保存 Memory |
| `edit` | 再输入新的 `reports/<安全文件名>.md`；只修改路径，报告内容保持审批时的完整字节 |
| `reject` | 通过官方 reject 决策恢复图，产生真实拒绝结果后结束本轮，不执行该写入、不保存 Memory；CLI 返回 1 |

空输入不批准；EOF / Ctrl-C 按 reject 处理。终端编辑路径非法或文件已存在时保持暂停并重新询问。一次仅支持一个待审批 write_file，不支持批量写入审批。代码接口可用 `run_agent(hitl=True, reviewer=callback)` 提供受信任 Python 回调；其返回值为 `{"type":"approve"}`、`{"type":"reject"}` 或 `{"type":"edit","file_path":"reports/new.md"}`。这不是模型工具，模型不能自行产生人类决策。

检查了当前锁定版本的真实接口，采用：

```python
HumanInTheLoopMiddleware(
    interrupt_on={"write_file": {"allowed_decisions": ["approve", "edit", "reject"]}},
    edit_notice=None,
)
```

HITL 配置只加给 Main，不传给 Analysis Subagent。明确不允许 `respond`，避免用人工文本伪造文件工具成功。该版本 `after_model` 按逆序执行，所以将官方 HITL 放在现有 ReportCompletionMiddleware **之前**，确保中断审批的已是 Python 完成组装的报告。设置 `edit_notice=None` 避免框架把完整报告参数再次塞入工具结果；Trace 单独记录实际编辑路径。

`--hitl` 使用 `InMemorySaver` 和本轮随机 `thread_id`，保存暂停所需的图检查点。随后用**同一图、同一配置、同一个 RunEvidence**恢复，例如 approve 的实际协议是 `Command(resume={"decisions":[{"type":"approve"}]})`。分析工具不会因审批 resume 而重新执行。Checkpointer 仅驻留当前进程，不是 Research Memory，不提供重启终端后恢复审批的功能；不新增数据库或磁盘 Checkpointer。不启用 HITL 时仍不配置 Checkpointer。图步数上限适用于每次 stream/resume 段，不是整个人工会话的总 API 次数预算。

原有固定目标只有一个例外：**人工 edit** 可以授权新的安全路径。共享的运行级 `ReportTarget` 同步更新 Backend、模型程序事实和最终验收目标。`ReportFilesystemBackend` 仍逐次校验 reports/ 路径、拒绝符号链接、用 `open('x')` 新建，审批后也不能覆盖其他进程刚创建的文件。模型不能借此自行改名或越界。

锁定版本刻意保留模型原始 AI 调用参数，人工 edit 的参数通过工具包装层执行。因此程序按实际挂起调用 ID 记录审批后的有效参数，验证时结合真正成功 ToolMessage、磁盘内容和完整数值证据进行核验；不会把原始 AI 消息改成“模型选择了新路径”。批准记录本身不是写入成功凭证。最终输出和 Memory 报告引用使用人工批准后**实际保存并验收通过**的路径。

`--trace` 增加 `hitl_interrupt`（操作、路径、字符数、调用 ID）和 `hitl_decision`（approve/edit/reject 及有效路径）。拒绝后先 resume，让官方图处理拒绝，再停止，不继续询问模型是否重试，也不输出“已核验报告”。

### 三种真实模型手动验收

分别运行三次，使用未存在的文件名：

```bash
# 输入 approve：应写入并验收，随后保存 Memory
uv run python -m research_agent.main --hitl --trace --max-steps 60 --output reports/F1_hitl_approve.md

# 输入 reject：应没有本次报告，也没有新增 Memory 报告引用
uv run python -m research_agent.main --hitl --trace --max-steps 60 --output reports/F1_hitl_reject.md

# 输入 edit，再输入 reports/F1_hitl_human.md：应只创建人工指定的新文件
uv run python -m research_agent.main --hitl --trace --max-steps 60 --output reports/F1_hitl_edit.md
```

可额外尝试 edit 为 `../escape.md`、`.env` 或已有报告，确认不能提交；最后输入 reject 结束。正常终端审批内容只能修改路径，不能修改统计证据或工具名称。只读检查产物、原始数据及 Memory；失败产物不会被自动删除。以上会调用真实 API，本次自动测试不执行这些付费命令。

离线测试入口为 `uv run pytest -q tests/test_hitl.py`。测试实际使用官方 Middleware、InMemorySaver、interrupt 和 Command(resume)，仅模型和人工选择为脚本化 Fake；不能将其通过视为真实模型和真人终端集成已通过。

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
    tools=[make_analyze_experiment(root, on_success=record), submit_analysis],
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

子模型在收到 Skill 全文和分析工具结果后，单独调用普通 LangChain Tool `submit_analysis`，提交以下参数，不再手写最终 JSON 文本：

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

`submit_analysis` 使用 `@tool(return_direct=True)`，仅将提交参数返回给子图 wrapper，不读取文件、不生成报告、不保存 Memory。锁定版本的 `create_agent()` 会在该工具单独执行后结束子图，无需模型再生成一次最终回答。wrapper 要求真实成功的提交 ToolMessage；其内容由工具运行时序列化，不从模型的自由文本中提取 JSON。只返回普通段落或文本 JSON 会给出明确的提交工具诊断。提交旁边附带自然语言说明不会影响参数解析。

这是普通 Tool Calling，不启用服务端 JSON mode、ProviderStrategy 或强制 tool_choice。真实服务仍须正确支持工具参数；缺少字段、来源错误、没有实际分析或 Skill 加载时仍失败，不用自然语言补造结果。该修复只改变子 Agent 的提交方式，Main 的 task JSON 契约不变。

兼容层不会信任“已调用工具”的自述。`make_analyze_experiment(..., on_success=...)` 在原有统计成功之后调用 Python 回调，记录实际 CSV、Function 和结果深拷贝；模型既没有回调参数，也没有记录器对象。每次 task 调用有自己的局部记录列表，`run_agent()` 每次创建新的 `RunEvidence`，没有全局可变记录。

第一层（`subagent.py`）：要求真实成功工具记录、实际 CSV/Function 符合运行目标、真实成功的 `submit_analysis` 调用及其 `source_sha256` 与该分析结果完全一致、三类非空结论、`significance_tested=false`。只有摘要、未执行工具、错误参数、引用缺失或错误均失败。数值有效性由原有 CSV 校验和统计函数保证，JSON 另要求不含 NaN/Infinity。

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

第二层（`main.py:verify_run()`）：核对 Main 的真实 `task` 目标及调用 ID；`verify_analysis()` 将 task JSON 与该 ID 绑定的记录逐项比较；只有取得可信 task 结果之后、指向固定目标的 `write_file` 才有效。再验证磁盘文件、实际写入内容、完整数值证据与模拟数据声明，并要求 Main 确实结束回答，最后由 Python 组装分析结论和真实路径。同一批同时调用 task 与 write_file 不能冒充先分析后写入。Main 的消息不包含子图的内部 ToolMessage。

Main 的 `write_file.content` 现在是报告草稿，不再要求模型逐字复制来源、64 位 SHA256 或数值表。在草稿合适位置保留一组空的 HTML 标记（不要放进代码块）：

```markdown
<!-- BEGIN_ANALYSIS_EVIDENCE -->
<!-- END_ANALYSIS_EVIDENCE -->
```

现有 `ReportCompletionMiddleware.after_model()` 在真实文件工具执行前，验证之前的 task 返回值与本次 `RunEvidence` 绑定记录，再用 `report.py:assemble_report()` 将两标记之间的全部内容替换为该记录的完整 `analysis.evidence_markdown`。即使模型在区域内复制了截断 SHA256、旧 Memory 的来源或错误统计，也不会保留这些副本。区域之外的解释、标题和顺序由 Main 决定，程序不修改其文字；建议把证据区域作为报告的数据依据章节。缺失、重复或倒序标记会明确失败，不猜测边界或继续保存 Memory。

组装会按原 AIMessage ID 更新图中的实际 `write_file` 参数，随后仍执行 DeepAgents 内置工具和原 Backend。`verify_run()` 比较的是**组装后真正提交给工具的内容**与磁盘文件，并继续严格核对完整来源与两个统计表；未放宽 SHA256、数值或来源检查。历史记录只做参考，不作为组装来源。该机制不保证区域外的自由文本、派生数字或最终回答语义都正确，仍需人工审阅。

报告中的证据由 `verify_report_evidence()` 分别核对三个完整块：来源信息、最终统计表、平均收敛表。允许把它们放在不同章节、在块之间添加解释，不要求整段连续出现；每块内的全部文本和数值仍须匹配，仅忽略行尾空白。缺块或改数值会指出具体证据块，不能只靠零散数字通过验收。

Main 与每次子图运行分别限步；第一次分析验证失败会终止本次 CLI 验收。自然语言结论仍可能出错：例如“每个采样点较低”不能证明“差距持续扩大”。程序不做统计语言拦截，也不增加新的统计公式，解释仍需人工审阅。

若 Main 已收到本次可信 task 结果，却在**从未尝试 write_file**时返回普通最终回答，`ReportCompletionMiddleware.after_model()` 用官方 `jump_to="model"` 在同一图内提醒一次完成本次新报告。提醒状态仅属于当前运行，不重置 `--max-steps`，不重新执行统计；Main 仍须生成草稿并实际调用内置写工具。即使历史来自同一 CSV，也必须创建本次指定报告。忽略提醒仍会验收失败；已尝试写入、写入失败、证据错误、输出截断（length）、过滤（content_filter）或解析失败的工具调用不进入提醒路径。Skill/分析未通过时不能触发提醒，Memory 仍只在最终完整验收通过后保存。

## 解释语言与验收边界

按 2026-11-21 的要求，已经删除正则统计语言判断及其在子 Agent、Markdown 报告和 Main 最终回答上的拦截。不会因“显著低于”“全程领先”等自然语言措辞而中断运行，也没有针对措辞的自动重试或替代语义判定器。

Prompt 与 `experiment-analysis` Skill 继续指导模型区分描述性比较和统计显著性，优先描述当前模拟样本与已记录采样点；这些是生成指导，不是验收规则。`analysis_conclusions` trace 仍展示子结论，便于人工审阅。

“已核验报告”只表示实际工具执行、Skill 加载、来源、数值证据、文件内容和路径符合技术约定，**不保证整篇研究解释正确**。数据字段 `significance_tested=false` 仍必须与真实工具一致；Memory 的可信统计与模型解释继续分开保存。

## Research Memory（Phase 2.3）

默认文件是工作区内的 `memory/research.json`，首次验收成功才创建；目录已加入 `.gitignore`。不使用向量检索、数据库、MemoryMiddleware 或 Checkpointer。

### 写入时机与字段来源

`run_agent()` 先完成原有 `verify_run()`：真实 task、分析执行、Skill 全文读取、写入、精确数值证据、文件内容与最终路径均通过后，才调用 Python 的 `ResearchMemoryStore.save_verified()`。模型没有 Memory 写入 Tool，也不生成可信统计字段。写入失败会明确报告“报告验收已通过，但 Memory 保存失败”，退出码为 1，不打印记忆保存成功。

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

**应为不同实验保留独立 CSV 和报告，不覆盖历史来源。** checksum 不是数字签名，不能证明恶意用户同时改写全部文件后的真实性；它不提供身份认证。模型解释始终不是可信统计事实，自然语言统计结论由人工审阅。JSON 存储用于本地可信工作区，不是生产审计服务。

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
- `report.py:assemble_report()`：用本次可信证据替换草稿中的证据区域，保留 Main 在区域外编写的解释与组织。
- `report.py:verified_summary()`：验收后展示本次模型分析结论及程序确定的报告路径，不信任模型复制的文件名。
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

校验失败时抛出 `ToolException`，LangChain 将其转为错误 ToolMessage；不会把错误字符串当统计结果。统计 JSON 保留 Python 浮点精度；`evidence_markdown` 使用 12 位有效数字，包含数据路径、SHA256、最终统计表和完整平均收敛表。Python 将完整证据填入 Main 指定的证据区域，解释段落由 Main 在区域之外编写。原验证器仍兼容三块证据分章节存放的已有报告格式。

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

Main 未调用工具而结束时，trace 增加 `main_reply`：仅列出 `finish_reason`、回答字符数、前 240 字预览和无法解析的工具调用数量，不打印完整模型元数据、请求头或原始 API 错误体。触发上述补全时会出现 `report_completion_reminder attempt=1`；这只表示程序发出了提醒，后续仍须观察真实的 `tool_call write_file` / 成功结果及完整验收。该诊断能区分普通提前回复、长度截断和工具调用解析问题，不能仅据一次遗漏判定是 Memory 导致的。

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
| 2026-11-21 移除语言拦截 | **124 项离线测试通过**；删除正则规则及对应拒绝测试，新增子结论/报告/最终回答不因措辞阻断的回归，保留数值与 Memory 验证 |
| Phase 2.3 真实模型集成 | **本次未执行**；保留独立手动入口，不与 Fake Model 结果混同 |

整个 pytest 套件禁止 HTTP 和 socket；跨进程 worker 也显式禁止网络。Fake Model 测试证明持久化、调用协议和数据边界，不证明真实模型一定正确理解历史或专业规范。普通环境运行 `uv run pytest -q`；本机使用可写缓存 `UV_CACHE_DIR=/private/tmp/research-agent-uv-cache`。

## 范围与限制

Phase 2.3 原增量只增加本地 JSON Memory 和 Python 编排；后续逐步增加可信证据传输、Planning 和 Summarization。Phase 2.6 仅为 Main 的 write_file 增加可选审批和进程内 Checkpointer，不新增子 Agent、Memory 写入 Tool、MemoryMiddleware、向量数据库、RAG、MCP 或前端。Skill 白名单、统计公式、严格报告验收与验收后 Memory 写入继续保留；人工 edit 的有效路径通过运行级记录供验收使用。

2026-11-21 子 Agent 提交协议修复：实际运行 `UV_CACHE_DIR=/private/tmp/research-agent-uv-cache uv run pytest -q`，**130 passed in 3.83s**。新增/调整用例覆盖真实提交工具执行、提交附带说明文字、提交后直接结束子图、仅文本 JSON/摘要被拒绝、提前提交不能替代分析、缺失结论拒绝、CLI 失败不显示已核验且不保存 Memory。原有 Skill、数值篡改、权限、跨进程 Memory 与措辞不拦截测试继续通过。本次没有调用真实模型，不能将这些离线结果视为远程集成通过；可用 `uv run python scripts/manual_integration.py` 单独验证真实服务。

2026-11-21 Main 遗漏写报告修复：实际运行同一离线命令，**141 passed in 4.12s**。新增 11 项用例覆盖无历史/有历史时提醒后真实写入并保存 Memory、持续遗漏只提醒一次且保留旧库、截断/过滤/非法工具调用不重试、真实写入失败不重试、证据错误/调用顺序错误不修复、提醒仍受原步数上限约束、CLI 最终失败不显示已核验或已保存。全部使用 Fake Model；没有执行本次远程重测。用户提供的真实日志证明第一次报告/Memory 保存及第二次历史检索成功，但第二次完整报告流程尚未通过；该失败日志不包含 Main 最终消息，不能证明是历史上下文导致了遗漏。

2026-11-21 SHA256 截断修复：实际运行 `UV_CACHE_DIR=/private/tmp/research-agent-uv-cache uv run pytest -q`，**148 passed in 4.34s**。新增 7 项离线用例覆盖模型草稿内截断 SHA256 的确定性替换、已注入历史时仍使用当前 CSV 证据、空证据区域生成完整报告、缺失/重复/倒序标记、无效草稿不得触达文件工具或写 Memory。验证实际工具参数与磁盘内容完全相同。原有错误证据测试改为注入组装故障后验证拒绝，未删除来源/统计完整性检查；截断草稿直接交给原验证器仍然失败。Skill、隔离、禁止覆盖和 Memory 回归继续通过。本次没有执行真实模型重测；已有失败报告不会被修补或覆盖，重测请指定新文件名。

Main 最小 Planning 增量：实际运行 `UV_CACHE_DIR=/private/tmp/research-agent-uv-cache uv run pytest -q`，**154 passed in 4.82s**。保留原 148 项测试，新增 6 项离线用例：无历史/有历史时真实计划创建和更新、官方工具实际更新图中 todos、报告写入失败时保持未完成且不写 Memory、清单全 completed 不能替代真实分析/报告、非法状态不会产生成功计划事件。测试驱动真实 DeepAgents 图和官方 Todo 工具，但模型是 Fake，没有调用付费 API，也没有执行本次 Planning 的远程模型验收。

2026-11-22 Summarization 验证：实际执行 `UV_CACHE_DIR=/private/tmp/research-agent-uv-cache uv run pytest -q`，**167 passed in 4.71s**，原 154 项回归保留，新增 13 项用例。自动触发、实际“摘要 + 近期消息”模型输入、原始消息状态、完整原始 task JSON 归档、多次摘要追加、独立 Backend 权限、归档失败拒绝验收均通过。联合 Fake 测试贯通 Planning + Summarization + Memory + Skills + Subagents：task 已不在模型上下文时，仍由真实工具记录生成完整报告并保存 Memory。测试禁止网络，未执行本次远程模型测试，不将 Fake 摘要质量等同于真实模型效果。

2026-11-22 摘要后重复写入/路径偏移修复：实际执行同一离线命令，**174 passed in 4.81s**，原 167 项保留并新增 7 项。模型从实际 System 程序事实读取目标，不依赖 Fake 缓存文件名；验证无历史/有历史时目标及成功状态在压缩后仍可见、成功后只提供计划工具、模型强行重复写入及改名不能生成额外产物、首次写错目标被拒绝、磁盘内容改变仍然失败、CLI 不展示模型虚构路径。最终位置由验收后确定性展示提供；原始 Main 结束回复保留为 model_summary，未被改写为假工具成功。本次未执行远程模型重测。

2026-11-22 Phase 2.6 HITL：实际运行 `UV_CACHE_DIR=/private/tmp/research-agent-uv-cache uv run pytest -q`，**191 passed in 5.21s**。原有 174 项回归全部保留，新增 17 项：真实 interrupt 前文件不存在、Command(resume) 批准后写入、人工 edit 后新路径及内容/Memory 一致、reject 经官方恢复后无文件无 Memory、危险/既有/符号链接路径拒绝、批准后创建文件的竞态仍不覆盖、终端 view/空输入、CLI edit/reject，以及 Planning + Summarization + Skills + Subagent + Memory 联合流程。分析失败或提前请求写入不会进入审批。模型和人工输入均为脚本化 Fake，使用的 HITL Middleware、Checkpointer、工具节点、文件操作和 resume 协议是真实实现；没有执行本阶段付费模型与真人终端验收。

仅支持本地单写者、顺序多进程运行；原始 CSV 与报告需保留。SHA256 完整性校验不等于密码学认证；自由文本解释没有程序语义判定，需人工审阅。真实模型如何使用相关历史仍需手动验证。

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
│   ├── memory.py
│   ├── report.py
│   ├── summarization.py
│   ├── hitl.py
│   ├── evaluation.py
│   ├── evaluation_model.py
│   └── tools/
│       ├── __init__.py
│       └── experiment.py
├── scripts/
│   ├── generate_sample.py
│   └── manual_integration.py
├── skills/experiment-analysis/SKILL.md
├── data/sample_experiments.csv
├── evals/cases.json
├── evaluation_runs/  # 自动生成的评估报告与隔离产物，不提交 Git
├── reports/.gitkeep
├── memory/research.json  # 验收后生成，不提交 Git
└── tests/
    ├── conftest.py
    ├── test_analysis.py
    ├── test_experiment_tool.py
    ├── test_agent_setup.py
    ├── test_skills.py
    ├── test_output_wording.py
    ├── test_submission.py
    ├── test_report_completion.py
    ├── test_report.py
    ├── test_planning.py
    ├── test_summarization.py
    ├── test_run_contract.py
    ├── test_hitl.py
    ├── test_evaluation.py
    ├── test_memory.py
    └── memory_worker.py
```

GitHub 仓库：[077cheerup/research-engineer-agent](https://github.com/077cheerup/research-engineer-agent)。按完成并验证的阶段提交和推送，密钥、运行报告与研究记忆不上传。
