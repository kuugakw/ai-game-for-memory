# AI Game for Memory

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![tests](https://github.com/kuugakw/ai-game-for-memory/actions/workflows/tests.yml/badge.svg)](https://github.com/kuugakw/ai-game-for-memory/actions/workflows/tests.yml)
[![Status](https://img.shields.io/badge/Status-prototype%20%2F%20archived-lightgrey.svg)]()

> 一套给**本地 / 离线 AI** 用的分层记忆骨架：上下文有上界，记忆无上界；状态用快照精确恢复，
> 知识用 RAG 按需检索；高风险决策走正反合审查。
>
> 纯 Python 标准库实现，复制目录就能用，不联网也能跑。**这是一个原型，不是生产级库。**

**仓库**：<https://github.com/kuugakw/ai-game-for-memory> · **Python 包**：`ai-game-for-memory` · **导入名**：`ai_game_for_memory`

---

## 这个仓库是什么

当时在做一个单机游戏，需要一个本地 AI 记住玩家的行为、任务状态、世界事实，又不能让上下文
无限膨胀，还得在完全离线的环境里跑——不能调云 API、不能依赖外部向量库，最好纯标准库就能落地。
于是就有了这套三层记忆结构。

后来游戏方向变了，框架没用上。代码一直躺在硬盘里，想了想还是开源出来：

- **它是**：一套可以直接读、直接改的分层记忆设计参考 + 可运行的纯标准库实现；
- **它不是**：成熟的记忆中间件、生产级向量数据库、经过大规模验证的 Agent 框架；
- **没有**：并发写保护、真正的语义 embedding、事务性持久化、持续维护承诺。

如果你的本地 AI / 离线 Agent / 单机 NPC 也在头疼"上下文有上界、记忆无上界"这件事，
里面的结构也许能给你一点参考。**欢迎 fork 改造成你自己的东西。**

---

## 它想解决的四个问题

| 问题 | 本项目的取向 |
|------|-------------|
| **上下文会爆**：多步任务、截图、检索结果迅速占满窗口 | 上下文窗口有上界，接近上限自动淘汰最旧内容 |
| **单模型决策风险**：一个模型既理解又执行，容易漏约束、在不可逆操作上出错 | 高风险决策走正反合审查，低风险直接放行 |
| **知识混在一起检索不准**：所有记忆塞进一个大索引，版本之间互相干扰 | 三层记忆分离，RAG 按分类按需检索 |
| **一清上下文就失忆**：全清重建成本高，且容易丢状态 | 快照保存全量状态，恢复时直接替换，不靠语义检索碰运气 |

### 设计原则

- **上下文有上界**：窗口有 token 上限，超限时按比例淘汰最旧内容；单条超大内容截断并标记
- **记忆无上界**：三层记忆都落在文件系统（JSON / JSONL），不怕数据变多
- **状态精确恢复**：快照是全量 dump，恢复时直接替换，不做模糊匹配
- **知识按需检索**：RAG 只检索知识分类，避免行为流水和知识互相污染
- **行为可复用可退场**：行为三态 `done / in_progress / retired`
- **高风险决策审查**：正反合（thesis-antithesis-synthesis）流程，权重可计算、可审计
- **宁可要"信息不足"，不要编理由**：缺少正反面理由时返回 `needs_more_info`，不替调用方拍板

---

## 架构

```text
                     ┌──────────────────── MemoryHub ────────────────────┐
                     │                                                   │
  每步动作/截图/决策 │  ① 临时流水 Staging       ② 分类归档 Archival     │
  错误/推理/状态 ────┼──▶ 按小时滚动 .jsonl  ──▶ 按语义路由到分类 .jsonl  │
                     │        (快、只追加)          (幂等、可重放)        │
                     │                                                   │
  不可逆操作前 ──────┼──▶ ③ 快照 Snapshot  全量 JSON dump，精确恢复      │
                     │                                                   │
  知识文档 ──────────┼──▶ RAG 检索  BM25 + 本地哈希向量（按分类过滤）   │
                     │                                                   │
  需要做的决策 ──────┼──▶ 决策审查  正反合 + 权重（高/中风险）          │
                     │                                                   │
  上下文 ────────────┼──▶ 上下文窗口  token 上界 + 自动淘汰 + 清空前快照 │
                     └───────────────────────────────────────────────────┘
```

数据流是单向的：**先写流水（快）→ 再整理归档（可重放）→ 状态用快照兜底**。
RAG 只负责"知识"，快照只负责"状态"，两者刻意分开。

---

## 快速开始

### 安装

从 GitHub 获取源码：

```bash
git clone https://github.com/kuugakw/ai-game-for-memory.git
cd ai-game-for-memory
```

方式一：直接复制包目录到你的项目里（零依赖，最快）：

```bash
cp -r ai_game_for_memory ./your_project/
```

方式二：用 pip 从本地源码安装：

```bash
pip install .            # 可选依赖：pip install ".[perf]"
```

方式三：直接从 GitHub 安装（需要 git）：

```bash
pip install "git+https://github.com/kuugakw/ai-game-for-memory.git"
```

要求 Python 3.8+，没有必装第三方依赖。

### 30 秒上手

```python
from ai_game_for_memory import MemoryHub

hub = MemoryHub(workspace="my_agent_memory")

# ① 记流水（快、只追加）
hub.stage_action("点击设置菜单", source="gui", risk_level="low")
hub.stage_error("保存超时", source="gui_api", risk_level="medium")

# ② 整理归档（按粗分类自动路由，可重复调用，幂等）
print(hub.process_archival())     # {'archived': 2, 'skipped': 0, 'failed': 0}

# ③ 不可逆操作前打快照
hub.create_snapshot("before_delete", state={"files": ["a.txt"]}, include_context=True)

# 知识检索
hub.add_knowledge("err_timeout_001",
                  "API 超时时先重试 3 次，记录错误日志，仍失败则通知用户",
                  category="errors_fixes")
for hit in hub.rag_search("超时怎么处理", top_k=3):
    print(hit["score"], hit["category"], hit["text"])

# 高风险决策：正反合审查
result = hub.execute_decision(
    decision_text="删除所有临时文件",
    risk_level="high",
    thesis_reason="释放磁盘空间，避免磁盘满载",
    antithesis_reason="可能丢失未归档的流水，需要先确认整理与快照已完成",
)
print(result["outcome"], result["approved"], result["weights"])

# 清上下文（默认会先自动打快照）
hub.clear_context()
hub.close()
```

完整的可运行示例见 [`examples/demo.py`](examples/demo.py)：

```bash
python examples/demo.py
```

---

## 三层记忆

### 第一层：临时流水（Staging）

每步动作、截图摘要、决策、错误、推理、状态都原样追加到**按小时滚动**的 JSONL 文件。

- **写得快**：写入时不做复杂语义判断，只带粗分类、来源、验证状态、风险等级
- **崩溃安全**：一行一条 JSON，写入后 flush；读取时跳过解析失败的坏行
- **重启安全**：重启后从已有文件恢复 `entry_index`，新记录的 `dedup_id` 不会和旧记录冲突
- **半行自动修复**：上次崩溃留下没有换行符的半行时，重新打开会先补一个换行，
  避免新记录和半行拼成一行、把两条记录同时弄坏
- **路径无关**：`dedup_id` 基于"相对路径 + 序号"，整个数据目录搬走也不会重复归档

```python
from ai_game_for_memory import MemoryHub

hub = MemoryHub(workspace="my_agent_memory")
hub.stage_action("点击设置菜单", source="gui", risk_level="low")
hub.stage_decision("关闭设置", source="gui", risk_level="low")
hub.stage_error("保存超时", source="gui_api", risk_level="medium")
hub.stage_screenshot("设置页三个选项卡", source="gui", risk_level="low")
hub.stage_reasoning("超时后应重试 3 次", source="agent")
hub.stage_state("当前阶段：配置管理", source="agent")

for record in hub.read_all_staging():
    print(record["timestamp"], record["coarse_type"], record["content"])
```

### 第二层：分类归档（Archival）

把流水按粗分类路由到不同分类文件。粗分类到目标分类的映射就是 `ArchivalManager.CATEGORY_MAP`：

| 粗分类 (`coarse_type`) | 归档分类 (`category`) | 内容 |
|---|---|---|
| `action` | `action_logs` | 动作流水 |
| `screenshot` | `ui_facts` | 界面事实 / 截图摘要 |
| `decision` | `decisions_reasoning` | 决策与理由 |
| `reasoning` | `decisions_reasoning` | 推理过程 |
| `error` | `errors_fixes` | 错误与修正 |
| `state` | `behavior_tristate` | 状态 / 行为三态 |
| `unknown` | `action_logs` | 未分类兜底 |

`ui_maps` 与 `snapshots` 是保留分类，可以手动调用 `archive_to_category(category, record)`
把界面地图、快照引用写进去。注意 `record` 必须带 `dedup_id`（流水记录本来就有；
手动写入时自己指定一个稳定 ID 即可）。

**幂等与可重放**：

- 每条流水有唯一 `dedup_id`（来源文件 + 序号哈希），归档前检查目标分类里是否已存在
- 每批整理在目标分类文件里写一行 `START_MARKER`（批次 ID、时间、来源文件），用于审计；
  真正的防重复靠 `dedup_id`
- 崩溃/中途失败后，再调用一次 `process_archival()` 就会补齐缺的部分，不会写重复条目
- 没有待归档内容时直接返回，不会往分类文件里灌无意义的 marker

```python
# 自动路由
stats = hub.process_archival()
# {'archived': 15, 'skipped': 3, 'failed': 0}

# 只把待归档流水写进某个分类（例如手动补 action_logs）
stats = hub.process_archival(category="action_logs")

# 看看有哪些分类
print(hub.list_archival_categories())
print(hub.get_archived_count("errors_fixes"))
```

> `full_cleanup()` 会先归档、再清理过期流水文件；如果本次归档有失败条目，
> 会跳过清理，避免"归档没成功、源文件先被删掉"。

### 第三层：精确状态快照（Snapshot）

快照是**全量状态 dump**，用于精确恢复，不依赖语义检索：

- `state` 由调用方给出，原样保存、原样返回
- `include_context=True` 时额外保存上下文原文，恢复时一并恢复上下文窗口
- 快照名会脱敏后用于文件名（`../x`、`a/b` 之类不会越界），原始名称保留在内容里
- 同一秒内创建同名快照会自动加序号，不覆盖已有文件

```python
info = hub.create_snapshot(
    "before_delete_all_files",
    state={"files_to_delete": ["a.txt", "b.txt"], "step": 3},
    include_context=True,
)
print(info["snapshot_id"])            # before_delete_all_files_20261112_190124

# 按 snapshot_id 或快照名恢复
state = hub.restore_snapshot("before_delete_all_files")
print(state["files_to_delete"])

# 列出、删除
print(hub.list_snapshots())
hub.delete_snapshot(info["snapshot_id"])
```

> 快照与 RAG 的分工：**状态**必须精确，走快照；**知识**可以模糊，走 RAG。

---

## RAG 检索（本地实现）

纯标准库的本地检索：BM25 关键词匹配 + hashing trick 向量，二者混合。

### 索引与一致性

- 文档 JSON（`rag_index/documents/*.json`）是**唯一事实来源**
- BM25 词项索引每次启动时由文档重建（不单独持久化），避免"索引文件和文档不同步"
- 向量随文档一起持久化，重启后直接复用
- 同一 `doc_id` 重复添加 = 覆盖，旧词项会被摘掉，不会残留幽灵命中
- `doc_id` 会被脱敏成文件名（`ui/settings` 这类 ID 不会写到目录外）

### 向量实现

- 用 `zlib.crc32` 稳定哈希把每个 token 映射到 128 维向量的某一维
  （不受 `PYTHONHASHSEED` 影响，跨进程可复现）
- 词频累加后做 L2 归一化，query 与 document 共享同一向量空间
- 可以 `add_knowledge(..., vector=[...])` 传入外部 embedding 覆盖本地向量

> 说明：这是为了让"离线也能跑"做的简化实现，本质是**词袋哈希向量，不是语义 embedding**。
> 如果你的本地 AI 已经带了 embedding 模型，建议通过 `vector` 参数传入真向量替换本地实现。
> 注意：外部传入的向量必须彼此同维度，几何上的相似度才有意义；混用不同维度会让评分失真
> （代码会按较短长度截断，不会报错，所以务必自己对齐）。

### 检索模式

| 模式 | 说明 |
|------|------|
| `"bm25"` | 仅关键词匹配（BM25，k1=1.5，b=0.75） |
| `"vector"` | 仅向量余弦相似度（本地哈希向量） |
| `"hybrid"` | 混合检索：0.6 × 归一化 BM25 + 0.4 × 余弦相似度（默认） |

返回结果会按分数降序排列，并过滤掉分数为 0（完全没命中）的文档。
每条结果包含 `doc_id / score / bm25_score / vector_score / category / text`。

```python
# 只检索错误与修正分类
hits = hub.rag_search("超时处理", category="errors_fixes", top_k=3)

# 三种模式
hits = hub.rag_search("如何处理超时", mode="hybrid", top_k=5)
hits = hub.rag_search("如何处理超时", mode="bm25")
hits = hub.rag_search("如何处理超时", mode="vector")

# 删除知识
hub.remove_knowledge("err_timeout_001")
```

---

## 决策审查：正反合

### 流程

1. **正（Thesis）**：支持该决策的理由
2. **反（Antithesis）**：反对该决策的理由
3. **合（Synthesis）**：综合正反面得出的结论（自动生成并留痕）

### 权重

```text
thesis_weight      = len(thesis)      / (len(thesis) + len(antithesis))
antithesis_weight  = len(antithesis)  / (len(thesis) + len(antithesis))
risk_adjusted_score = antithesis_weight × risk_multiplier
risk_multiplier: low=1.0, medium=1.2, high=1.5
```

`risk_adjusted_score` 表示**风险调整后的反对强度**，越高越危险。

### 审查规则

| 风险等级 | 规则 |
|---|---|
| `low` | 直接放行，不浪费 token（`reviewed=False`） |
| `medium` | 默认放行；`review_high_risk_only=False` 时审查，`antithesis_weight ≤ 0.5` 批准，否则拒绝 |
| `high` | 必须审查，`risk_adjusted_score ≤ 0.3` 才批准（即反对理由占比要低于约 20%） |
| 缺理由 | 需要审查但没给正/反理由 → `needs_more_info`，不批准也不编默认理由 |

### 示例（真实运行结果）

```python
result = hub.execute_decision(
    decision_text="删除所有临时文件",
    risk_level="high",
    thesis_reason="释放磁盘空间，避免磁盘满载，当前磁盘使用率已达 90%",
    antithesis_reason="可能会丢失未归档的流水记录，需要确认整理已完成且快照已创建",
)

result["outcome"]      # 'rejected'
result["approved"]     # False
result["weights"]
# {
#   'thesis_weight': 0.4821, 'antithesis_weight': 0.5179,
#   'risk_adjusted_score': 0.7768, 'risk_multiplier': 1.5,
#   'risk_level': 'high', 'thesis_length': 27, 'antithesis_length': 29,
#   'threshold': 0.3,
# }
result["synthesis"]
# '综合正反面理由：rejected（risk_adjusted_score=0.7768 / 阈值 0.3）'
```

`risk_adjusted_score = 0.5179 × 1.5 = 0.7768 > 0.3`，所以拒绝——**反对理由足够强时，高风险决策不会放行**。

> 注意：这里的权重只看理由文本长度，非常粗糙，只够做个示意。真要用于生产，
> 应换成模型给出的结构化置信度，或至少按语义强度打分。

---

## 行为三态

| 状态 | 说明 |
|---|---|
| `in_progress` | 正在使用的行为模式 |
| `done` | 已完成的行为模式 |
| `retired` | 已退场、不可再用的行为模式 |

```python
hub.register_behavior(
    behavior_id="handle_timeout",
    pattern="重试 3 次 → 记录错误 → 通知用户",
    description="处理 API 超时的标准行为模式",
    status="in_progress",
)

hub.complete_behavior("handle_timeout")   # in_progress → done
hub.retire_behavior("handle_timeout")     # → retired
hub.activate_behavior("handle_timeout")   # 重新启用 → in_progress

print(hub.list_behaviors(status="in_progress"))
```

重新注册同一个 `behavior_id` 会更新 `pattern / description / status`（不是只改状态），
状态切换会追加一行 `workflow/transitions.log` 作为审计记录。

---

## 上下文窗口

- `context_max_tokens` 是**估算**上限（中文约 1.5 字符/token，英文约 4 字符/token）
- 超过上限时按 `context_eviction_ratio` 淘汰最旧的行
- 单条内容本身就超过上限时截断，并追加 `…[truncated]` 标记，保证上界严格成立
- 清除上下文前默认先创建 `before_context_clear` 快照（`snapshot_on_context_clear`）
- 上下文状态写入 `context_state.json`，进程重启后可恢复

```python
hub.add_to_context("用户询问如何处理超时", source="user")
print(hub.get_context_info())
# {'current_tokens': 8, 'max_tokens': 128000, 'usage_ratio': 0.0001,
#  'line_count': 1, 'workspace': 'my_agent_memory'}

hub.clear_context()   # 返回被清掉的 token 数，并自动打快照
```

---

## API 参考

### MemoryHub

| 方法 | 说明 |
|------|------|
| `stage_action(content, source, verify_status, risk_level, metadata)` | 追加动作流水 |
| `stage_decision(...)` | 追加决策流水 |
| `stage_error(...)` | 追加错误流水（默认 `verify_status="fail"`, `risk_level="medium"`） |
| `stage_screenshot(...)` | 追加截图摘要流水 |
| `stage_reasoning(...)` | 追加推理过程流水 |
| `stage_state(...)` | 追加状态流水 |
| `read_all_staging()` | 读取所有流水记录 |
| `process_archival(category=None)` | 触发整理，按粗分类路由到分类文件 |
| `archive_to_category(category, record)` | 手动把单条记录写入指定分类 |
| `get_archived_count(category=None)` | 归档条目数 |
| `list_archival_categories()` | 已归档的分类列表 |
| `create_snapshot(name, state, include_context, metadata)` | 创建快照 |
| `restore_snapshot(snapshot_id)` | 按 ID 或名称恢复快照，返回 state；有上下文则一并恢复 |
| `list_snapshots()` | 列出快照元信息 |
| `delete_snapshot(snapshot_id)` | 删除快照 |
| `rag_search(query, top_k, category, mode)` | 检索（`bm25` / `vector` / `hybrid`） |
| `add_knowledge(doc_id, text, category, vector)` | 添加/覆盖知识文档 |
| `remove_knowledge(doc_id)` | 删除知识文档 |
| `get_category_stats()` | 各分类文档数 |
| `export_to_dataframe()` / `export_rag_to_csv(path)` / `export_staging_to_csv(path)` | pandas 导出（可选依赖） |
| `execute_decision(decision_text, risk_level, thesis_reason, antithesis_reason)` | 决策审查或自动放行 |
| `should_review(risk_level)` | 当前配置下是否需要审查 |
| `create_decision_review(decision_text, risk_level)` | 手动创建一条审查记录 |
| `register_behavior(behavior_id, pattern, description, status)` | 注册/更新行为模式 |
| `activate_behavior(id)` / `complete_behavior(id)` / `retire_behavior(id)` | 切换行为状态 |
| `get_behavior(id)` / `list_behaviors(status)` | 查询行为 |
| `add_to_context(text, source)` | 往上下文窗口追加内容 |
| `get_context_info()` | 上下文窗口信息 |
| `clear_context(create_snapshot=None)` | 清空上下文（默认先打快照） |
| `full_cleanup()` | 归档 + 清理过期文件 + 统计 |
| `clear_all()` | 清空所有数据 |
| `close()` | 关闭文件句柄并保存状态（也支持 `with MemoryHub(...) as hub:`） |

### Config

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `workspace` | `str` | `"ai_game_for_memory_data"` | 数据根目录 |
| `context_max_tokens` | `int` | `128000` | 上下文 token 估算上限 |
| `context_eviction_ratio` | `float` | `0.5` | 超限时淘汰比例，`(0, 1]` |
| `staging_roll_hours` | `int` | `1` | 流水滚动小时数，`1~24` |
| `archival_max_file_size_mb` | `float` | `50.0` | 保留字段（当前实现不按大小切分文件） |
| `rag_top_k` | `int` | `5` | 检索默认条数 |
| `review_high_risk_only` | `bool` | `True` | 是否只审查高风险决策 |
| `snapshot_on_irreversible` | `bool` | `True` | 保留字段，供调用方在不可逆操作前自行判断 |
| `snapshot_on_context_clear` | `bool` | `True` | 清上下文前是否自动打快照 |
| `retention_days` | `int` | `7` | 流水文件保留天数 |

### 枚举

| 枚举 | 值 |
|---|---|
| `RiskLevel` | `low` / `medium` / `high` |
| `VerifyStatus` | `pending` / `pass` / `fail` / `skip` |
| `CoarseType` | `action` / `screenshot` / `decision` / `error` / `reasoning` / `state` / `unknown` |
| `ArchiveCategory` | `ui_facts` / `action_logs` / `decisions_reasoning` / `ui_maps` / `errors_fixes` / `behavior_tristate` / `snapshots` |
| `ReviewOutcome` | `approved` / `rejected` / `needs_more_info` |

以上枚举成员既可以传枚举本身，也可以直接传对应的字符串值
（例如 `risk_level=RiskLevel.HIGH` 与 `risk_level="high"` 等价）。

---

## 数据目录结构

```text
my_agent_memory/
├── context_state.json                 # 上下文窗口与 token 计数
├── staging/
│   └── staging_20261112T1900.jsonl    # 第一层：按小时滚动的流水
├── archival/
│   ├── action_logs.jsonl              # 第二层：分类归档
│   ├── errors_fixes.jsonl
│   ├── decisions_reasoning.jsonl
│   └── ...
├── snapshots/
│   └── before_delete_20261112_190124.json   # 第三层：全量状态快照
├── rag_index/
│   └── documents/
│       └── err_timeout_001.json       # 知识文档（含向量）
└── workflow/
    ├── behaviors/<behavior_id>.json   # 行为三态
    ├── reviews/decision_<id>.json     # 决策审查记录
    └── transitions.log                # 状态切换审计日志
```

---

## 设计取舍与已知限制

这些是**有意为之的取舍**，不是待修的 bug：

1. **权重只看长度**：决策审查的权重基于理由文本长度，粗糙，只适合做示意。
2. **向量不是语义 embedding**：hashing trick 词袋向量，同义改写可能检索不到。
3. **没有并发写保护**：同一个 workspace 只应由单个写者使用；多进程同时写会互相覆盖。
4. **没有事务/原子写**：崩溃可能留下半行坏记录（读取时会跳过），
   但不会做 WAL / fsync 级别的保证。
5. **token 是估算值**：不是 tokenizer，只用于相对控制上下文规模。
6. **文件不按大小切分**：`archival_max_file_size_mb` 目前是保留字段。
7. **没有实现自动触发**：跨小时 / 文件过大 / 阶段结束等触发点需要调用方自己调用
   `process_archival()` / `full_cleanup()`。
8. **原型状态**：代码写过、demo 和测试跑通过，但没有在生产环境或大规模数据上验证过。

## 如果你要拿去做正经项目

建议先做这几件事：

1. 加文件锁 / 单写者约束（本地多进程写同一 workspace 会出问题）
2. 把 RAG 向量换成真实 embedding（本地小模型即可），并给 `add_knowledge` 传 `vector`
3. 按自己的场景重写 `CATEGORY_MAP` 和行为三态定义
4. 把 `context_max_tokens` 调到贴合你本地模型窗口的值，并调高 `context_eviction_ratio`
5. 决策审查接模型输出的结构化置信度，替换"按长度算权重"
6. 需要断点续跑的话，把 `process_archival` 挂到阶段结束、不可逆操作前等时机

---

## 项目结构

```text
ai-game-for-memory/
├── ai_game_for_memory/
│   ├── __init__.py       # 包初始化与公共 API 导出
│   ├── config.py         # 配置项与枚举
│   ├── utils.py          # 去重 ID、时间戳、token 估算、文件名脱敏
│   ├── staging.py        # 第一层：临时流水（按小时滚动、崩溃安全）
│   ├── archival.py       # 第二层：分类归档（语义路由、幂等、可重放）
│   ├── snapshot.py       # 第三层：精确状态快照
│   ├── rag.py            # RAG 检索（BM25 + 本地哈希向量）
│   ├── workflow.py       # 决策审查（正反合）与行为三态
│   └── core.py           # 编排器 MemoryHub（含上下文窗口管理）
├── examples/
│   └── demo.py           # 端到端示例（临时 workspace，跑完自动清理）
├── tests/
│   └── test_ai_game_for_memory.py   # 46 个单元测试，不依赖 pytest 也能跑
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## 测试

```bash
# 不装 pytest 也能跑（内置 runner）
python tests/test_ai_game_for_memory.py

# 或者用 pytest
pip install pytest
python -m pytest tests -q
```

测试覆盖：去重 ID 与时间戳、配置校验、流水的崩溃/重启恢复、归档幂等与路由、
快照脱敏与恢复、RAG 重建/覆盖/非法参数、正反合三种结果、上下文淘汰与截断、
`clear_all`（Windows 文件锁）、上下文跨实例恢复等。

## 可选依赖

| 模块 | 用途 | 不装时 |
|---|---|---|
| `numpy` | 余弦相似度向量化加速 | 回退纯 Python 实现 |
| `jieba` | 中文分词（BM25 / 向量） | 回退"英文按词、中文按字"切分 |
| `pandas` | `export_to_dataframe` / CSV 导出 | 导出方法返回 `None` / `False` 并打印提示 |

```bash
pip install ".[perf]"      # 或 pip install numpy jieba pandas
```

## License

MIT，见 [LICENSE](LICENSE)。随便拿去改（见 License 条款）。
