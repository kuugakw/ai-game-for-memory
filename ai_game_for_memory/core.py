"""核心编排模块 (MemoryHub)。

MemoryHub 是插件的主入口，负责协调以下组件：
  - StagingManager（第一层：临时流水）
  - ArchivalManager（第二层：分类归档）
  - SnapshotManager（第三层：精确状态快照）
  - RAGEngine（知识按需检索）
  - WorkflowManager（工作流与决策审查）

同时管理上下文窗口有上界：
  - 跟踪当前上下文 token 用量
  - 接近上限时自动淘汰旧内容
  - 清除上下文前自动创建快照（防止失忆）
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

from .archival import ArchivalManager
from .config import Config
from .rag import RAGEngine
from .snapshot import SnapshotManager
from .staging import StagingManager
from .utils import estimate_tokens, format_timestamp
from .workflow import WorkflowManager

# 单条内容被截断时追加的标记
_TRUNCATION_MARKER = "…[truncated]"


class MemoryHub:
    """分层记忆架构核心编排器。

    集成五个子管理器，提供统一的 API 接口。
    上下文窗口有上界，记忆无上界。
    """

    def __init__(self, workspace: Optional[str] = None,
                 config: Optional[Config] = None):
        """初始化 MemoryHub。

        参数:
            workspace: 数据存储根目录路径。
            config: Config 配置实例；提供时以 config.workspace 为准。

        注意:
            同时传入 workspace 与 config 且两者不一致时会抛 ValueError，
            避免"以为改的是 workspace、实际用的是 config"的静默错误。
        """
        if config is not None:
            if not isinstance(config, Config):
                raise TypeError("config 必须是 Config 实例")
            if workspace is not None and os.path.abspath(workspace) != os.path.abspath(config.workspace):
                raise ValueError(
                    "workspace 与 config.workspace 冲突：{0!r} != {1!r}；"
                    "请只传其中一个".format(workspace, config.workspace)
                )
            workspace = config.workspace
            self.config = config
        else:
            workspace = workspace or "ai_game_for_memory_data"
            self.config = Config(workspace=workspace)

        self.workspace = workspace
        os.makedirs(self.workspace, exist_ok=True)

        # 初始化各子管理器
        self.staging = StagingManager(
            workspace=self.workspace,
            roll_hours=self.config.staging_roll_hours
        )
        self.archival = ArchivalManager(workspace=self.workspace)
        self.snapshot = SnapshotManager(workspace=self.workspace)
        self.rag = RAGEngine(
            workspace=self.workspace,
            top_k=self.config.rag_top_k
        )
        self.workflow = WorkflowManager(
            workspace=self.workspace,
            review_high_risk_only=self.config.review_high_risk_only
        )

        # 上下文窗口管理
        self.context_window: str = ""
        self.context_tokens: int = 0
        self.context_max_tokens: int = self.config.context_max_tokens

        # 持久化上下文状态
        self._context_file = os.path.join(self.workspace, "context_state.json")
        self._load_context_state()

    # ---- 上下文状态持久化 ----

    def _load_context_state(self):
        """加载之前保存的上下文状态（用于跨会话恢复）。"""
        if not os.path.exists(self._context_file):
            return
        try:
            with open(self._context_file, "r", encoding="utf-8") as f:
                state = json.load(f)
        except (IOError, OSError, json.JSONDecodeError) as e:
            print("[MemoryHub] 加载上下文状态失败: {0}".format(e))
            self.context_window = ""
            self.context_tokens = 0
            return

        window = state.get("context_window", "")
        self.context_window = window if isinstance(window, str) else ""
        # 以实际窗口内容重新估算，避免读到不一致的历史计数
        self.context_tokens = estimate_tokens(self.context_window)

    def _save_context_state(self):
        """持久化当前上下文状态。"""
        try:
            state = {
                "context_window": self.context_window,
                "context_tokens": self.context_tokens,
                "timestamp": format_timestamp(),
            }
            with open(self._context_file, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except (IOError, OSError) as e:
            print("[MemoryHub] 保存上下文状态失败: {0}".format(e))

    # ---- 上下文窗口管理 ----

    def _fit_entry(self, entry: str) -> str:
        """把单条内容裁剪到不超过上下文上限，保证 token 上界成立。

        单条内容本身大于窗口时，淘汰旧内容也没用，只能截断；
        截断会留下明确标记，避免"看起来完整、实际被悄悄丢掉"。
        计算时把写入时追加的换行符也算进去，保证 ``context_tokens`` 不越界。
        """
        max_tokens = self.context_max_tokens
        if estimate_tokens(entry + "\n") <= max_tokens:
            return entry

        if estimate_tokens(_TRUNCATION_MARKER + "\n") > max_tokens:
            # 窗口小到连标记都放不下：按字符硬截断
            return entry[:max(1, max_tokens)]

        low, high = 0, len(entry)
        while low < high:
            mid = (low + high + 1) // 2
            if estimate_tokens(entry[:mid] + _TRUNCATION_MARKER + "\n") <= max_tokens:
                low = mid
            else:
                high = mid - 1
        return entry[:low] + _TRUNCATION_MARKER

    def add_to_context(self, text: str, source: str = "") -> bool:
        """将文本添加到当前上下文窗口。

        接近上限时按 ``context_eviction_ratio`` 淘汰最旧内容；
        单条内容超过窗口上限时截断并标记。

        参数:
            text: 要添加到上下文的文本。
            source: 来源标识。

        返回:
            成功返回 True，文本为空返回 False。
        """
        if not text:
            return False
        if not isinstance(text, str):
            text = str(text)

        entry = self._fit_entry("[{0}] {1}".format(source, text))
        token_count = estimate_tokens(entry)

        # 循环淘汰，直到装得下（或无可淘汰）
        while (self.context_tokens + token_count > self.context_max_tokens
               and self.context_window):
            before = self.context_tokens
            self._evict_old_context()
            if self.context_tokens >= before:
                break

        self.context_window += entry + "\n"
        self.context_tokens = estimate_tokens(self.context_window)

        # token 估算是整数截断的，逐条相加可能与整段重算差 1；
        # 这里以整段重算为准，必要时再淘汰一轮，保证上界严格成立。
        while (self.context_tokens > self.context_max_tokens
               and self.context_window.count("\n") > 1):
            before = self.context_tokens
            self._evict_old_context()
            if self.context_tokens >= before:
                break

        self._save_context_state()
        return True

    def _evict_old_context(self):
        """淘汰旧上下文内容（当接近上限时）。

        按配置的 eviction ratio 淘汰最旧的行，随后用剩余内容重新估算
        token 数，保证计数与窗口内容永远一致。
        """
        if not self.context_window:
            self.context_tokens = 0
            return

        lines = [line for line in self.context_window.split("\n") if line]
        if not lines:
            self.context_window = ""
            self.context_tokens = 0
            self._save_context_state()
            return

        keep_ratio = 1.0 - self.config.context_eviction_ratio
        keep_count = max(1, int(len(lines) * keep_ratio))
        if keep_count >= len(lines):
            return  # 无可淘汰

        self.context_window = "\n".join(lines[-keep_count:]) + "\n"
        self.context_tokens = estimate_tokens(self.context_window)
        self._save_context_state()

    def clear_context(self, create_snapshot: Optional[bool] = None) -> int:
        """清除当前上下文窗口。

        如果配置了 snapshot_on_context_clear，清除前会自动创建快照。

        参数:
            create_snapshot: 是否创建快照。如果为 None，使用配置值。

        返回:
            被清除的 token 数量。
        """
        if create_snapshot is None:
            create_snapshot = self.config.snapshot_on_context_clear

        tokens_cleared = self.context_tokens

        if create_snapshot:
            self.create_snapshot(
                "before_context_clear",
                state={
                    "context_tokens": self.context_tokens,
                    "context_lines": self._context_line_count(),
                },
                include_context=True,
            )

        self.context_window = ""
        self.context_tokens = 0
        self._save_context_state()

        return tokens_cleared

    def _context_line_count(self) -> int:
        """当前上下文的有效行数。"""
        if not self.context_window:
            return 0
        return len([line for line in self.context_window.split("\n") if line])

    def get_context_info(self) -> Dict[str, Any]:
        """获取当前上下文窗口的信息。"""
        return {
            "current_tokens": self.context_tokens,
            "max_tokens": self.context_max_tokens,
            "usage_ratio": round(self.context_tokens / max(1, self.context_max_tokens), 4),
            "line_count": self._context_line_count(),
            "workspace": self.workspace,
        }

    # ---- 第一层：临时流水 API ----

    def stage_action(self, content: str, source: str = "",
                     verify_status: str = "pending",
                     risk_level: str = "low",
                     metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """便捷方法：追加一条动作流水到第一层。"""
        record = self.staging.stage_action(content, source=source,
                                           verify_status=verify_status,
                                           risk_level=risk_level,
                                           metadata=metadata)
        self.add_to_context("[ACTION] {0}".format(content), source=source)
        return record

    def stage_decision(self, content: str, source: str = "",
                       verify_status: str = "pending",
                       risk_level: str = "low",
                       metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """便捷方法：追加一条决策流水到第一层。"""
        record = self.staging.stage_decision(content, source=source,
                                             verify_status=verify_status,
                                             risk_level=risk_level,
                                             metadata=metadata)
        self.add_to_context("[DECISION] {0}".format(content), source=source)
        return record

    def stage_error(self, content: str, source: str = "",
                    verify_status: str = "fail",
                    risk_level: str = "medium",
                    metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """便捷方法：追加一条错误流水到第一层。"""
        record = self.staging.stage_error(content, source=source,
                                          verify_status=verify_status,
                                          risk_level=risk_level,
                                          metadata=metadata)
        self.add_to_context("[ERROR] {0}".format(content), source=source)
        return record

    def stage_screenshot(self, content: str, source: str = "",
                         verify_status: str = "pending",
                         risk_level: str = "low",
                         metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """便捷方法：追加一条截图摘要流水到第一层。"""
        record = self.staging.stage_screenshot(content, source=source,
                                               verify_status=verify_status,
                                               risk_level=risk_level,
                                               metadata=metadata)
        self.add_to_context("[SCREENSHOT] {0}".format(content), source=source)
        return record

    def stage_reasoning(self, content: str, source: str = "",
                        verify_status: str = "pending",
                        risk_level: str = "low",
                        metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """便捷方法：追加一条推理过程流水到第一层。"""
        record = self.staging.stage_reasoning(content, source=source,
                                              verify_status=verify_status,
                                              risk_level=risk_level,
                                              metadata=metadata)
        self.add_to_context("[REASONING] {0}".format(content), source=source)
        return record

    def stage_state(self, content: str, source: str = "",
                    verify_status: str = "pending",
                    risk_level: str = "low",
                    metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """便捷方法：追加一条状态流水到第一层。"""
        record = self.staging.stage_state(content, source=source,
                                          verify_status=verify_status,
                                          risk_level=risk_level,
                                          metadata=metadata)
        self.add_to_context("[STATE] {0}".format(content), source=source)
        return record

    def read_all_staging(self) -> List[Dict[str, Any]]:
        """读取所有临时流水记录。"""
        return self.staging.read_all()

    # ---- 第二层：分类归档 API ----

    def process_archival(self, category: Optional[str] = None) -> Dict[str, int]:
        """触发整理流程：将临时流水按语义路由到分类文件。

        参数:
            category: 可选的特定分类。如果提供，只把流水写入该分类。

        返回:
            统计字典，包含 archived, skipped, failed 数量。
        """
        if category is not None:
            return self.archival.process_staging(self.staging, category=category)
        return self.archival.process_all(self.staging)

    def archive_to_category(self, category: str,
                            record: Dict[str, Any]) -> bool:
        """将单条记录归档到指定分类。"""
        return self.archival.archive_entry(category, record)

    def get_archived_count(self, category: Optional[str] = None) -> int:
        """获取归档文件中的条目数。"""
        return self.archival.get_archived_count(category)

    def list_archival_categories(self) -> List[str]:
        """列出所有已归档的分类。"""
        return self.archival.list_categories()

    # ---- 第三层：快照 API ----

    def create_snapshot(self, name: str, state: Optional[Dict[str, Any]] = None,
                        include_context: bool = False,
                        metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """创建一个状态快照。

        参数:
            name: 快照名称。
            state: 需要精确恢复的状态；为 None 时只记录上下文统计信息。
            include_context: 是否把当前上下文原文一起存进快照。
            metadata: 可选附加信息。

        注意:
            上下文原文默认不写入 state，只有 include_context=True 时才单独保存，
            避免同一份上下文在快照文件里出现两遍。
        """
        if state is None:
            state = {
                "context_tokens": self.context_tokens,
                "context_lines": self._context_line_count(),
            }

        return self.snapshot.create_snapshot(
            name=name,
            state=state,
            include_context=include_context,
            context=self.context_window if include_context else None,
            metadata=metadata
        )

    def restore_snapshot(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """通过 snapshot_id（或快照名称）恢复状态。

        如果快照里保存了上下文原文，会一并恢复上下文窗口。
        兼容旧格式：state 里含 ``context_window`` 时也会用其恢复。
        """
        data = self.snapshot.load_snapshot(snapshot_id)
        if data is None:
            return None

        state = data.get("state")
        if not isinstance(state, dict):
            state = {}

        context = data.get("context")
        if not isinstance(context, str):
            legacy_context = state.get("context_window")
            context = legacy_context if isinstance(legacy_context, str) else None

        if context is not None:
            # 空字符串也要恢复：快照捕获的是"空上下文"，恢复就该清空
            self.context_window = context
            self.context_tokens = estimate_tokens(context)
            self._save_context_state()

        return state

    def list_snapshots(self) -> List[Dict[str, Any]]:
        """列出所有已保存的快照元信息。"""
        return self.snapshot.list_snapshots()

    def delete_snapshot(self, snapshot_id: str) -> bool:
        """删除指定快照。"""
        return self.snapshot.delete_snapshot(snapshot_id)

    # ---- RAG 检索 API ----

    def rag_search(self, query: str, top_k: Optional[int] = None,
                   category: Optional[str] = None,
                   mode: str = "hybrid") -> List[Dict[str, Any]]:
        """检索相关文档。"""
        return self.rag.search(query, top_k=top_k, category=category, mode=mode)

    def add_knowledge(self, doc_id: str, text: str, category: str,
                      vector: Optional[List[float]] = None) -> bool:
        """添加一条知识文档到 RAG 索引。"""
        return self.rag.add_document(doc_id, text, category, vector)

    def remove_knowledge(self, doc_id: str) -> bool:
        """从 RAG 索引中删除一条知识文档。"""
        return self.rag.remove_document(doc_id)

    def get_category_stats(self) -> Dict[str, int]:
        """获取各分类的文档数量统计。"""
        return self.rag.get_category_stats()

    def export_to_dataframe(self):
        """将所有 RAG 文档导出为 pandas DataFrame；未安装 pandas 时返回 None。"""
        try:
            import pandas as pd
        except ImportError:
            print("[MemoryHub] pandas 未安装，跳过 DataFrame 导出")
            return None

        rows = self.rag.all_documents()
        if rows:
            return pd.DataFrame(rows)
        return pd.DataFrame(columns=["doc_id", "category", "text", "token_count"])

    def export_rag_to_csv(self, filepath: str) -> bool:
        """将 RAG 文档导出为 CSV 文件。"""
        df = self.export_to_dataframe()
        if df is None:  # pandas 未安装（export_to_dataframe 已打印提示）
            return False

        try:
            df.to_csv(filepath, index=False, encoding="utf-8-sig")
            return True
        except (IOError, OSError) as e:
            print("[MemoryHub] 导出 CSV 失败 {0}: {1}".format(filepath, e))
            return False

    def export_staging_to_csv(self, filepath: str) -> bool:
        """将临时流水记录导出为 CSV 文件。"""
        try:
            import pandas as pd
        except ImportError:
            print("[MemoryHub] pandas 未安装，跳过 CSV 导出")
            return False

        records = self.staging.read_all()
        if records:
            df = pd.DataFrame(records)
        else:
            df = pd.DataFrame(columns=["dedup_id", "source_file", "entry_index",
                                       "timestamp", "coarse_type", "content",
                                       "source", "verify_status", "risk_level",
                                       "token_estimate"])

        try:
            df.to_csv(filepath, index=False, encoding="utf-8-sig")
            return True
        except (IOError, OSError) as e:
            print("[MemoryHub] 导出 CSV 失败 {0}: {1}".format(filepath, e))
            return False

    # ---- 工作流与决策 API ----

    def execute_decision(self, decision_text: str, risk_level: str,
                         thesis_reason: Optional[str] = None,
                         antithesis_reason: Optional[str] = None) -> Dict[str, Any]:
        """执行一个决策（带正反合审查或自动放行）。"""
        return self.workflow.execute_decision(
            decision_text=decision_text,
            risk_level=risk_level,
            thesis_reason=thesis_reason,
            antithesis_reason=antithesis_reason
        )

    def should_review(self, risk_level: str) -> bool:
        """判断是否需要对某个风险等级的决策启动审查。"""
        return self.workflow.should_review(risk_level)

    def create_decision_review(self, decision_text: str,
                               risk_level: str = "low"):
        """创建一个新的决策审查记录。"""
        return self.workflow.create_decision_review(decision_text, risk_level)

    # ---- 行为三态 API ----

    def register_behavior(self, behavior_id: str, pattern: str,
                          description: str, status: str = "in_progress") -> Dict[str, Any]:
        """注册行为模式（三态：done / in_progress / retired）。"""
        return self.workflow.register_behavior(behavior_id, pattern, description, status)

    def retire_behavior(self, behavior_id: str) -> bool:
        """将某个行为标记为 retired（退场）。"""
        return self.workflow.retire_behavior(behavior_id)

    def get_behavior(self, behavior_id: str) -> Optional[Dict[str, Any]]:
        """获取某个行为的信息。"""
        return self.workflow.get_behavior(behavior_id)

    def list_behaviors(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出所有行为记录。"""
        return self.workflow.list_behaviors(status)

    def complete_behavior(self, behavior_id: str) -> bool:
        """将某个行为标记为 done（已完成）。"""
        return self.workflow.complete_behavior(behavior_id)

    def activate_behavior(self, behavior_id: str) -> bool:
        """将某个行为标记为 in_progress（使用中）。"""
        return self.workflow.activate_behavior(behavior_id)

    # ---- 综合整理与清理 ----

    def full_cleanup(self) -> Dict[str, Any]:
        """执行完整整理流程：归档 + 清理过期临时文件 + 统计。

        注意：如果本次归档存在失败条目，会跳过过期文件清理，
        避免"归档没成功、源文件先被删掉"导致数据丢失。
        """
        archival_stats = self.process_archival()
        expired_count = 0
        if archival_stats.get("failed", 0) == 0:
            expired_count = self._cleanup_expired_staging()
        else:
            print("[MemoryHub] 归档存在失败条目，本次跳过过期文件清理")

        return {
            "archival": archival_stats,
            "expired_staging_removed": expired_count,
            "archival_categories": self.list_archival_categories(),
            "rag_category_stats": self.get_category_stats(),
            "context_info": self.get_context_info(),
            "snapshots": len(self.list_snapshots()),
        }

    def _cleanup_expired_staging(self) -> int:
        """清理超过保留期的临时流水文件（跳过当前正在写的文件）。"""
        if not os.path.exists(self.staging.staging_dir):
            return 0

        now = time.time()
        retention_seconds = self.config.retention_days * 24 * 3600
        count = 0
        active = self.staging.current_file_path
        active_abs = os.path.abspath(active) if active else None

        for filename in os.listdir(self.staging.staging_dir):
            if not filename.endswith(".jsonl"):
                continue
            file_path = os.path.join(self.staging.staging_dir, filename)
            if active_abs is not None and os.path.abspath(file_path) == active_abs:
                continue  # 正在写的文件删不掉（Windows）也不该删
            try:
                if now - os.path.getmtime(file_path) > retention_seconds:
                    os.remove(file_path)
                    count += 1
            except (OSError, IOError):
                pass
        return count

    def clear_all(self) -> int:
        """清空所有数据（staging / archival / snapshots / rag / workflow）。

        返回:
            删除的文件数量。
        """
        self.clear_context(create_snapshot=False)

        # 先关闭并重置 staging，否则 Windows 上正在打开的文件无法删除
        self.staging.reset()
        staging_count = self._clear_staging()
        archival_count = self.archival.clear_all()
        snapshot_count = self.snapshot.clear_all()
        rag_count = self.rag.clear_all()
        workflow_count = self.workflow.clear_all()

        return staging_count + archival_count + snapshot_count + rag_count + workflow_count

    def _clear_staging(self) -> int:
        """清空临时流水文件。"""
        if not os.path.exists(self.staging.staging_dir):
            return 0
        count = 0
        for filename in os.listdir(self.staging.staging_dir):
            if filename.endswith(".jsonl"):
                try:
                    os.remove(os.path.join(self.staging.staging_dir, filename))
                    count += 1
                except (IOError, OSError):
                    pass
        return count

    # ---- 生命周期 ----

    def close(self):
        """关闭文件句柄并保存上下文状态。"""
        self.staging.close()
        self._save_context_state()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
