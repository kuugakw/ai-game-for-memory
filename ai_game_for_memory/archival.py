"""第二层记忆：分类归档 (Archival)。

设计要点：
  - 跨小时、文件过大、阶段结束、清除上下文前、不可逆操作前触发整理
  - 整理把临时流水按语义路由到分类文件：界面事实、动作流水、
    决策与理由、界面地图、错误与修正、行为三态、快照
  - 幂等、可恢复：按"来源文件 + 序号"去重，崩溃后能重放未完成的整理
  - 旧临时文件不删（由 retention_days 配置定期清理）

幂等性保证：
  - 每个流水条目有唯一的 dedup_id（来源文件 + 序号 哈希生成）
  - 归档时检查目标分类文件是否已存在该 dedup_id
  - 重复处理同一批流水不会导致重复归档

崩溃恢复：
  - 每个归档批次在目标分类文件里写一行 START_MARKER，记录批次 ID、时间、来源文件
    （审计用途：能看到每批整理的范围；真正防重复靠 dedup_id）
  - 崩溃后重新调用 process_archival() 即可补齐未归档的条目
"""

import json
import os
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

from .config import ArchiveCategory, CoarseType, coerce_enum
from .staging import StagingManager
from .utils import format_timestamp


class ArchivalManager:
    """分类归档管理器。

    负责第二层记忆：将临时流水按语义路由到分类文件。
    幂等、可恢复，支持崩溃重放。
    """

    # 粗分类到分类归档的映射规则
    CATEGORY_MAP = {
        CoarseType.ACTION.value: ArchiveCategory.ACTION_LOGS.value,
        CoarseType.DECISION.value: ArchiveCategory.DECISIONS_REASONING.value,
        CoarseType.ERROR.value: ArchiveCategory.ERRORS_FIXES.value,
        CoarseType.SCREENSHOT.value: ArchiveCategory.UI_FACTS.value,
        CoarseType.REASONING.value: ArchiveCategory.DECISIONS_REASONING.value,
        CoarseType.STATE.value: ArchiveCategory.BEHAVIOR_TRISTATE.value,
        CoarseType.UNKNOWN.value: ArchiveCategory.ACTION_LOGS.value,
    }

    def __init__(self, workspace: str):
        """初始化 ArchivalManager。

        参数:
            workspace: 数据存储根目录路径。
        """
        self.workspace = workspace
        self.archival_dir = os.path.join(workspace, "archival")
        os.makedirs(self.archival_dir, exist_ok=True)

        # 记录每个分类文件的已处理 dedup_id 集合（内存缓存）
        self._processed_ids: Dict[str, Set[str]] = {}
        # 当前批次已写入的起点标记（仅用于本次调用，结束后清空）
        self._start_markers: List[Dict[str, Any]] = []

    # ---- 分类与文件 ----

    @staticmethod
    def _normalize_category(category: str) -> str:
        """校验并返回规范化的分类名（接受 ArchiveCategory 枚举或字符串）。"""
        category = coerce_enum(category)
        valid = ArchiveCategory.values()
        if category not in valid:
            raise ValueError(
                "category '{0}' 不是有效分类，可选值：{1}".format(category, valid)
            )
        return category

    def _get_category_file(self, category: str) -> str:
        """获取指定分类的归档文件路径。"""
        category = self._normalize_category(category)
        return os.path.join(self.archival_dir, "{0}.jsonl".format(category))

    # ---- 已处理集合 ----

    def _load_processed_ids(self, category: str) -> Set[str]:
        """加载指定分类文件已处理的 dedup_id 集合。

        支持崩溃重放：重新加载时能识别哪些条目已经归档过。
        """
        file_path = self._get_category_file(category)
        processed: Set[str] = set()

        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except (json.JSONDecodeError, ValueError, TypeError):
                            # 跳过崩溃留下的不完整行
                            continue
                        if not isinstance(data, dict):
                            continue
                        if data.get("type") == "START_MARKER":
                            continue
                        dedup_id_field = data.get("dedup_id")
                        if dedup_id_field:
                            processed.add(dedup_id_field)
            except (IOError, OSError) as e:
                print("[ArchivalManager] 读取归档文件失败 {0}: {1}".format(file_path, e))

        self._processed_ids[category] = processed
        return processed

    def _is_already_processed(self, category: str, dedup_id_field: str) -> bool:
        """检查某个 dedup_id 是否已经在该分类中处理过。"""
        if category not in self._processed_ids:
            self._load_processed_ids(category)
        return dedup_id_field in self._processed_ids.get(category, set())

    def _mark_processed(self, category: str, dedup_id_field: str):
        """标记某个 dedup_id 已在该分类中处理过。"""
        self._processed_ids.setdefault(category, set()).add(dedup_id_field)

    # ---- 写入 ----

    def _write_start_marker(self, category: str, source_files: List[str]) -> str:
        """在分类文件中写入一个整理起点标记（审计用）。

        参数:
            category: 分类名称。
            source_files: 本次整理涉及的来源文件列表。

        返回:
            起点标记的批次 ID。
        """
        batch_id = uuid.uuid4().hex[:12]
        file_path = self._get_category_file(category)

        marker = {
            "type": "START_MARKER",
            "batch_id": batch_id,
            "timestamp": format_timestamp(),
            "source_files": source_files,
            "category": category,
        }

        try:
            with open(file_path, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(marker, ensure_ascii=False) + "\n")
        except (IOError, OSError) as e:
            print("[ArchivalManager] 写入起点标记失败 {0}: {1}".format(file_path, e))

        self._start_markers.append(marker)
        return batch_id

    def archive_entry(self, category: str, record: Dict[str, Any]) -> bool:
        """将一条流水记录归档到指定分类文件。

        幂等检查：如果该 dedup_id 已存在于目标分类，则跳过。

        参数:
            category: 目标分类名称。
            record: 流水记录字典。

        返回:
            成功归档返回 True，已处理过返回 False。
        """
        category = self._normalize_category(category)
        if not isinstance(record, dict):
            print("[ArchivalManager] 记录不是 dict，跳过归档: {0!r}".format(record))
            return False
        dedup_id_field = record.get("dedup_id", "")
        if not dedup_id_field:
            print("[ArchivalManager] 记录缺少 dedup_id（手动归档请自行提供稳定 ID），跳过归档: {0}".format(record))
            return False

        if self._is_already_processed(category, dedup_id_field):
            return False  # 已处理过，跳过

        file_path = self._get_category_file(category)

        archive_record = {
            "type": "ARCHIVE_ENTRY",
            "dedup_id": dedup_id_field,
            "source_file": record.get("source_file", ""),
            "entry_index": record.get("entry_index", -1),
            "timestamp": record.get("timestamp", ""),
            "coarse_type": record.get("coarse_type", ""),
            "content": record.get("content", ""),
            "source": record.get("source", ""),
            "verify_status": record.get("verify_status", ""),
            "risk_level": record.get("risk_level", ""),
            "archived_at": format_timestamp(),
        }
        if record.get("metadata"):
            archive_record["metadata"] = record["metadata"]

        try:
            with open(file_path, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(archive_record, ensure_ascii=False) + "\n")
            self._mark_processed(category, dedup_id_field)
            return True
        except (IOError, OSError) as e:
            print("[ArchivalManager] 归档写入失败 {0}: {1}".format(file_path, e))
            return False

    # ---- 整理流程 ----

    def process_staging(self, staging_manager: StagingManager,
                        category: Optional[str] = None) -> Dict[str, int]:
        """处理临时流水层，将未归档的条目按语义路由到分类文件。

        参数:
            staging_manager: 第一层临时流水管理器实例。
            category: 可选的特定分类。如果提供，只把流水写入该分类；
                      如果为 None，则按 CATEGORY_MAP 自动路由。

        返回:
            统计字典，包含 archived, skipped, failed 数量。
        """
        stats = {"archived": 0, "skipped": 0, "failed": 0}

        if category is not None:
            category = self._normalize_category(category)

        staging_records = staging_manager.read_all()
        if not staging_records:
            return stats

        # 先筛选出未归档记录，避免无意义地写 START_MARKER
        pending: List[Tuple[Dict[str, Any], List[str]]] = []
        for record in staging_records:
            dedup_id_field = record.get("dedup_id", "")
            if not dedup_id_field:
                stats["failed"] += 1
                continue

            if category is not None:
                cats = [category]
            else:
                coarse_type = record.get("coarse_type", "unknown")
                cats = [self.CATEGORY_MAP.get(
                    coarse_type, ArchiveCategory.ACTION_LOGS.value)]

            if all(self._is_already_processed(c, dedup_id_field) for c in cats):
                stats["skipped"] += 1
                continue
            pending.append((record, cats))

        if not pending:
            return stats

        # 只对本次真正涉及的目标分类写起点标记
        source_files = sorted({r.get("source_file", "") for r, _ in pending})
        cats_to_marker = sorted({c for _, cats in pending for c in cats})
        for cat in cats_to_marker:
            self._write_start_marker(cat, source_files)

        for record, cats in pending:
            dedup_id_field = record["dedup_id"]
            for cat in cats:
                if self._is_already_processed(cat, dedup_id_field):
                    continue
                if self.archive_entry(cat, record):
                    stats["archived"] += 1
                else:
                    stats["failed"] += 1

        self._start_markers.clear()
        return stats

    def process_all(self, staging_manager: StagingManager) -> Dict[str, int]:
        """处理所有临时流水（自动路由到各分类）。"""
        return self.process_staging(staging_manager, category=None)

    # ---- 统计与清理 ----

    def get_archived_count(self, category: Optional[str] = None) -> int:
        """获取归档文件中的条目数（不含 START_MARKER）。

        参数:
            category: 可选的特定分类。如果提供，只统计该分类。
        """
        if category is not None:
            category = self._normalize_category(category)

        def _count_file(file_path: str) -> int:
            if not os.path.exists(file_path):
                return 0
            c = 0
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except (json.JSONDecodeError, ValueError, TypeError):
                            continue
                        if isinstance(data, dict) and data.get("type") != "START_MARKER":
                            c += 1
            except (IOError, OSError):
                pass
            return c

        if category is not None:
            return _count_file(self._get_category_file(category))

        total = 0
        if os.path.exists(self.archival_dir):
            for filename in os.listdir(self.archival_dir):
                if filename.endswith(".jsonl"):
                    total += _count_file(os.path.join(self.archival_dir, filename))
        return total

    def list_categories(self) -> List[str]:
        """列出所有已有归档文件的分类。"""
        if not os.path.exists(self.archival_dir):
            return []
        return sorted(f[:-6] for f in os.listdir(self.archival_dir) if f.endswith(".jsonl"))

    def clear_all(self) -> int:
        """清空所有归档文件并重置内存缓存。

        返回:
            已删除的文件数量。
        """
        self._processed_ids.clear()
        self._start_markers.clear()

        if not os.path.exists(self.archival_dir):
            return 0

        count = 0
        for filename in os.listdir(self.archival_dir):
            if filename.endswith(".jsonl"):
                try:
                    os.remove(os.path.join(self.archival_dir, filename))
                    count += 1
                except (IOError, OSError):
                    pass
        return count

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
