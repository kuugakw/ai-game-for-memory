"""第三层记忆：快照 (Snapshot)。

设计要点：
  - 状态必须精确恢复，不能靠语义检索碰运气
  - 快照是全量状态 dump，恢复时直接替换，不依赖增量
  - 在不可逆操作前、清除上下文前自动创建快照
  - 快照文件独立存储，与 RAG 知识索引分离

快照与 RAG 的区别：
  - 快照 = 精确状态恢复（全量 JSON dump）
  - RAG = 知识按需检索（语义/关键词匹配）
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from .utils import format_timestamp, safe_stem


class SnapshotManager:
    """快照管理器。

    负责第三层记忆：创建和管理精确状态快照。
    """

    def __init__(self, workspace: str):
        """初始化 SnapshotManager。

        参数:
            workspace: 数据存储根目录路径。
        """
        self.workspace = workspace
        self.snapshot_dir = os.path.join(workspace, "snapshots")
        os.makedirs(self.snapshot_dir, exist_ok=True)

    # ---- 路径 ----

    def _snapshot_path(self, snapshot_id: str) -> str:
        """由 snapshot_id 得到快照文件路径（ID 已脱敏，不会越界）。"""
        return os.path.join(self.snapshot_dir, "{0}.json".format(snapshot_id))

    def _snapshot_ids(self) -> List[str]:
        """按写入时间从旧到新返回所有 snapshot_id。"""
        if not os.path.exists(self.snapshot_dir):
            return []
        files = [f for f in os.listdir(self.snapshot_dir) if f.endswith(".json")]
        files.sort(key=lambda f: os.path.getmtime(os.path.join(self.snapshot_dir, f)))
        return [f[:-5] for f in files]

    # ---- 创建 / 读取 / 恢复 ----

    def create_snapshot(self, name: str, state: Dict[str, Any],
                        include_context: bool = False,
                        context: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """创建一个状态快照。

        参数:
            name: 快照名称（会脱敏后用于文件名，原始名称保留在内容里）。
            state: 需要精确恢复的状态字典。
            include_context: 是否把当前上下文原文一并存入快照。
            context: 上下文原文（include_context 为 True 时保存）。
            metadata: 可选附加信息。

        返回:
            快照元信息字典（含 snapshot_id / file_path）。
        """
        if not isinstance(state, dict):
            raise TypeError("state 必须为 dict")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name 必须为非空字符串")

        dt = datetime.now()
        stem = safe_stem(name, fallback="snapshot")
        base_id = "{0}_{1}".format(stem, dt.strftime("%Y%m%d_%H%M%S"))

        # 同一秒内创建同名快照时追加序号，避免覆盖
        snapshot_id = base_id
        suffix = 1
        while os.path.exists(self._snapshot_path(snapshot_id)):
            suffix += 1
            snapshot_id = "{0}_{1}".format(base_id, suffix)

        snapshot_data = {
            "snapshot_id": snapshot_id,
            "name": name,
            "timestamp": format_timestamp(dt),
            "state": state,
        }
        if include_context and context is not None:
            snapshot_data["context"] = context
        if metadata:
            snapshot_data["metadata"] = metadata

        snapshot_file = self._snapshot_path(snapshot_id)
        with open(snapshot_file, "w", encoding="utf-8") as f:
            json.dump(snapshot_data, f, ensure_ascii=False, indent=2)

        return {
            "snapshot_id": snapshot_id,
            "file_path": snapshot_file,
            "name": name,
            "timestamp": snapshot_data["timestamp"],
            "state_size": len(state),
            "has_context": "context" in snapshot_data,
        }

    def _find_snapshot_id(self, snapshot_id_or_name: str) -> Optional[str]:
        """按 snapshot_id 或 name 查找快照，优先精确 ID，其次名称，最后 ID 前缀。"""
        if not snapshot_id_or_name:
            return None

        ids = self._snapshot_ids()
        if snapshot_id_or_name in ids:
            return snapshot_id_or_name

        # 按 name 匹配（取最新一个）
        by_name = None
        for sid in ids:
            data = self._read_snapshot(sid)
            if data and data.get("name") == snapshot_id_or_name:
                by_name = sid
        if by_name:
            return by_name

        # 按 ID 前缀匹配（取最新一个）
        by_prefix = None
        for sid in ids:
            if sid.startswith(snapshot_id_or_name):
                by_prefix = sid
        return by_prefix

    def _read_snapshot(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """读取完整快照数据（包含 state / context / metadata）。"""
        file_path = self._snapshot_path(snapshot_id)
        if not os.path.exists(file_path):
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (IOError, OSError, json.JSONDecodeError) as e:
            print("[SnapshotManager] 读取快照失败 {0}: {1}".format(file_path, e))
            return None
        return data if isinstance(data, dict) else None

    def load_snapshot(self, snapshot_id_or_name: str) -> Optional[Dict[str, Any]]:
        """按 ID 或名称读取完整快照数据；找不到返回 None。"""
        sid = self._find_snapshot_id(snapshot_id_or_name)
        if sid is None:
            return None
        return self._read_snapshot(sid)

    def restore_snapshot(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """通过 snapshot_id（或名称）恢复一个快照的状态。

        返回:
            快照中的 state 字典；找不到或读取失败返回 None。
        """
        data = self.load_snapshot(snapshot_id)
        if data is None:
            return None
        return data.get("state")

    def list_snapshots(self) -> List[Dict[str, Any]]:
        """列出所有已保存的快照元信息（按时间从旧到新）。"""
        results = []
        for sid in self._snapshot_ids():
            file_path = self._snapshot_path(sid)
            data = self._read_snapshot(sid)
            if data is None:
                continue
            results.append({
                "snapshot_id": data.get("snapshot_id", sid),
                "name": data.get("name", ""),
                "timestamp": data.get("timestamp", ""),
                "state_size": len(data.get("state", {}) or {}),
                "has_context": "context" in data,
                "file_path": file_path,
            })
        return results

    def delete_snapshot(self, snapshot_id: str) -> bool:
        """删除指定 snapshot_id 的快照文件。"""
        file_path = self._snapshot_path(snapshot_id)
        if not os.path.exists(file_path):
            return False
        try:
            os.remove(file_path)
            return True
        except (IOError, OSError) as e:
            print("[SnapshotManager] 删除快照失败 {0}: {1}".format(file_path, e))
            return False

    def clear_all(self) -> int:
        """清除所有快照文件。"""
        if not os.path.exists(self.snapshot_dir):
            return 0

        count = 0
        for filename in os.listdir(self.snapshot_dir):
            if not filename.endswith(".json"):
                continue
            file_path = os.path.join(self.snapshot_dir, filename)
            try:
                os.remove(file_path)
                count += 1
            except (IOError, OSError) as e:
                print("[SnapshotManager] 清除快照失败 {0}: {1}".format(file_path, e))
        return count

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
