"""第一层记忆：临时流水 (Staging)。

设计要点：
  - 每步动作、截图摘要、决策、错误原样追加到按小时滚动的临时文件
  - 写入时不做复杂分类判断，只带粗分类提示、来源、验证状态、风险等级
  - 目标：快、不阻塞主流程、崩溃最多坏最后一行

崩溃安全机制：
  - 每条记录独立一行 JSON，写入后 flush（最多丢失最后一行）
  - 读取时跳过解析失败的行
  - 重新打开文件前会检查文件末尾：如果上次崩溃留下了没有换行符的半行，
    先补一个换行符再追加，避免新记录和半行拼成一行、把两条记录同时弄坏
  - 进程重启后从已有文件恢复 entry_index，保证 dedup_id 不与旧记录冲突

滚动机制：
  - 按 roll_hours 对齐小时生成新文件，文件名含小时时间戳
  - 旧文件不删除（由 retention_days 配置定期清理）
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from .config import CoarseType, RiskLevel, VerifyStatus, coerce_enum
from .utils import dedup_id, estimate_tokens, format_timestamp


class StagingManager:
    """临时流水管理器。

    负责第一层记忆：将原始流水按小时滚动追加到临时文件。
    写入快速、非阻塞、崩溃安全。
    """

    def __init__(self, workspace: str, roll_hours: int = 1):
        """初始化 StagingManager。

        参数:
            workspace: 数据存储根目录路径。
            roll_hours: 滚动小时数，1~24。
        """
        if not isinstance(roll_hours, int) or not (1 <= roll_hours <= 24):
            raise ValueError("roll_hours 必须为 1~24 之间的整数")
        self.workspace = workspace
        self.roll_hours = roll_hours
        self.staging_dir = os.path.join(workspace, "staging")
        os.makedirs(self.staging_dir, exist_ok=True)

        # 当前正在写入的流水文件路径与句柄
        self._current_file_path: Optional[str] = None
        self._current_file_handle = None
        self._current_hour_key: Optional[str] = None
        # 绝对路径 -> 已写入条目数（重启后从文件恢复）
        self._entry_index_map: Dict[str, int] = {}

    # ---- 路径与计数 ----

    def _get_hour_key(self, dt: Optional[datetime] = None) -> str:
        """获取当前小时键（用于文件名）。

        参数:
            dt: 可选的 datetime 对象，默认为当前时间。
        """
        if dt is None:
            dt = datetime.now()
        aligned = dt.replace(minute=0, second=0, microsecond=0)
        hour = aligned.hour // self.roll_hours * self.roll_hours
        aligned = aligned.replace(hour=hour)
        return aligned.strftime("%Y%m%dT%H00")

    def _get_file_path(self, dt: Optional[datetime] = None) -> str:
        """获取当前小时对应的流水文件路径。"""
        hour_key = self._get_hour_key(dt)
        return os.path.join(self.staging_dir, "staging_{0}.jsonl".format(hour_key))

    def _source_key(self, file_path: str) -> str:
        """把绝对路径转成相对 workspace 的稳定标识（用于 dedup_id）。

        使用相对路径而不是绝对路径，数据目录整体搬迁后 dedup_id 保持不变，
        归档不会因为路径变化而重复写入。
        """
        try:
            rel = os.path.relpath(file_path, self.workspace)
        except ValueError:  # Windows 跨盘符时 relpath 会失败
            rel = os.path.basename(file_path)
        return rel.replace("\\", "/")

    @staticmethod
    def _count_lines(file_path: str) -> int:
        """统计文件已有条目数（用于重启后恢复 entry_index）。"""
        if not os.path.exists(file_path):
            return 0
        count = 0
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        count += 1
        except (IOError, OSError):
            pass
        return count

    @staticmethod
    def _repair_truncated_tail(file_path: str):
        """如果文件末尾没有换行符（上次写入中途崩溃），补一个换行符。

        否则以追加模式打开后会直接接在半行后面写，结果是两条记录
        拼成一行，两条都无法解析。
        """
        if not os.path.exists(file_path):
            return
        try:
            size = os.path.getsize(file_path)
            if size == 0:
                return
            with open(file_path, "rb+") as f:
                f.seek(size - 1)
                if f.read(1) != b"\n":
                    f.seek(0, os.SEEK_END)
                    f.write(b"\n")
        except (IOError, OSError):
            pass

    def _ensure_file(self, dt: Optional[datetime] = None):
        """确保当前小时的文件句柄已打开。

        如果小时键发生变化（跨过滚动边界），关闭旧文件、打开新文件。
        首次打开某个文件时会修复半行尾巴，并从文件恢复 entry_index。
        """
        file_dt = dt or datetime.now()
        hour_key = self._get_hour_key(file_dt)
        file_path = self._get_file_path(file_dt)

        if self._current_hour_key != hour_key or self._current_file_handle is None:
            if self._current_file_handle:
                self._current_file_handle.flush()
                self._current_file_handle.close()
                self._current_file_handle = None

            os.makedirs(self.staging_dir, exist_ok=True)

            if file_path not in self._entry_index_map:
                self._repair_truncated_tail(file_path)
                self._entry_index_map[file_path] = self._count_lines(file_path)

            # newline="\n"：强制使用 LF，避免 Windows 下写出 CRLF 造成跨平台差异
            self._current_file_handle = open(file_path, "a", encoding="utf-8", newline="\n")
            self._current_file_path = file_path
            self._current_hour_key = hour_key

    # ---- 写入 ----

    def append(self, content: str, coarse_type: str = "unknown",
               source: str = "", verify_status: str = "pending",
               risk_level: str = "low",
               metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """追加一条流水记录到当前小时的临时文件。

        参数:
            content: 流水内容文本。
            coarse_type: 粗分类类型（默认为 "unknown"）。
            source: 来源标识（如 "gui_click", "model_decision"）。
            verify_status: 验证状态（默认为 "pending"）。
            risk_level: 风险等级（默认为 "low"）。
            metadata: 可选的附加元数据字典。

        返回:
            写入的流水记录字典。
        """
        if not isinstance(content, str) or not content:
            raise ValueError("content 必须为非空字符串")

        coarse_type = coerce_enum(coarse_type)
        risk_level = coerce_enum(risk_level)
        verify_status = coerce_enum(verify_status)

        valid_types = [t.value for t in CoarseType]
        if coarse_type not in valid_types:
            raise ValueError(
                "coarse_type '{0}' 不是有效类型，可选值：{1}".format(coarse_type, valid_types)
            )

        valid_risks = [r.value for r in RiskLevel]
        if risk_level not in valid_risks:
            raise ValueError(
                "risk_level '{0}' 不是有效等级，可选值：{1}".format(risk_level, valid_risks)
            )

        valid_verifies = [v.value for v in VerifyStatus]
        if verify_status not in valid_verifies:
            raise ValueError(
                "verify_status '{0}' 不是有效状态，可选值：{1}".format(verify_status, valid_verifies)
            )

        dt = datetime.now()
        self._ensure_file(dt)

        file_path = self._current_file_path
        source_key = self._source_key(file_path)
        index = self._entry_index_map.get(file_path, 0)
        dedup = dedup_id(source_key, index)

        record = {
            "dedup_id": dedup,
            "source_file": source_key,
            "entry_index": index,
            "timestamp": format_timestamp(dt),
            "coarse_type": coarse_type,
            "content": content,
            "source": source,
            "verify_status": verify_status,
            "risk_level": risk_level,
            "token_estimate": estimate_tokens(content),
        }
        if metadata:
            record["metadata"] = metadata

        line = json.dumps(record, ensure_ascii=False) + "\n"
        try:
            self._current_file_handle.write(line)
            self._current_file_handle.flush()
        except (IOError, OSError) as e:  # 写失败时不要把索引推进
            raise IOError("写入临时流水失败 {0}: {1}".format(file_path, e))
        self._entry_index_map[file_path] = index + 1

        return record

    def stage_action(self, content: str, source: str = "", verify_status: str = "pending",
                     risk_level: str = "low",
                     metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """便捷方法：追加一条动作流水。"""
        return self.append(content, coarse_type="action", source=source,
                           verify_status=verify_status, risk_level=risk_level,
                           metadata=metadata)

    def stage_decision(self, content: str, source: str = "", verify_status: str = "pending",
                       risk_level: str = "low",
                       metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """便捷方法：追加一条决策流水。"""
        return self.append(content, coarse_type="decision", source=source,
                           verify_status=verify_status, risk_level=risk_level,
                           metadata=metadata)

    def stage_error(self, content: str, source: str = "", verify_status: str = "fail",
                    risk_level: str = "medium",
                    metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """便捷方法：追加一条错误流水。"""
        return self.append(content, coarse_type="error", source=source,
                           verify_status=verify_status, risk_level=risk_level,
                           metadata=metadata)

    def stage_screenshot(self, content: str, source: str = "", verify_status: str = "pending",
                         risk_level: str = "low",
                         metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """便捷方法：追加一条截图摘要流水。"""
        return self.append(content, coarse_type="screenshot", source=source,
                           verify_status=verify_status, risk_level=risk_level,
                           metadata=metadata)

    def stage_reasoning(self, content: str, source: str = "", verify_status: str = "pending",
                        risk_level: str = "low",
                        metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """便捷方法：追加一条推理过程流水。"""
        return self.append(content, coarse_type="reasoning", source=source,
                           verify_status=verify_status, risk_level=risk_level,
                           metadata=metadata)

    def stage_state(self, content: str, source: str = "", verify_status: str = "pending",
                    risk_level: str = "low",
                    metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """便捷方法：追加一条状态流水。"""
        return self.append(content, coarse_type="state", source=source,
                           verify_status=verify_status, risk_level=risk_level,
                           metadata=metadata)

    # ---- 读取 ----

    def read_all(self) -> List[Dict[str, Any]]:
        """读取所有小时的流水记录（跳过崩溃时不完整的行）。

        返回:
            流水记录列表。
        """
        results: List[Dict[str, Any]] = []
        if not os.path.exists(self.staging_dir):
            return results

        files = sorted(f for f in os.listdir(self.staging_dir) if f.endswith(".jsonl"))
        for filename in files:
            file_path = os.path.join(self.staging_dir, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        record = self._parse_safe(line)
                        if record is not None:
                            results.append(record)
            except (IOError, OSError) as e:
                print("[StagingManager] 读取文件失败 {0}: {1}".format(file_path, e))
        return results

    @staticmethod
    def _parse_safe(line: str) -> Optional[Dict[str, Any]]:
        """安全解析一行 JSON，失败返回 None（崩溃安全）。"""
        try:
            data = json.loads(line)
        except (json.JSONDecodeError, ValueError, TypeError):
            return None
        return data if isinstance(data, dict) else None

    # ---- 状态 ----

    @property
    def current_file_path(self) -> Optional[str]:
        """当前正在写入的文件路径（未写入过则为 None）。"""
        return self._current_file_path

    def get_current_file_info(self) -> Dict[str, Any]:
        """获取当前正在写入的文件信息。"""
        current = self._current_file_path or ""
        return {
            "file_path": current,
            "hour_key": self._current_hour_key or "",
            "entry_count": self._entry_index_map.get(current, 0),
        }

    def close(self):
        """关闭当前文件句柄（资源清理，可继续使用该实例）。"""
        if self._current_file_handle:
            try:
                self._current_file_handle.flush()
            finally:
                self._current_file_handle.close()
                self._current_file_handle = None
                self._current_file_path = None
                self._current_hour_key = None

    def reset(self):
        """关闭句柄并清空内部计数（用于 clear_all / 删除文件前）。"""
        self.close()
        self._entry_index_map.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
