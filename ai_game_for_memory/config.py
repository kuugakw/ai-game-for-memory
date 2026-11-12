"""配置项与枚举定义。

本模块定义了插件使用的所有枚举类型和配置参数。
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class RiskLevel(Enum):
    """风险等级枚举。

    用于标记动作/决策的风险程度，决定是否需要正反合审查。
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class VerifyStatus(Enum):
    """验证状态枚举。

    标记一个流水条目是否经过验证。
    """
    PENDING = "pending"
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"


class CoarseType(Enum):
    """临时流水的粗分类枚举。

    第一层流水只做粗分类，不做精细语义判断。
    """
    ACTION = "action"
    SCREENSHOT = "screenshot"
    DECISION = "decision"
    ERROR = "error"
    REASONING = "reasoning"
    STATE = "state"
    UNKNOWN = "unknown"


class ArchiveCategory(Enum):
    """第二层分类归档的目标类别。

    按语义路由到不同类别文件，避免版本干扰。
    """
    UI_FACTS = "ui_facts"
    ACTION_LOGS = "action_logs"
    DECISIONS_REASONING = "decisions_reasoning"
    UI_MAPS = "ui_maps"
    ERRORS_FIXES = "errors_fixes"
    BEHAVIOR_TRISTATE = "behavior_tristate"
    SNAPSHOTS = "snapshots"

    @classmethod
    def values(cls):
        """返回所有合法分类值。"""
        return [c.value for c in cls]

    @classmethod
    def human_readable(cls) -> Dict[str, str]:
        """返回人类可读的类别名称映射。"""
        return {
            cls.UI_FACTS.value: "界面事实",
            cls.ACTION_LOGS.value: "动作流水",
            cls.DECISIONS_REASONING.value: "决策与理由",
            cls.UI_MAPS.value: "界面地图",
            cls.ERRORS_FIXES.value: "错误与修正",
            cls.BEHAVIOR_TRISTATE.value: "行为三态",
            cls.SNAPSHOTS.value: "快照",
        }


class ReviewOutcome(Enum):
    """正反合审查的输出结果。"""
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_MORE_INFO = "needs_more_info"


def coerce_enum(value):
    """接受枚举成员或其字符串值，统一返回字符串值。

    这样 ``stage_action(..., risk_level=RiskLevel.HIGH)`` 与
    ``stage_action(..., risk_level="high")`` 都能用。
    """
    if isinstance(value, Enum):
        return value.value
    return value


@dataclass
class Config:
    """AI Game for Memory 配置。

    参数:
        workspace: 数据存储根目录路径。
        context_max_tokens: 上下文窗口最大 token 数（估算值）。
        context_eviction_ratio: 接近上限时淘汰的比例（0.0~1.0）。
        staging_roll_hours: 临时流水滚动小时数（1~24）。
        archival_max_file_size_mb: 分类归档单文件最大 MB 数（保留字段）。
        rag_top_k: RAG 检索默认返回条数。
        review_high_risk_only: 是否仅对高风险决策启动正反合审查。
        snapshot_on_irreversible: 不可逆操作前是否自动创建快照。
        snapshot_on_context_clear: 清除上下文前是否自动创建快照。
        retention_days: 临时流水保留天数（超过自动清理）。
    """
    workspace: str = "ai_game_for_memory_data"
    context_max_tokens: int = 128_000
    context_eviction_ratio: float = 0.5
    staging_roll_hours: int = 1
    archival_max_file_size_mb: float = 50.0
    rag_top_k: int = 5
    review_high_risk_only: bool = True
    snapshot_on_irreversible: bool = True
    snapshot_on_context_clear: bool = True
    retention_days: int = 7

    def __post_init__(self):
        """校验配置参数的合法性。"""
        if not isinstance(self.workspace, str) or not self.workspace:
            raise ValueError("workspace 必须为非空字符串")
        if not isinstance(self.context_max_tokens, int) or self.context_max_tokens <= 0:
            raise ValueError("context_max_tokens 必须为正整数")
        if not (0.0 < self.context_eviction_ratio <= 1.0):
            raise ValueError("context_eviction_ratio 必须在 (0.0, 1.0] 范围内")
        if not isinstance(self.staging_roll_hours, int) or not (1 <= self.staging_roll_hours <= 24):
            raise ValueError("staging_roll_hours 必须为 1~24 之间的整数")
        if self.archival_max_file_size_mb <= 0:
            raise ValueError("archival_max_file_size_mb 必须为正数")
        if not isinstance(self.rag_top_k, int) or self.rag_top_k <= 0:
            raise ValueError("rag_top_k 必须为正整数")
        if not isinstance(self.retention_days, int) or self.retention_days <= 0:
            raise ValueError("retention_days 必须为正整数")
