"""AI Game for Memory - 为本地 / 离线 AI 游戏与 Agent 设计的分层记忆框架。

一个自包含（纯 Python 标准库）的分层记忆管理库，解决以下问题：

1. 上下文会爆：GUI 自动化、多步任务、截图和检索结果迅速占满窗口
2. 单模型决策风险：一个模型既理解又执行，容易理解偏差、漏约束、不可逆操作出错
3. 知识混在一起检索不准：所有记忆塞进一个大索引，版本之间互相干扰
4. 一清上下文就失忆：全清重建成本高，且容易丢状态

设计取向：上下文有上界、记忆无上界；状态用快照精确恢复、知识用 RAG 按需检索、
行为可复用可退场；高风险决策走正反合审查，低风险不浪费；权重可计算可审计。

三层记忆结构：
  第一层 - 临时流水 (staging)：按小时滚动写入，快、不阻塞主流程
  第二层 - 分类归档 (archival)：语义路由到分类文件，幂等、可恢复
  第三层 - 快照 (snapshot)：精确状态恢复，不靠语义检索碰运气

示例用法::

    from ai_game_for_memory import MemoryHub

    hub = MemoryHub(workspace="my_agent_memory")
    hub.stage_action("点击设置", verify_status="pass", risk_level="low")
    hub.create_snapshot("before_delete", include_context=True)
    results = hub.rag_search("如何处理超时", top_k=5)
    hub.process_archival()
"""

__version__ = "1.0.0"
__author__ = "kuugakw"

from .core import MemoryHub
from .config import (
    Config,
    RiskLevel,
    VerifyStatus,
    CoarseType,
    ArchiveCategory,
    ReviewOutcome,
)
from .staging import StagingManager
from .archival import ArchivalManager
from .snapshot import SnapshotManager
from .rag import RAGEngine
from .workflow import WorkflowManager
from .utils import (
    estimate_tokens,
    dedup_id,
    format_timestamp,
    safe_stem,
    sanitize_filename,
)

__all__ = [
    "MemoryHub",
    "Config",
    "RiskLevel",
    "VerifyStatus",
    "CoarseType",
    "ArchiveCategory",
    "ReviewOutcome",
    "StagingManager",
    "ArchivalManager",
    "SnapshotManager",
    "RAGEngine",
    "WorkflowManager",
    "estimate_tokens",
    "dedup_id",
    "format_timestamp",
    "safe_stem",
    "sanitize_filename",
]
