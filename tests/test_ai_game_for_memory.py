#!/usr/bin/env python3
"""AI Game for Memory 插件单元测试。

使用方式：
    python -m pytest tests/test_ai_game_for_memory.py -v

或（不依赖 pytest 的内置 runner）：
    python tests/test_ai_game_for_memory.py
"""

import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone

# 将插件根目录加入 Python 路径
PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PLUGIN_ROOT)

from ai_game_for_memory.config import (  # noqa: E402
    ArchiveCategory,
    CoarseType,
    Config,
    ReviewOutcome,
    RiskLevel,
    VerifyStatus,
)
from ai_game_for_memory.utils import (  # noqa: E402
    dedup_id,
    estimate_tokens,
    format_timestamp,
    safe_json_loads,
    safe_stem,
    sanitize_filename,
)
from ai_game_for_memory.staging import StagingManager  # noqa: E402
from ai_game_for_memory.archival import ArchivalManager  # noqa: E402
from ai_game_for_memory.snapshot import SnapshotManager  # noqa: E402
from ai_game_for_memory.rag import RAGEngine  # noqa: E402
from ai_game_for_memory.workflow import DecisionReview, WorkflowManager  # noqa: E402
from ai_game_for_memory.core import MemoryHub  # noqa: E402


def _assert_raises(exc_type, func, *args, **kwargs):
    """断言调用会抛出指定异常（不依赖 pytest.raises）。"""
    try:
        func(*args, **kwargs)
    except exc_type:
        return
    raise AssertionError("应抛出 {0}".format(exc_type.__name__))


class _TempWorkspace:
    """临时工作区上下文管理器，测试结束自动删除。"""

    def __init__(self, prefix="test_ai_game_for_memory_"):
        self.prefix = prefix
        self.path = None

    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix=self.prefix)
        return self.path

    def __exit__(self, exc_type, exc_val, exc_tb):
        shutil.rmtree(self.path, ignore_errors=True)
        return False


# ============================================================
# 工具函数测试
# ============================================================

def test_dedup_id():
    """测试去重 ID 生成。"""
    id1 = dedup_id("file_a.jsonl", 0)
    id2 = dedup_id("file_a.jsonl", 0)
    id3 = dedup_id("file_a.jsonl", 1)
    id4 = dedup_id("file_b.jsonl", 0)

    assert id1 == id2, "相同来源+序号应生成相同 ID"
    assert id1 != id3, "不同序号应生成不同 ID"
    assert id1 != id4, "不同来源应生成不同 ID"
    assert len(id1) == 40, "ID 应为 40 位十六进制"


def test_format_timestamp():
    """测试时间戳格式化。"""
    dt = datetime(2026, 1, 7, 12, 30, 45, tzinfo=timezone.utc)
    assert format_timestamp(dt, iso=True) == "2026-01-07T12:30:45+00:00"
    assert format_timestamp(dt, iso=False) == "2026-01-07 12:30:45"

    # naive datetime 按本地时间处理，生成的 ISO 串必须带时区偏移
    # （旧实现把本地时间直接标成 UTC）
    naive_iso = format_timestamp(datetime(2026, 1, 7, 12, 30, 45), iso=True)
    assert naive_iso.startswith("2026-01-07T12:30:45")
    assert naive_iso.endswith("+00:00") or naive_iso[-6] in "+-"


def test_estimate_tokens():
    """测试 token 估算。"""
    assert estimate_tokens("") == 0
    assert estimate_tokens("hello") > 0
    assert estimate_tokens("你好世界") > 0
    assert estimate_tokens("hello 你好 world 世界") > 0


def test_safe_json_loads():
    """测试安全 JSON 解析。"""
    assert safe_json_loads('{"key": "value"}') == {"key": "value"}
    assert safe_json_loads("{invalid json") is None
    assert safe_json_loads("") is None


def test_sanitize_filename():
    """测试文件名安全处理。"""
    assert sanitize_filename("test/file:*.txt") == "test_file__.txt"
    assert sanitize_filename("normal.txt") == "normal.txt"


def test_safe_stem():
    """测试稳定文件名主干。"""
    assert safe_stem("doc_001") == "doc_001"
    # 不同非法 ID 不应落到同一文件名
    assert safe_stem("a/b") != safe_stem("a:b")
    assert safe_stem("a/b") == safe_stem("a/b")  # 稳定
    # 纯中文名也要有可用的 ASCII 主干
    stem = safe_stem("界面事实")
    assert stem
    assert all(c.isalnum() or c in "_-" for c in stem)
    assert "/" not in safe_stem("a/b")


# ============================================================
# 配置测试
# ============================================================

def test_config_valid():
    """测试有效配置。"""
    config = Config(workspace="test_workspace")
    assert config.workspace == "test_workspace"
    assert config.context_max_tokens == 128_000
    assert config.review_high_risk_only is True


def test_config_invalid_workspace():
    """测试空 workspace 报错。"""
    _assert_raises(ValueError, Config, workspace="")


def test_config_invalid_eviction_ratio():
    """测试无效 eviction_ratio 报错。"""
    _assert_raises(ValueError, Config, workspace="test", context_eviction_ratio=0.0)
    _assert_raises(ValueError, Config, workspace="test", context_eviction_ratio=1.5)


def test_config_invalid_roll_hours():
    """测试无效 staging_roll_hours 报错。"""
    _assert_raises(ValueError, Config, workspace="test", staging_roll_hours=0)
    _assert_raises(ValueError, Config, workspace="test", staging_roll_hours=25)


def test_enum_values():
    """测试枚举值。"""
    assert RiskLevel.LOW.value == "low"
    assert RiskLevel.MEDIUM.value == "medium"
    assert RiskLevel.HIGH.value == "high"

    assert VerifyStatus.PENDING.value == "pending"
    assert VerifyStatus.PASS.value == "pass"
    assert VerifyStatus.FAIL.value == "fail"
    assert VerifyStatus.SKIP.value == "skip"

    assert CoarseType.ACTION.value == "action"
    assert CoarseType.UNKNOWN.value == "unknown"

    assert ArchiveCategory.UI_FACTS.value == "ui_facts"
    assert ArchiveCategory.ACTION_LOGS.value == "action_logs"
    assert ArchiveCategory.DECISIONS_REASONING.value == "decisions_reasoning"
    assert ArchiveCategory.UI_MAPS.value == "ui_maps"
    assert ArchiveCategory.ERRORS_FIXES.value == "errors_fixes"
    assert ArchiveCategory.BEHAVIOR_TRISTATE.value == "behavior_tristate"
    assert ArchiveCategory.SNAPSHOTS.value == "snapshots"

    assert ReviewOutcome.APPROVED.value == "approved"
    assert ReviewOutcome.REJECTED.value == "rejected"
    assert ReviewOutcome.NEEDS_MORE_INFO.value == "needs_more_info"


# ============================================================
# 第一层：临时流水测试
# ============================================================

def test_staging_basic():
    """测试临时流水基本功能。"""
    with _TempWorkspace() as workspace:
        staging = StagingManager(workspace=workspace, roll_hours=1)
        record = staging.stage_action("点击设置", source="gui", risk_level="low")
        assert record["content"] == "点击设置"
        assert record["coarse_type"] == "action"
        assert record["risk_level"] == "low"
        assert record["dedup_id"] != ""
        assert os.path.isabs(record["source_file"]) is False, "来源文件应为相对路径"

        records = staging.read_all()
        assert len(records) >= 1
        assert records[-1]["content"] == "点击设置"
        staging.close()


def test_staging_hourly_roll():
    """测试按小时滚动写入。"""
    with _TempWorkspace() as workspace:
        staging = StagingManager(workspace=workspace, roll_hours=1)
        staging.stage_action("test action", source="test")

        files = [f for f in os.listdir(staging.staging_dir) if f.endswith(".jsonl")]
        assert len(files) == 1
        assert "staging_" in files[0]
        staging.close()


def test_staging_crash_safe():
    """测试崩溃安全：不完整行被跳过。"""
    with _TempWorkspace() as workspace:
        staging = StagingManager(workspace=workspace, roll_hours=1)
        file_path = staging._get_file_path()
        with open(file_path, "a", encoding="utf-8") as f:
            f.write('{"incomplete": true, "content": "crash"}\n')
            f.write('{"incomplete": true, "content": "partial"')  # 不完整行

        records = staging.read_all()
        assert len(records) == 1
        assert records[0]["content"] == "crash"
        staging.close()


def test_staging_repair_truncated_tail():
    """回归：崩溃留下半行后重启，新记录不能与半行拼成一行。

    旧实现直接以追加模式打开，新记录接在半行后面，导致两条记录都解析失败。
    """
    with _TempWorkspace() as workspace:
        staging = StagingManager(workspace=workspace, roll_hours=1)
        file_path = staging._get_file_path()
        staging.close()
        with open(file_path, "a", encoding="utf-8") as f:
            f.write('{"content": "ok"}\n')
            f.write('{"content": "half')  # 崩溃，没有换行符

        staging2 = StagingManager(workspace=workspace, roll_hours=1)
        staging2.stage_action("正常记录", source="test")
        contents = [r.get("content") for r in staging2.read_all()]
        assert contents == ["ok", "正常记录"]
        staging2.close()


def test_staging_restart_index():
    """测试重启后 entry_index 继续累加，dedup_id 不重复。"""
    with _TempWorkspace() as workspace:
        s1 = StagingManager(workspace=workspace, roll_hours=1)
        r1 = s1.stage_action("第一条", source="test")
        s1.close()

        s2 = StagingManager(workspace=workspace, roll_hours=1)
        r2 = s2.stage_action("第二条", source="test")
        assert r2["entry_index"] == r1["entry_index"] + 1
        assert r2["dedup_id"] != r1["dedup_id"]
        assert len(s2.read_all()) == 2
        s2.close()


# ============================================================
# 第三层：快照测试
# ============================================================

def test_snapshot_basic():
    """测试快照基本功能。"""
    with _TempWorkspace() as workspace:
        snapshot_mgr = SnapshotManager(workspace=workspace)
        info = snapshot_mgr.create_snapshot(
            "test_snapshot",
            state={"key1": "value1", "key2": [1, 2, 3]},
            include_context=True,
            context="test context text"
        )
        assert info["name"] == "test_snapshot"
        assert info["state_size"] == 2
        assert info["has_context"] is True

        snapshots = snapshot_mgr.list_snapshots()
        assert len(snapshots) == 1
        assert snapshots[0]["name"] == "test_snapshot"

        state = snapshot_mgr.restore_snapshot(info["snapshot_id"])
        assert state is not None
        assert state["key1"] == "value1"
        assert state["key2"] == [1, 2, 3]

        # 按名称恢复
        assert snapshot_mgr.restore_snapshot("test_snapshot")["key1"] == "value1"

        assert snapshot_mgr.delete_snapshot(info["snapshot_id"]) is True
        assert snapshot_mgr.delete_snapshot("nonexistent_id") is False
        assert snapshot_mgr.clear_all() == 0


def test_snapshot_unsafe_name():
    """回归：快照名不能把文件写到 snapshots 目录之外。"""
    with _TempWorkspace() as workspace:
        snapshot_mgr = SnapshotManager(workspace=workspace)
        info = snapshot_mgr.create_snapshot("../evil/name", {"k": 1})

        assert os.path.dirname(os.path.abspath(info["file_path"])) == \
            os.path.abspath(snapshot_mgr.snapshot_dir)
        assert os.path.exists(info["file_path"])
        assert snapshot_mgr.restore_snapshot("../evil/name") == {"k": 1}

        # 同名快照同一秒创建不覆盖
        second = snapshot_mgr.create_snapshot("../evil/name", {"k": 2})
        assert second["snapshot_id"] != info["snapshot_id"]
        assert len(snapshot_mgr.list_snapshots()) == 2


# ============================================================
# RAG 检索测试
# ============================================================

def test_rag_add_and_search():
    """测试 RAG 添加文档和检索。"""
    with _TempWorkspace() as workspace:
        rag = RAGEngine(workspace=workspace, top_k=5)
        assert rag.add_document("doc_001", "API 超时时重试 3 次", "errors_fixes") is True
        assert rag.add_document("doc_002", "设置页面包含用户信息选项卡", "ui_facts") is True
        assert rag.add_document("doc_003", "网络错误时记录日志并通知用户", "errors_fixes") is True

        results = rag.search("超时", mode="hybrid", top_k=5)
        assert len(results) >= 1
        assert results[0]["doc_id"] == "doc_001" or any(
            r["doc_id"] in ("doc_001", "doc_003") for r in results)

        err_results = rag.search("错误", category="errors_fixes", top_k=5)
        assert len(err_results) >= 1
        for r in err_results:
            assert r["category"] == "errors_fixes"

        assert len(rag.search("超时", mode="bm25", top_k=5)) >= 1
        assert len(rag.search("超时", mode="vector", top_k=5)) >= 1


def test_rag_remove_document():
    """测试 RAG 删除文档。"""
    with _TempWorkspace() as workspace:
        rag = RAGEngine(workspace=workspace, top_k=5)
        rag.add_document("doc_001", "测试内容", "test_category")
        assert rag.remove_document("doc_001") is True
        assert rag.remove_document("nonexistent") is False
        assert len(rag.search("测试", top_k=5)) == 0


def test_rag_reload_rebuild():
    """回归：新实例必须能从文档重建索引（不能依赖单独的索引文件）。"""
    with _TempWorkspace() as workspace:
        rag = RAGEngine(workspace=workspace)
        rag.add_document("doc_001", "API 超时时重试 3 次", "errors_fixes")
        rag.add_document("doc_002", "设置页面包含用户信息", "ui_facts")

        reloaded = RAGEngine(workspace=workspace)
        assert reloaded.get_document("doc_001") is not None
        assert len(reloaded.search("超时", mode="bm25")) >= 1
        assert reloaded.get_category_stats() == {"errors_fixes": 1, "ui_facts": 1}


def test_rag_doc_id_with_path_separator():
    """回归：doc_id 含 / 时不能把文件写到别的目录。"""
    with _TempWorkspace() as workspace:
        rag = RAGEngine(workspace=workspace)
        rag.add_document("ui/settings", "设置页面", "ui_facts")
        assert rag.get_document("ui/settings") is not None

        files = os.listdir(rag.documents_dir)
        assert len(files) == 1
        assert os.path.dirname(os.path.join(rag.documents_dir, files[0])) == \
            os.path.abspath(rag.documents_dir)

        reloaded = RAGEngine(workspace=workspace)
        assert reloaded.get_document("ui/settings")["text"] == "设置页面"


def test_rag_duplicate_add():
    """回归：覆盖同一 doc_id 不应残留旧词项。"""
    with _TempWorkspace() as workspace:
        rag = RAGEngine(workspace=workspace)
        rag.add_document("doc_001", "旧内容 超时", "errors_fixes")
        rag.add_document("doc_001", "新内容 网络", "ui_facts")

        assert rag.get_document("doc_001")["text"] == "新内容 网络"
        assert rag.get_category_stats() == {"ui_facts": 1}
        # 旧词不应再命中该文档
        results = rag.search("超时", mode="bm25")
        assert all(r["doc_id"] != "doc_001" for r in results)


def test_rag_invalid_arguments():
    """测试非法参数校验。"""
    with _TempWorkspace() as workspace:
        rag = RAGEngine(workspace=workspace)
        rag.add_document("doc_001", "内容", "cat")
        _assert_raises(ValueError, rag.search, "内容", mode="bogus")
        _assert_raises(ValueError, rag.add_document, "", "内容", "cat")
        _assert_raises(ValueError, rag.add_document, "id", "内容", "")
        _assert_raises(ValueError, rag.add_document, "id", "内容", "cat", vector=[])


# ============================================================
# 第二层：分类归档测试
# ============================================================

def test_archival_idempotent():
    """测试归档的幂等性。"""
    with _TempWorkspace() as workspace:
        staging = StagingManager(workspace=workspace, roll_hours=1)
        archival = ArchivalManager(workspace=workspace)

        staging.stage_action("点击设置", source="gui", risk_level="low")
        staging.stage_action("输入用户名", source="gui", risk_level="low")

        stats1 = archival.process_all(staging)
        assert stats1["archived"] == 2
        assert stats1["skipped"] == 0

        stats2 = archival.process_all(staging)
        assert stats2["archived"] == 0
        assert stats2["skipped"] == 2

        assert "action_logs" in archival.list_categories()
        assert archival.get_archived_count("action_logs") == 2
        staging.close()


def test_archival_routing():
    """测试粗分类到归档分类的路由。"""
    with _TempWorkspace() as workspace:
        staging = StagingManager(workspace=workspace, roll_hours=1)
        archival = ArchivalManager(workspace=workspace)

        staging.stage_action("动作", source="gui")
        staging.stage_screenshot("界面", source="gui")
        staging.stage_error("错误", source="api")
        staging.stage_decision("决策", source="agent")
        staging.stage_reasoning("推理", source="agent")
        staging.stage_state("状态", source="agent")

        stats = archival.process_all(staging)
        assert stats["archived"] == 6
        categories = archival.list_categories()
        for expected in ("action_logs", "ui_facts", "errors_fixes",
                         "decisions_reasoning", "behavior_tristate"):
            assert expected in categories
        staging.close()


def test_archival_crash_replay():
    """测试崩溃重放：重复处理不会重复写入。"""
    with _TempWorkspace() as workspace:
        staging = StagingManager(workspace=workspace, roll_hours=1)
        archival = ArchivalManager(workspace=workspace)

        for i in range(5):
            staging.stage_action("action {0}".format(i), source="test", risk_level="low")

        stats1 = archival.process_staging(staging, category="action_logs")
        assert stats1["archived"] == 5

        stats2 = archival.process_staging(staging, category="action_logs")
        assert stats2["archived"] == 0
        assert stats2["skipped"] == 5
        assert archival.get_archived_count("action_logs") == 5
        staging.close()


def test_archival_invalid_category():
    """测试非法归档分类被拒绝。"""
    with _TempWorkspace() as workspace:
        staging = StagingManager(workspace=workspace, roll_hours=1)
        archival = ArchivalManager(workspace=workspace)
        staging.stage_action("动作", source="gui")
        _assert_raises(ValueError, archival.process_staging, staging, "not_a_category")
        staging.close()


def test_archival_marker_only_for_touched_categories():
    """测试起点标记只写入实际涉及的分类。"""
    with _TempWorkspace() as workspace:
        staging = StagingManager(workspace=workspace, roll_hours=1)
        archival = ArchivalManager(workspace=workspace)
        staging.stage_action("动作", source="gui")
        archival.process_all(staging)

        # 没有待归档内容时不应再写 marker
        stats = archival.process_all(staging)
        assert stats["archived"] == 0

        marker_file = os.path.join(archival.archival_dir, "action_logs.jsonl")
        with open(marker_file, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        markers = [line for line in lines if line.get("type") == "START_MARKER"]
        assert len(markers) == 1
        assert not os.path.exists(os.path.join(archival.archival_dir, "ui_facts.jsonl"))
        staging.close()


# ============================================================
# 工作流与决策审查测试
# ============================================================

def test_decision_review_high_risk_rejected():
    """高风险决策：反对理由明显更强时应拒绝。"""
    review = DecisionReview("test_decision", "删除所有文件", "high")
    review.add_thesis("释放磁盘空间")
    review.add_antithesis("可能丢失未归档的流水记录")
    review.add_synthesis("综合正反面，建议先创建快照再删除")

    outcome = review.evaluate()
    assert outcome == ReviewOutcome.REJECTED.value
    assert review.approved is False
    assert review.weights["antithesis_weight"] > 0.5
    assert review.weights["risk_adjusted_score"] > 0.3


def test_decision_review_high_risk_approved():
    """高风险决策：反对理由足够弱时才批准。"""
    review = DecisionReview("test_decision_ok", "清理日志", "high")
    review.add_thesis("日志已归档到别处，磁盘使用率 95%，不清理会写满导致服务不可用")
    review.add_antithesis("风险")
    assert review.evaluate() == ReviewOutcome.APPROVED.value
    assert review.approved is True


def test_decision_review_low_risk_auto_approve():
    """测试低风险决策自动放行。"""
    review = DecisionReview("test_low", "关闭设置页面", "low")
    assert review.evaluate() == ReviewOutcome.APPROVED.value
    assert review.approved is True


def test_decision_review_needs_more_info():
    """回归：缺少正/反理由时不能编造理由替调用方决定。"""
    review = DecisionReview("test_missing", "删除所有文件", "high")
    review.add_thesis("释放空间")
    assert review.evaluate() == ReviewOutcome.NEEDS_MORE_INFO.value
    assert review.approved is False

    _assert_raises(ValueError, DecisionReview, "x", "y", "unknown")


def test_workflow_manager():
    """测试 WorkflowManager 完整流程。"""
    with _TempWorkspace() as workspace:
        wf = WorkflowManager(workspace=workspace, review_high_risk_only=True)

        # 低风险：直接放行（未经审查）
        low_result = wf.auto_approve_low_risk("关闭设置", "low")
        assert low_result["approved"] is True
        assert low_result["reviewed"] is False
        assert low_result["outcome"] == ReviewOutcome.APPROVED.value

        # 高风险：必须审查
        high_result = wf.execute_decision(
            decision_text="删除所有临时文件",
            risk_level="high",
            thesis_reason="释放磁盘空间",
            antithesis_reason="可能丢失未归档记录"
        )
        assert high_result["reviewed"] is True
        assert high_result["decision_text"] == "删除所有临时文件"
        assert high_result["outcome"] in (
            ReviewOutcome.APPROVED.value, ReviewOutcome.REJECTED.value)
        assert high_result["weights"]["thesis_length"] == len("释放磁盘空间")

        # 高风险但没给理由：needs_more_info
        missing = wf.execute_decision("删除所有临时文件", "high")
        assert missing["reviewed"] is True
        assert missing["approved"] is False
        assert missing["outcome"] == ReviewOutcome.NEEDS_MORE_INFO.value

        # 非法风险等级
        _assert_raises(ValueError, wf.execute_decision, "x", "critical")

        # 行为三态
        wf.register_behavior(
            behavior_id="test_behavior",
            pattern="重试 → 记录 → 通知",
            description="测试行为模式",
            status="in_progress"
        )
        assert wf.get_behavior("test_behavior")["status"] == "in_progress"
        assert wf.retire_behavior("test_behavior") is True
        assert wf.get_behavior("test_behavior")["status"] == "retired"


def test_workflow_medium_review():
    """中风险决策在 review_high_risk_only=False 时走审查。"""
    with _TempWorkspace() as workspace:
        wf = WorkflowManager(workspace=workspace, review_high_risk_only=False)

        approved = wf.execute_decision(
            "调整重试次数", "medium",
            thesis_reason="减少无效等待时间",
            antithesis_reason="可能")
        assert approved["reviewed"] is True
        assert approved["outcome"] == ReviewOutcome.APPROVED.value

        rejected = wf.execute_decision(
            "调整重试次数", "medium",
            thesis_reason="快",
            antithesis_reason="可能导致请求风暴，压垮下游服务")
        assert rejected["outcome"] == ReviewOutcome.REJECTED.value


def test_behavior_reregister_updates_definition():
    """回归：重新注册同一行为应更新 pattern/description，而不只是 status。"""
    with _TempWorkspace() as workspace:
        wf = WorkflowManager(workspace=workspace)
        wf.register_behavior("b1", "旧模式", "旧描述", "in_progress")
        wf.register_behavior("b1", "新模式", "新描述", "done")

        record = wf.get_behavior("b1")
        assert record["pattern"] == "新模式"
        assert record["description"] == "新描述"
        assert record["status"] == "done"

        # 重新加载后仍一致（文件里确实更新了）
        reloaded = WorkflowManager(workspace=workspace)
        assert reloaded.get_behavior("b1")["pattern"] == "新模式"


# ============================================================
# 核心编排器 (MemoryHub) 测试
# ============================================================

def test_memoryhub_full_flow():
    """测试 MemoryHub 完整流程。"""
    with _TempWorkspace() as workspace:
        hub = MemoryHub(workspace=workspace)

        hub.stage_action("点击设置", source="gui", risk_level="low")
        hub.stage_error("API 超时", source="api", risk_level="medium")

        stats = hub.process_archival()
        assert stats["archived"] == 2

        hub.create_snapshot("test_snapshot", state={"key": "value"})
        assert len(hub.list_snapshots()) >= 1

        hub.add_knowledge("doc_001", "超时重试 3 次", "errors_fixes")
        assert len(hub.rag_search("超时", top_k=5)) >= 1

        result = hub.execute_decision("关闭设置", "low")
        assert result["approved"] is True

        ctx_info = hub.get_context_info()
        assert ctx_info["max_tokens"] == 128_000
        hub.close()


def test_memoryhub_context_eviction():
    """测试上下文窗口淘汰机制。"""
    with _TempWorkspace() as workspace:
        config = Config(
            workspace=workspace,
            context_max_tokens=500,
            context_eviction_ratio=0.5,
        )
        hub = MemoryHub(config=config)

        long_text = "这是一段测试文本。" * 50
        hub.add_to_context(long_text, source="test")
        assert hub.get_context_info()["current_tokens"] <= 500

        hub.add_to_context("新增内容", source="test")
        info = hub.get_context_info()
        assert info["current_tokens"] <= 500
        # token 计数必须与窗口内容一致
        assert info["current_tokens"] == estimate_tokens(hub.context_window)

        cleared = hub.clear_context()
        assert cleared > 0
        assert hub.get_context_info()["current_tokens"] == 0
        hub.close()


def test_memoryhub_oversized_entry_truncated():
    """回归：单条内容超过窗口时截断，保证 token 上界成立。"""
    with _TempWorkspace() as workspace:
        config = Config(workspace=workspace, context_max_tokens=20)
        hub = MemoryHub(config=config)
        hub.add_to_context("中" * 200, source="test")

        info = hub.get_context_info()
        assert info["current_tokens"] <= 20
        assert "truncated" in hub.context_window
        hub.close()


def test_memoryhub_rag_category_stats():
    """测试 RAG 分类统计。"""
    with _TempWorkspace() as workspace:
        hub = MemoryHub(workspace=workspace)
        hub.add_knowledge("doc_001", "超时处理", "errors_fixes")
        hub.add_knowledge("doc_002", "设置页面", "ui_facts")
        hub.add_knowledge("doc_003", "网络错误", "errors_fixes")

        stats = hub.get_category_stats()
        assert stats.get("errors_fixes") == 2
        assert stats.get("ui_facts") == 1
        hub.close()


def test_memoryhub_cleanup():
    """测试完整整理。"""
    with _TempWorkspace() as workspace:
        hub = MemoryHub(workspace=workspace)
        hub.stage_action("测试", source="test")
        hub.create_snapshot("test_snapshot", state={"key": "value"})
        hub.add_knowledge("doc_001", "测试内容", "test_category")

        info = hub.full_cleanup()
        assert info["archival"]["archived"] == 1
        assert info["snapshots"] >= 1
        hub.close()


def test_memoryhub_snapshot_restores_context():
    """回归：清除上下文前的快照必须能恢复上下文原文。"""
    with _TempWorkspace() as workspace:
        hub = MemoryHub(workspace=workspace)
        hub.add_to_context("准备删除全部文件", source="user")
        hub.add_to_context("已确认备份完成", source="user")
        original = hub.context_window

        hub.clear_context()
        assert hub.get_context_info()["current_tokens"] == 0

        snapshots = [s for s in hub.list_snapshots() if s["name"] == "before_context_clear"]
        assert len(snapshots) == 1
        assert snapshots[0]["has_context"] is True

        state = hub.restore_snapshot("before_context_clear")
        assert state is not None
        assert hub.context_window == original
        assert hub.get_context_info()["current_tokens"] > 0
        hub.close()


def test_memoryhub_clear_all():
    """回归：clear_all 必须能删掉正在写入的 staging 文件（Windows 文件锁）。"""
    with _TempWorkspace() as workspace:
        hub = MemoryHub(workspace=workspace)
        hub.stage_action("测试", source="test")
        hub.add_knowledge("doc_001", "内容", "cat")
        hub.create_snapshot("snap", state={"k": 1})

        removed = hub.clear_all()
        assert removed >= 3
        assert hub.read_all_staging() == []
        assert hub.get_category_stats() == {}
        assert hub.list_snapshots() == []
        staging_files = [f for f in os.listdir(hub.staging.staging_dir)
                         if f.endswith(".jsonl")]
        assert staging_files == []
        hub.close()


def test_memoryhub_workspace_config_conflict():
    """workspace 与 config.workspace 冲突时应报错，而不是静默忽略。"""
    with _TempWorkspace() as workspace:
        config = Config(workspace=workspace + "_other")
        _assert_raises(ValueError, MemoryHub, workspace=workspace, config=config)


def test_memoryhub_context_persistence():
    """测试上下文状态跨实例持久化。"""
    with _TempWorkspace() as workspace:
        hub = MemoryHub(workspace=workspace)
        hub.add_to_context("需要记住的关键状态", source="user")
        hub.close()

        reopened = MemoryHub(workspace=workspace)
        assert "需要记住的关键状态" in reopened.context_window
        assert reopened.get_context_info()["current_tokens"] > 0
        reopened.close()


def test_memoryhub_enum_arguments():
    """枚举成员与字符串值都可以直接传入。"""
    with _TempWorkspace() as workspace:
        hub = MemoryHub(workspace=workspace)
        record = hub.staging.append("枚举参数", coarse_type=CoarseType.ERROR,
                                    risk_level=RiskLevel.HIGH,
                                    verify_status=VerifyStatus.PASS)
        assert record["coarse_type"] == "error"
        assert record["risk_level"] == "high"
        assert record["verify_status"] == "pass"

        result = hub.execute_decision("删除", RiskLevel.HIGH,
                                      thesis_reason="已备份，磁盘即将写满，不清理会导致服务不可用",
                                      antithesis_reason="风险")
        assert result["risk_level"] == "high"
        assert result["outcome"] == ReviewOutcome.APPROVED.value

        stats = hub.process_archival(category=ArchiveCategory.ERRORS_FIXES)
        assert stats["archived"] == 1
        assert "errors_fixes" in hub.list_archival_categories()
        hub.close()


# ============================================================
# 运行测试（内置 runner，不依赖 pytest）
# ============================================================

TEST_GROUPS = [
    ("工具函数测试", [
        test_dedup_id,
        test_format_timestamp,
        test_estimate_tokens,
        test_safe_json_loads,
        test_sanitize_filename,
        test_safe_stem,
    ]),
    ("配置测试", [
        test_config_valid,
        test_config_invalid_workspace,
        test_config_invalid_eviction_ratio,
        test_config_invalid_roll_hours,
        test_enum_values,
    ]),
    ("第一层：临时流水测试", [
        test_staging_basic,
        test_staging_hourly_roll,
        test_staging_crash_safe,
        test_staging_repair_truncated_tail,
        test_staging_restart_index,
    ]),
    ("第三层：快照测试", [
        test_snapshot_basic,
        test_snapshot_unsafe_name,
    ]),
    ("RAG 检索测试", [
        test_rag_add_and_search,
        test_rag_remove_document,
        test_rag_reload_rebuild,
        test_rag_doc_id_with_path_separator,
        test_rag_duplicate_add,
        test_rag_invalid_arguments,
    ]),
    ("第二层：分类归档测试", [
        test_archival_idempotent,
        test_archival_routing,
        test_archival_crash_replay,
        test_archival_invalid_category,
        test_archival_marker_only_for_touched_categories,
    ]),
    ("工作流与决策审查测试", [
        test_decision_review_high_risk_rejected,
        test_decision_review_high_risk_approved,
        test_decision_review_low_risk_auto_approve,
        test_decision_review_needs_more_info,
        test_workflow_manager,
        test_workflow_medium_review,
        test_behavior_reregister_updates_definition,
    ]),
    ("核心编排器测试", [
        test_memoryhub_full_flow,
        test_memoryhub_context_eviction,
        test_memoryhub_oversized_entry_truncated,
        test_memoryhub_rag_category_stats,
        test_memoryhub_cleanup,
        test_memoryhub_snapshot_restores_context,
        test_memoryhub_clear_all,
        test_memoryhub_workspace_config_conflict,
        test_memoryhub_context_persistence,
        test_memoryhub_enum_arguments,
    ]),
]


def main():
    """不依赖 pytest 的简单测试运行器。"""
    passed = 0
    failed = 0
    failures = []

    for group_name, test_list in TEST_GROUPS:
        print("\n" + "=" * 60)
        print(group_name)
        print("=" * 60)

        for test_func in test_list:
            try:
                test_func()
                print("  ✓ {0}".format(test_func.__name__))
                passed += 1
            except AssertionError as e:
                print("  ✗ {0}: {1}".format(test_func.__name__, e))
                failures.append((test_func.__name__, str(e)))
                failed += 1
            except Exception as e:
                print("  ✗ {0}: 发生异常 - {1}".format(test_func.__name__, e))
                failures.append((test_func.__name__, repr(e)))
                failed += 1

    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    print("通过: {0}".format(passed))
    print("失败: {0}".format(failed))
    print("总计: {0}".format(passed + failed))

    if failed == 0:
        print("\n✓ 所有测试通过！")
        return 0

    print("\n✗ {0} 个测试失败：".format(failed))
    for name, message in failures:
        print("  - {0}: {1}".format(name, message))
    return 1


if __name__ == "__main__":
    sys.exit(main())
