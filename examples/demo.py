#!/usr/bin/env python3
"""AI Game for Memory 插件使用示例。

本示例演示分层记忆架构的完整流程：
  1. 第一层：临时流水（staging）
  2. 第二层：分类归档（archival）
  3. 第三层：精确状态快照（snapshot）
  4. RAG 知识检索
  5. 工作流决策审查（正反合）
  6. 行为三态管理
  7. 上下文窗口管理
  8. 完整整理（full cleanup）
  9. pandas 数据导出（CSV 导出）

运行方式：
    python examples/demo.py
"""

import os
import sys
import tempfile
import shutil

# 将插件根目录加入 Python 路径
PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PLUGIN_ROOT)

from ai_game_for_memory import MemoryHub, Config, RiskLevel


def main():
    """演示完整流程。"""
    # 使用临时目录作为 workspace，避免污染用户文件
    workspace = tempfile.mkdtemp(prefix="ai_game_for_memory_demo_")
    print("使用临时工作区: {0}\n".format(workspace))

    hub = None
    try:
        # 使用自定义配置
        config = Config(
            workspace=workspace,
            context_max_tokens=10000,  # 演示用较小的值
            context_eviction_ratio=0.5,
            retention_days=7,
            review_high_risk_only=True,
        )
        hub = MemoryHub(config=config)

        # ============================================
        # 1. 第一层：临时流水
        # ============================================
        print("=" * 60)
        print("1. 第一层：临时流水 (Staging)")
        print("=" * 60)

        hub.stage_action("点击设置菜单", source="gui", risk_level="low")
        hub.stage_action("输入用户名: admin", source="gui", risk_level="low")
        hub.stage_decision("关闭设置", source="gui", risk_level="low")
        hub.stage_error("API 超时", source="api", risk_level="medium")
        hub.stage_reasoning("超时后应重试 3 次", source="agent", risk_level="low")
        hub.stage_screenshot("设置页面截图摘要", source="gui", risk_level="low")
        hub.stage_state("当前阶段: 配置管理", source="agent", risk_level="low")

        staging_records = hub.read_all_staging()
        print(f"已写入 {len(staging_records)} 条临时流水记录\n")

        # ============================================
        # 2. 第二层：分类归档
        # ============================================
        print("=" * 60)
        print("2. 第二层：分类归档 (Archival)")
        print("=" * 60)

        stats = hub.process_archival()
        print(f"整理统计: {stats}")
        print(f"归档分类: {hub.list_archival_categories()}")
        print(f"归档总条目数: {hub.get_archived_count()}\n")

        # ============================================
        # 3. 第三层：精确状态快照
        # ============================================
        print("=" * 60)
        print("3. 第三层：精确状态快照 (Snapshot)")
        print("=" * 60)

        # 不可逆操作前创建快照
        snapshot_info = hub.create_snapshot(
            "before_delete_all",
            state={"files_to_delete": ["temp1.txt", "temp2.txt"]},
            include_context=True
        )
        print(f"快照已创建: {snapshot_info['snapshot_id']}")

        # 列出所有快照
        snapshots = hub.list_snapshots()
        print(f"当前快照数量: {len(snapshots)}\n")

        # ============================================
        # 4. RAG 知识检索
        # ============================================
        print("=" * 60)
        print("4. RAG 知识检索")
        print("=" * 60)

        # 添加知识文档
        hub.add_knowledge(
            doc_id="err_timeout_001",
            text="API 超时时先重试 3 次，记录错误日志，仍失败则通知用户",
            category="errors_fixes"
        )
        hub.add_knowledge(
            doc_id="ui_settings_001",
            text="设置页面包含用户信息、安全设置、通知偏好三个选项卡",
            category="ui_facts"
        )
        hub.add_knowledge(
            doc_id="behavior_retry_001",
            text="处理超时的标准行为：重试 3 次 → 记录错误 → 通知用户",
            category="behavior_tristate"
        )

        # 混合检索
        results = hub.rag_search("超时处理", mode="hybrid", top_k=5)
        print("查询 '超时处理' 的检索结果:")
        for r in results:
            print(f"  - [{r['category']}] (score: {r['score']:.4f}) {r['text'][:40]}...")
        print()

        # 按类别过滤检索
        err_results = hub.rag_search("超时", category="errors_fixes", top_k=3)
        print(f"仅检索 errors_fixes 分类: {len(err_results)} 条结果\n")

        # ============================================
        # 5. 工作流决策审查（正反合）
        # ============================================
        print("=" * 60)
        print("5. 工作流决策审查 (Thesis-Antithesis-Synthesis)")
        print("=" * 60)

        # 低风险决策：直接放行
        low_result = hub.execute_decision(
            decision_text="关闭设置页面",
            risk_level="low"
        )
        print(f"低风险决策: {low_result['decision_text']}")
        print(f"  已审查: {low_result['reviewed']}, 已批准: {low_result['approved']}")
        print(f"  原因: {low_result['reason']}\n")

        # 高风险决策：触发正反合审查
        high_result = hub.execute_decision(
            decision_text="删除所有临时文件",
            risk_level=RiskLevel.HIGH.value,
            thesis_reason="释放磁盘空间，避免磁盘满载，当前磁盘使用率已达 90%",
            antithesis_reason="可能会丢失未归档的流水记录，需要确认整理已完成且快照已创建"
        )
        print(f"高风险决策: {high_result['decision_text']}")
        print(f"  已审查: {high_result['reviewed']}, 已批准: {high_result['approved']}")
        print(f"  结果: {high_result['outcome']}")
        print(f"  权重: {high_result['weights']}")
        print(f"  正 (Thesis): {high_result['thesis']}")
        print(f"  反 (Antithesis): {high_result['antithesis']}")
        print(f"  合 (Synthesis): {high_result['synthesis']}\n")

        # 需要审查但没给理由：返回 needs_more_info，不替调用方编造理由
        missing_result = hub.execute_decision(
            decision_text="删除所有临时文件",
            risk_level="high"
        )
        print(f"高风险决策（未提供理由）: {missing_result['outcome']}")
        print(f"  原因: {missing_result['reason']}\n")

        # ============================================
        # 6. 行为三态管理
        # ============================================
        print("=" * 60)
        print("6. 行为三态管理 (Done / In Progress / Retired)")
        print("=" * 60)

        # 注册行为模式
        hub.register_behavior(
            behavior_id="handle_timeout",
            pattern="重试 3 次 → 记录错误 → 通知用户",
            description="处理 API 超时的标准行为模式",
            status="in_progress"
        )
        print("已注册行为: handle_timeout (in_progress)")

        # 标记为已完成
        hub.complete_behavior("handle_timeout")
        print("已标记为完成: handle_timeout (done)")

        # 注册新的行为模式
        hub.register_behavior(
            behavior_id="handle_network_error",
            pattern="记录错误 → 通知用户 → 重试 1 次",
            description="处理网络错误的标准行为模式",
            status="in_progress"
        )
        print("已注册行为: handle_network_error (in_progress)")

        # 列出所有 in_progress 的行为
        in_progress = hub.list_behaviors(status="in_progress")
        print(f"\n当前 in_progress 的行为: {[b['behavior_id'] for b in in_progress]}")

        # 退场旧行为
        hub.retire_behavior("handle_timeout")
        print("已退场行为: handle_timeout (retired)\n")

        # ============================================
        # 7. 上下文窗口管理
        # ============================================
        print("=" * 60)
        print("7. 上下文窗口管理")
        print("=" * 60)

        # 添加内容到上下文
        hub.add_to_context("用户询问如何处理 API 超时问题", source="user")
        hub.add_to_context("模型回答：建议先重试 3 次，记录错误日志", source="assistant")
        hub.add_to_context("用户表示已经重试过，但仍然失败", source="user")

        # 查看上下文信息
        ctx_info = hub.get_context_info()
        print(f"上下文信息: {ctx_info}")
        print(f"当前上下文 token 数: {ctx_info['current_tokens']}")
        print(f"上限: {ctx_info['max_tokens']}")
        print(f"使用率: {ctx_info['usage_ratio']:.2%}\n")

        # 清除上下文（自动创建快照）
        cleared_tokens = hub.clear_context()
        print(f"已清除上下文，释放 {cleared_tokens} tokens\n")

        # ============================================
        # 8. 完整整理
        # ============================================
        print("=" * 60)
        print("8. 完整整理 (Full Cleanup)")
        print("=" * 60)

        info = hub.full_cleanup()
        print("整理统计:")
        print(f"  归档: {info['archival']}")
        print(f"  清理过期文件: {info['expired_staging_removed']}")
        print(f"  归档分类: {info['archival_categories']}")
        print(f"  RAG 分类统计: {info['rag_category_stats']}")
        print(f"  快照数量: {info['snapshots']}\n")

        # ============================================
        # 9. pandas 数据导出
        # ============================================
        print("=" * 60)
        print("9. pandas 数据导出")
        print("=" * 60)

        # 导出 RAG 文档为 CSV
        csv_path = os.path.join(workspace, "rag_docs.csv")
        success = hub.export_rag_to_csv(csv_path)
        if success:
            print(f"RAG 文档已导出到: {csv_path}")
            # 显示文件大小
            if os.path.exists(csv_path):
                size = os.path.getsize(csv_path)
                print(f"文件大小: {size} bytes\n")
        else:
            print("pandas 未安装，跳过 CSV 导出\n")

        # 导出临时流水为 CSV
        staging_csv_path = os.path.join(workspace, "staging_records.csv")
        success = hub.export_staging_to_csv(staging_csv_path)
        if success:
            print(f"临时流水记录已导出到: {staging_csv_path}")
            if os.path.exists(staging_csv_path):
                size = os.path.getsize(staging_csv_path)
                print(f"文件大小: {size} bytes\n")

        # 导出为 DataFrame
        df = hub.export_to_dataframe()
        if df is not None:
            print(f"DataFrame 包含 {len(df)} 条 RAG 文档记录")
            print(f"列名: {list(df.columns)}\n")

        # ============================================
        # 总结
        # ============================================
        print("=" * 60)
        print("总结")
        print("=" * 60)
        print(f"工作区: {workspace}")
        print(f"临时流水记录数: {len(staging_records)}")
        print(f"归档分类数: {len(hub.list_archival_categories())}")
        print(f"RAG 知识文档数: {sum(hub.get_category_stats().values())}")
        print(f"快照数量: {len(hub.list_snapshots())}")
        print(f"上下文当前 token 数: {hub.get_context_info()['current_tokens']}")
        print(f"行为模式数: {len(hub.list_behaviors())}")

    finally:
        # 先关闭句柄再删目录，否则 Windows 上 staging 文件被占用、目录删不掉
        if hub is not None:
            hub.close()
        shutil.rmtree(workspace, ignore_errors=True)
        print("\n临时工作区已清理。")


if __name__ == "__main__":
    main()
