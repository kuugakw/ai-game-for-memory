"""工作流与决策管理模块。

设计要点：
  - 高风险决策走正反合审查（thesis-antithesis-synthesis）
  - 低风险决策直接放行，不浪费 token
  - 权重可计算可审计
  - 行为可复用可退场（behavior tri-state）

正反合审查流程：
  正（thesis）：支持该决策的理由
  反（antithesis）：反对该决策的理由
  合（synthesis）：综合正反而得出的最终结论

行为三态：
  - done：已完成的行为模式
  - in_progress：正在使用的行为模式
  - retired：已退场、不可再用的行为模式
"""

import json
import os
import uuid
from typing import Any, Dict, List, Optional

from .config import ReviewOutcome, RiskLevel, coerce_enum
from .utils import format_timestamp, safe_stem

# 行为三态的合法状态
BEHAVIOR_STATUSES = ("in_progress", "done", "retired")

# 各风险等级的审查阈值（risk_adjusted_score / antithesis_weight）
HIGH_RISK_SCORE_THRESHOLD = 0.3
MEDIUM_RISK_ANTITHESIS_THRESHOLD = 0.5

# 风险乘数：高风险时对反对理由更敏感
RISK_MULTIPLIER = {
    RiskLevel.LOW.value: 1.0,
    RiskLevel.MEDIUM.value: 1.2,
    RiskLevel.HIGH.value: 1.5,
}


def validate_risk_level(risk_level: str) -> str:
    """校验风险等级，返回字符串值；非法时抛 ValueError。

    同时接受 ``RiskLevel`` 枚举成员与 ``"high"`` 这样的字符串。
    """
    risk_level = coerce_enum(risk_level)
    valid = [r.value for r in RiskLevel]
    if risk_level not in valid:
        raise ValueError(
            "risk_level '{0}' 不是有效等级，可选值：{1}".format(risk_level, valid)
        )
    return risk_level


class DecisionReview:
    """单次决策审查记录。

    记录正反合审查的全过程，包括正反面理由和最终结论。
    支持权重计算和审计追踪。
    """

    def __init__(self, decision_id: str, decision_text: str, risk_level: str):
        """初始化决策审查记录。"""
        self.decision_id = decision_id
        self.decision_text = decision_text
        self.risk_level = validate_risk_level(risk_level)
        self.thesis: str = ""
        self.antithesis: str = ""
        self.synthesis: str = ""
        self.outcome: str = ReviewOutcome.NEEDS_MORE_INFO.value
        self.weights: Dict[str, Any] = {}
        self.timestamp: str = format_timestamp()
        self.approved: bool = False

    def add_thesis(self, reason: str):
        """添加支持该决策的理由（正）。"""
        self.thesis = reason or ""

    def add_antithesis(self, reason: str):
        """添加反对该决策的理由（反）。"""
        self.antithesis = reason or ""

    def add_synthesis(self, conclusion: str):
        """添加综合正反后的最终结论（合）。"""
        self.synthesis = conclusion or ""

    def _calculate_weights(self) -> Dict[str, Any]:
        """计算可审计的权重分数。

        risk_adjusted_score 语义：风险调整后的"反对强度"。
        - 反对理由占比越高，值越大；
        - 风险等级越高，乘数越大（高风险时对反对更敏感）；
        - 该值越高表示决策越危险，应拒绝。
        """
        thesis_len = len(self.thesis)
        antithesis_len = len(self.antithesis)
        total = thesis_len + antithesis_len

        if total == 0:
            return {
                "thesis_weight": 0.0,
                "antithesis_weight": 0.0,
                "risk_adjusted_score": 0.0,
                "risk_level": self.risk_level,
                "thesis_length": 0,
                "antithesis_length": 0,
            }

        thesis_weight = thesis_len / total
        antithesis_weight = antithesis_len / total
        risk_multiplier = RISK_MULTIPLIER.get(self.risk_level, 1.0)

        # 反对强度 = 反对理由占比 × 风险乘数
        risk_adjusted = antithesis_weight * risk_multiplier

        return {
            "thesis_weight": round(thesis_weight, 4),
            "antithesis_weight": round(antithesis_weight, 4),
            "risk_adjusted_score": round(risk_adjusted, 4),
            "risk_multiplier": risk_multiplier,
            "risk_level": self.risk_level,
            "thesis_length": thesis_len,
            "antithesis_length": antithesis_len,
        }

    def evaluate(self) -> str:
        """根据正反面理由评估最终结论。

        评估规则：
        - 低风险：无需审查，直接批准
        - 中风险：反对理由占比 ≤ 0.5 批准，否则拒绝
        - 高风险：反对强度（risk_adjusted_score）≤ 0.3 才批准，
                  即反对理由占比需低于约 20%（0.3 / 1.5）
        - 缺少正面或反面理由：needs_more_info（不做没有依据的判断）

        返回:
            审查结果（approved / rejected / needs_more_info）。
        """
        if self.risk_level == RiskLevel.LOW.value:
            self.outcome = ReviewOutcome.APPROVED.value
            self.approved = True
            return self.outcome

        if not self.thesis or not self.antithesis:
            self.outcome = ReviewOutcome.NEEDS_MORE_INFO.value
            self.approved = False
            self.weights = self._calculate_weights()
            return self.outcome

        self.weights = self._calculate_weights()

        if self.risk_level == RiskLevel.HIGH.value:
            risk_adjusted = self.weights.get("risk_adjusted_score", 0.0)
            self.weights["threshold"] = HIGH_RISK_SCORE_THRESHOLD
            if risk_adjusted <= HIGH_RISK_SCORE_THRESHOLD:
                self.outcome = ReviewOutcome.APPROVED.value
                self.approved = True
            else:
                self.outcome = ReviewOutcome.REJECTED.value
                self.approved = False
        elif self.risk_level == RiskLevel.MEDIUM.value:
            antithesis_weight = self.weights.get("antithesis_weight", 0.0)
            self.weights["threshold"] = MEDIUM_RISK_ANTITHESIS_THRESHOLD
            if antithesis_weight <= MEDIUM_RISK_ANTITHESIS_THRESHOLD:
                self.outcome = ReviewOutcome.APPROVED.value
                self.approved = True
            else:
                self.outcome = ReviewOutcome.REJECTED.value
                self.approved = False
        else:
            self.outcome = ReviewOutcome.NEEDS_MORE_INFO.value
            self.approved = False

        return self.outcome

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。"""
        return {
            "decision_id": self.decision_id,
            "decision_text": self.decision_text,
            "risk_level": self.risk_level,
            "thesis": self.thesis,
            "antithesis": self.antithesis,
            "synthesis": self.synthesis,
            "outcome": self.outcome,
            "weights": self.weights,
            "timestamp": self.timestamp,
            "approved": self.approved,
        }


class WorkflowManager:
    """Agent 工作流管理器。

    负责：
    - 决策审查（正反合）
    - 行为三态管理（done / in_progress / retired）
    - 权重可计算可审计
    """

    def __init__(self, workspace: str, review_high_risk_only: bool = True):
        """初始化 WorkflowManager。

        参数:
            workspace: 数据存储根目录路径。
            review_high_risk_only: 是否仅对高风险决策启动正反合审查。
        """
        self.workspace = workspace
        self.review_high_risk_only = review_high_risk_only
        self.workflow_dir = os.path.join(workspace, "workflow")
        self.behaviors_dir = os.path.join(self.workflow_dir, "behaviors")
        self.reviews_dir = os.path.join(self.workflow_dir, "reviews")
        os.makedirs(self.behaviors_dir, exist_ok=True)
        os.makedirs(self.reviews_dir, exist_ok=True)

        self._behaviors: Dict[str, Dict[str, Any]] = {}
        self._load_behaviors()

    # ---- 行为三态 ----

    def _behavior_file(self, behavior_id: str) -> str:
        """由 behavior_id 得到行为文件路径（ID 脱敏）。"""
        return os.path.join(self.behaviors_dir, "{0}.json".format(safe_stem(behavior_id, "behavior")))

    def _load_behaviors(self):
        """加载已保存的行为三态记录。"""
        if not os.path.exists(self.behaviors_dir):
            return

        for filename in sorted(os.listdir(self.behaviors_dir)):
            if not filename.endswith(".json"):
                continue
            file_path = os.path.join(self.behaviors_dir, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (IOError, OSError, json.JSONDecodeError) as e:
                print("[WorkflowManager] 加载行为失败 {0}: {1}".format(file_path, e))
                continue
            if not isinstance(data, dict):
                continue
            behavior_id = data.get("behavior_id") or filename[:-5]
            self._behaviors[behavior_id] = data

    def _save_behavior(self, behavior_id: str, data: Dict[str, Any]):
        """保存行为三态记录。"""
        file_path = self._behavior_file(behavior_id)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except (IOError, OSError) as e:
            print("[WorkflowManager] 保存行为失败 {0}: {1}".format(file_path, e))

    def register_behavior(self, behavior_id: str, pattern: str,
                          description: str, status: str = "in_progress") -> Dict[str, Any]:
        """注册一个行为模式（三态：done / in_progress / retired）。

        已存在时更新 pattern / description / status（而不是只改 status），
        否则修改行为定义后旧定义会一直留在文件里。
        """
        if not isinstance(behavior_id, str) or not behavior_id:
            raise ValueError("behavior_id 必须为非空字符串")
        if status not in BEHAVIOR_STATUSES:
            raise ValueError(
                "status '{0}' 不是有效值，可选值：{1}".format(status, list(BEHAVIOR_STATUSES))
            )

        now = format_timestamp()
        if behavior_id in self._behaviors:
            record = self._behaviors[behavior_id]
            record["pattern"] = pattern
            record["description"] = description
            record["status"] = status
            record["updated_at"] = now
        else:
            record = {
                "behavior_id": behavior_id,
                "pattern": pattern,
                "description": description,
                "status": status,
                "created_at": now,
                "updated_at": now,
            }
            self._behaviors[behavior_id] = record

        self._save_behavior(behavior_id, record)
        return dict(record)

    def update_behavior_status(self, behavior_id: str, new_status: str) -> bool:
        """更新某个行为的状态（三态切换）。"""
        if behavior_id not in self._behaviors:
            return False
        if new_status not in BEHAVIOR_STATUSES:
            raise ValueError(
                "status '{0}' 不是有效值，可选值：{1}".format(new_status, list(BEHAVIOR_STATUSES))
            )

        old_status = self._behaviors[behavior_id].get("status", "unknown")
        self._behaviors[behavior_id]["status"] = new_status
        self._behaviors[behavior_id]["updated_at"] = format_timestamp()
        self._save_behavior(behavior_id, self._behaviors[behavior_id])

        self._log_transition(behavior_id, old_status, new_status)
        return True

    def _log_transition(self, behavior_id: str, old_status: str, new_status: str):
        """记录行为状态转换（审计用）。"""
        transition_file = os.path.join(self.workflow_dir, "transitions.log")
        entry = {
            "timestamp": format_timestamp(),
            "behavior_id": behavior_id,
            "from": old_status,
            "to": new_status,
        }
        try:
            with open(transition_file, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except (IOError, OSError) as e:
            print("[WorkflowManager] 记录状态转换失败: {0}".format(e))

    def get_behavior(self, behavior_id: str) -> Optional[Dict[str, Any]]:
        """获取某个行为的信息（返回副本）。"""
        record = self._behaviors.get(behavior_id)
        return dict(record) if record is not None else None

    def list_behaviors(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出所有行为记录。"""
        records = list(self._behaviors.values())
        if status:
            records = [b for b in records if b.get("status") == status]
        return [dict(b) for b in records]

    def retire_behavior(self, behavior_id: str) -> bool:
        """将某个行为标记为 retired（退场）。"""
        return self.update_behavior_status(behavior_id, "retired")

    def activate_behavior(self, behavior_id: str) -> bool:
        """将某个行为标记为 in_progress（使用中）。"""
        return self.update_behavior_status(behavior_id, "in_progress")

    def complete_behavior(self, behavior_id: str) -> bool:
        """将某个行为标记为 done（已完成）。"""
        return self.update_behavior_status(behavior_id, "done")

    # ---- 决策审查 ----

    def create_decision_review(self, decision_text: str,
                               risk_level: str = "low") -> DecisionReview:
        """创建一个新的决策审查记录。"""
        validate_risk_level(risk_level)
        decision_id = "decision_{0}".format(uuid.uuid4().hex[:8])
        return DecisionReview(decision_id, decision_text, risk_level)

    def should_review(self, risk_level: str) -> bool:
        """判断是否需要对某个风险等级的决策启动审查。

        返回 False 表示"当前配置下无需审查，应直接放行"。
        """
        validate_risk_level(risk_level)
        if self.review_high_risk_only:
            return risk_level == RiskLevel.HIGH.value
        return risk_level in (RiskLevel.HIGH.value, RiskLevel.MEDIUM.value)

    def _save_review(self, review: DecisionReview):
        """保存决策审查记录到文件。"""
        file_path = os.path.join(
            self.reviews_dir, "{0}.json".format(safe_stem(review.decision_id, "decision"))
        )
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(review.to_dict(), f, ensure_ascii=False, indent=2)
        except (IOError, OSError) as e:
            print("[WorkflowManager] 保存审查记录失败 {0}: {1}".format(file_path, e))

    def _auto_approve(self, decision_text: str, risk_level: str) -> Dict[str, Any]:
        """对当前配置下无需审查的决策直接放行。"""
        risk_level = validate_risk_level(risk_level)
        review = self.create_decision_review(decision_text, risk_level)
        review.outcome = ReviewOutcome.APPROVED.value
        review.approved = True
        review.synthesis = "auto_approved: {0} 风险决策在当前配置下无需审查".format(risk_level)
        review.weights = {
            "thesis_weight": 1.0,
            "antithesis_weight": 0.0,
            "risk_adjusted_score": 0.0,
            "risk_level": risk_level,
            "auto_approved": True,
        }
        self._save_review(review)

        reason = "{0} 风险决策在当前配置下无需审查，自动放行".format(risk_level)
        return {
            "decision_text": decision_text,
            "risk_level": risk_level,
            "reviewed": False,
            "approved": True,
            "outcome": ReviewOutcome.APPROVED.value,
            "reason": reason,
            "weights": review.weights,
            "synthesis": review.synthesis,
            "decision_id": review.decision_id,
        }

    def auto_approve_low_risk(self, decision_text: str,
                              risk_level: str) -> Dict[str, Any]:
        """兼容旧接口：对无需审查的决策直接放行。"""
        return self._auto_approve(decision_text, risk_level)

    def _build_synthesis(self, review: DecisionReview) -> str:
        """根据审查结果生成"合"的结论文本（审计留痕）。"""
        if review.outcome == ReviewOutcome.NEEDS_MORE_INFO.value:
            return ("缺少正面和/或反面理由，无法完成正反合审查；"
                    "请补充 thesis_reason 与 antithesis_reason 后重试。")
        weights = review.weights
        if review.risk_level == RiskLevel.HIGH.value:
            detail = "risk_adjusted_score={0} / 阈值 {1}".format(
                weights.get("risk_adjusted_score"), weights.get("threshold"))
        else:
            detail = "antithesis_weight={0} / 阈值 {1}".format(
                weights.get("antithesis_weight"), weights.get("threshold"))
        return "综合正反面理由：{0}（{1}）".format(review.outcome, detail)

    def execute_decision(self, decision_text: str,
                         risk_level: str,
                         thesis_reason: Optional[str] = None,
                         antithesis_reason: Optional[str] = None) -> Dict[str, Any]:
        """执行一个决策（带审查或自动放行）。

        参数:
            decision_text: 决策内容。
            risk_level: 风险等级（low / medium / high）。
            thesis_reason: 支持理由。需要审查时必须提供。
            antithesis_reason: 反对理由。需要审查时必须提供。

        返回:
            执行结果字典（approved / outcome / weights / reason 等）。

        注意：需要审查但没有给出正反面理由时返回 ``needs_more_info``，
        不会用编造的默认理由替调用方做决定。
        """
        risk_level = validate_risk_level(risk_level)
        if not isinstance(decision_text, str) or not decision_text:
            raise ValueError("decision_text 必须为非空字符串")

        # 当前配置下无需审查的决策直接放行
        if not self.should_review(risk_level):
            return self._auto_approve(decision_text, risk_level)

        # 需要审查：启动正反合流程
        review = self.create_decision_review(decision_text, risk_level)
        if thesis_reason:
            review.add_thesis(thesis_reason)
        if antithesis_reason:
            review.add_antithesis(antithesis_reason)

        outcome = review.evaluate()
        review.add_synthesis(self._build_synthesis(review))
        self._save_review(review)

        return {
            "decision_text": decision_text,
            "risk_level": risk_level,
            "reviewed": True,
            "approved": review.approved,
            "outcome": outcome,
            "reason": review.synthesis,
            "weights": review.weights,
            "thesis": review.thesis,
            "antithesis": review.antithesis,
            "synthesis": review.synthesis,
            "decision_id": review.decision_id,
        }

    # ---- 清理 ----

    def clear_all(self) -> int:
        """清空所有工作流数据（行为、审查记录、转换日志）。"""
        count = 0
        for directory in (self.behaviors_dir, self.reviews_dir, self.workflow_dir):
            if not os.path.exists(directory):
                continue
            for filename in os.listdir(directory):
                if filename.endswith(".json") or filename.endswith(".log"):
                    try:
                        os.remove(os.path.join(directory, filename))
                        count += 1
                    except (IOError, OSError):
                        pass
        self._behaviors.clear()
        return count

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
