"""工具函数模块。

提供哈希去重 ID 生成、时间戳格式化、token 估算等通用功能。
所有函数均为纯函数，不依赖文件系统或网络。
"""

import hashlib
import json
import re
from datetime import datetime
from typing import Any, Optional


def dedup_id(source_file: str, entry_index: int) -> str:
    """生成去重 ID：由"来源文件 + 序号"组合哈希得到。

    用于第二层归档的幂等去重。崩溃重放时，通过该 ID 判断某条流水
    是否已经被处理过，避免重复归档。

    注意：``source_file`` 应当是 **相对于 workspace 的路径**，而不是绝对路径。
    使用绝对路径会让数据目录一旦移动（换机器、换盘符），同一批流水就
    生成不同的 ID，从而被重复归档。

    参数:
        source_file: 来源文件路径（相对 workspace 的路径或任意稳定标识符）。
        entry_index: 该条目在来源文件中的行序号（从 0 开始）。

    返回:
        40 位十六进制 SHA256 前缀，作为去重 ID。
    """
    raw = f"{source_file}\n{entry_index}\n"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:40]


def format_timestamp(dt: Optional[datetime] = None, iso: bool = True) -> str:
    """格式化时间戳。

    参数:
        dt: 待格式化的 datetime 对象，默认为当前时间。
            naive datetime 会被视为本地时间。
        iso: 是否为 ISO 8601 格式；False 则为人读格式。

    返回:
        时间字符串。ISO 格式带时区偏移，例如 ``2026-01-07T12:30:45+08:00``。
    """
    if dt is None:
        dt = datetime.now().astimezone()
    elif dt.tzinfo is None:
        # naive datetime 视为本地时间，避免把本地时间误标成 UTC
        dt = dt.astimezone()
    if iso:
        return dt.isoformat()
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def estimate_tokens(text: str) -> int:
    """粗略估算文本的 token 数量。

    使用字符数除以平均 token 密度的启发式方法：
    中文约 1.5 字符/token，英文约 4 字符/token。
    此函数用于上下文窗口管理，不要求精确。

    参数:
        text: 待估算的文本。

    返回:
        估算的 token 数；空文本返回 0。
    """
    if not text:
        return 0
    chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    other_chars = len(text) - chinese_chars
    tokens = chinese_chars / 1.5 + other_chars / 4.0
    return max(1, int(tokens))


def safe_json_loads(line: str) -> Any:
    """安全地 JSON 反序列化一行，失败返回 None。

    用于第一层流水读取，崩溃时可能最后一行不完整。
    """
    try:
        return json.loads(line)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def sanitize_filename(name: str) -> str:
    """将字符串安全地转换为文件名片段（移除非法字符）。

    只做字符替换，不保证结果唯一、也允许返回空字符串；
    需要稳定且唯一的文件名时请使用 :func:`safe_stem`。
    """
    return re.sub(r'[\s/\\:*?"<>|]', "_", name).strip()


def safe_stem(name: str, fallback: str = "item") -> str:
    """把任意标识符转换成稳定、安全、可跨平台的文件名主干。

    - 只保留 ``[a-zA-Z0-9_-]``，其余字符替换为 ``_``；
    - 如果发生了替换（或结果为空），追加原始名称的哈希后缀，
      避免 ``a/b`` 与 ``a:b`` 这类不同 ID 落到同一个文件上；
    - 同一输入永远得到同一输出（提供哈希而非随机数）。

    参数:
        name: 原始标识符，例如 doc_id / behavior_id / 快照名。
        fallback: 名称被完全清空时使用的前缀。

    返回:
        适合直接拼进文件名的字符串。
    """
    raw = name if isinstance(name, str) else str(name)
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", raw).strip("_")
    if safe and safe == raw:
        return safe
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    if safe:
        return f"{safe}_{digest}"
    return f"{fallback}_{digest}"


__all__ = [
    "dedup_id",
    "format_timestamp",
    "estimate_tokens",
    "safe_json_loads",
    "sanitize_filename",
    "safe_stem",
]
