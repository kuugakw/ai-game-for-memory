"""知识检索层 (RAG)。

设计要点：
  - 知识可以按需检索，检索不到可以重新勘察
  - 所有记忆塞进一个大索引会导致版本之间互相干扰
  - 因此 RAG 只检索知识分类，不检索状态/行为流水
  - 基础实现使用纯 Python 标准库；安装了 numpy / jieba 后自动启用优化

检索策略：
  - BM25 关键词匹配（主）+ 余弦相似度向量匹配（辅）
  - 混合检索（hybrid）默认开启：0.6 × 归一化 BM25 + 0.4 × 余弦相似度
  - 支持按类别过滤

索引与一致性：
  - 文档 JSON（``rag_index/documents/*.json``）是唯一事实来源
  - BM25 词项索引在每次启动时由文档重建，不再单独持久化，
    避免"索引文件与文档不同步"导致检索结果诡异
  - 向量随文档一起持久化，重建时直接复用

向量检索实现：
  - 使用 hashing trick（zlib.crc32 稳定哈希）将 token 映射到固定维度
  - 向量为 TF 的 L2 归一化结果，query 与 document 共享同一向量空间
  - 用户可通过 add_document(vector=...) 传入外部 embedding
  - 这是"离线也能跑"的词袋哈希向量，不是真正的语义 embedding

与快照的区别：
  - 快照 = 精确状态恢复（全量 dump）
  - RAG = 知识按需检索（语义/关键词匹配）
"""

import json
import math
import os
import re
import zlib
from collections import defaultdict
from typing import Any, Dict, List, Optional

from .utils import safe_stem

# 可选依赖：安装后自动启用优化，未安装时回退到纯标准库实现
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover - 取决于运行环境
    _HAS_NUMPY = False

try:
    import jieba
    _HAS_JIEBA = True
except ImportError:  # pragma: no cover - 取决于运行环境
    _HAS_JIEBA = False


class RAGEngine:
    """本地 RAG 检索引擎。

    使用 BM25 关键词匹配 + 余弦相似度向量进行混合检索。
    向量使用 hashing trick 本地生成，无需外部 embedding 服务。
    """

    # 本地哈希向量维度
    VECTOR_DIM = 128
    # BM25 参数
    BM25_K1 = 1.5
    BM25_B = 0.75

    def __init__(self, workspace: str, top_k: int = 5):
        """初始化 RAGEngine。

        参数:
            workspace: 数据存储根目录路径。
            top_k: 默认返回结果条数。
        """
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k 必须为正整数")
        self.workspace = workspace
        self.top_k = top_k
        self.index_dir = os.path.join(workspace, "rag_index")
        self.documents_dir = os.path.join(self.index_dir, "documents")
        os.makedirs(self.documents_dir, exist_ok=True)

        # 内存索引（启动时由文档重建）
        self._bm25_index: Dict[str, Dict[str, int]] = {}
        self._doc_term_freq: Dict[str, Dict[str, int]] = {}
        self._doc_lengths: Dict[str, int] = {}
        self._doc_texts: Dict[str, str] = {}
        self._doc_categories: Dict[str, str] = {}
        self._doc_vectors: Dict[str, List[float]] = {}
        self._avg_doc_length: float = 0.0
        self._num_docs: int = 0

        self.reload()

    # ---- 索引构建 ----

    def reload(self):
        """重新从文件加载全部文档并重建索引。"""
        self._clear_index()

        if not os.path.isdir(self.documents_dir):
            return

        for filename in sorted(os.listdir(self.documents_dir)):
            if not filename.endswith(".json"):
                continue
            doc_path = os.path.join(self.documents_dir, filename)
            try:
                with open(doc_path, "r", encoding="utf-8") as f:
                    doc_data = json.load(f)
            except (IOError, OSError, json.JSONDecodeError) as e:
                print("[RAGEngine] 加载文档失败 {0}: {1}".format(doc_path, e))
                continue
            if not isinstance(doc_data, dict):
                continue

            doc_id = doc_data.get("doc_id") or filename[:-5]
            text = doc_data.get("text", "")
            if not isinstance(text, str):
                continue
            self._doc_texts[doc_id] = text
            self._doc_categories[doc_id] = doc_data.get("category", "unknown")

            stored_vec = doc_data.get("vector")
            if isinstance(stored_vec, list) and stored_vec:
                self._doc_vectors[doc_id] = [float(v) for v in stored_vec]
            else:
                self._doc_vectors[doc_id] = self._build_vector(text)

        self._rebuild_index()

    def _rebuild_index(self):
        """根据内存中的文档重建 BM25 索引与统计量。"""
        self._bm25_index = {}
        self._doc_term_freq = {}
        self._doc_lengths = {}

        for doc_id, text in self._doc_texts.items():
            tokens = self._tokenize(text)
            freq: Dict[str, int] = defaultdict(int)
            for token in tokens:
                freq[token] += 1
            self._doc_term_freq[doc_id] = dict(freq)
            self._doc_lengths[doc_id] = len(tokens)
            for term, count in freq.items():
                self._bm25_index.setdefault(term, {})[doc_id] = count

        self._num_docs = len(self._doc_texts)
        total_length = sum(self._doc_lengths.values())
        self._avg_doc_length = (total_length / self._num_docs) if self._num_docs else 0.0

    def _clear_index(self):
        """清空内存中的索引。"""
        self._bm25_index = {}
        self._doc_term_freq = {}
        self._doc_lengths = {}
        self._doc_texts = {}
        self._doc_categories = {}
        self._doc_vectors = {}
        self._avg_doc_length = 0.0
        self._num_docs = 0

    # ---- 分词与向量 ----

    @staticmethod
    def _is_useful_token(token: str) -> bool:
        """过滤纯空白 / 纯标点 token（jieba 会切出这类 token）。"""
        return bool(token) and not re.fullmatch(r"[\W_]+", token, flags=re.UNICODE)

    def _tokenize(self, text: str) -> List[str]:
        """将文本分词为关键词列表。

        安装 jieba 后使用中文分词；否则回退到"英文按词、中文按字"的切分。
        """
        if not text or not isinstance(text, str):
            return []

        if _HAS_JIEBA:
            try:
                tokens = list(jieba.cut(text))
                return [t.strip().lower() for t in tokens
                        if self._is_useful_token(t.strip())]
            except Exception as e:  # pragma: no cover - 分词库异常时兜底
                print("[RAGEngine] jieba 分词失败，回退到字符级切分: {0}".format(e))

        tokens: List[str] = []
        current_word = ""
        for char in text:
            if "\u4e00" <= char <= "\u9fff":
                if current_word:
                    tokens.append(current_word)
                    current_word = ""
                tokens.append(char)
            elif char.isalnum() or char == ".":
                current_word += char
            else:
                if current_word:
                    tokens.append(current_word)
                    current_word = ""
        if current_word:
            tokens.append(current_word)
        return [t.lower() for t in tokens if self._is_useful_token(t)]

    def _build_vector(self, text: str) -> List[float]:
        """用 hashing trick 将文本映射为固定维度向量（TF + L2 归一化）。

        使用 zlib.crc32 稳定哈希（不受 PYTHONHASHSEED 影响），
        保证跨进程、跨会话的 query 与 document 向量在同一空间可比较。
        """
        vec = [0.0] * self.VECTOR_DIM
        tokens = self._tokenize(text)
        if not tokens:
            return vec

        tf: Dict[str, int] = defaultdict(int)
        for token in tokens:
            tf[token] += 1

        for token, freq in tf.items():
            idx = zlib.crc32(token.encode("utf-8")) % self.VECTOR_DIM
            vec[idx] += float(freq)

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    # ---- 文档增删 ----

    def _doc_file_path(self, doc_id: str) -> str:
        """由 doc_id 得到文档文件路径。

        doc_id 可能包含 ``/``、空格、非 ASCII 等不适合直接做文件名的字符，
        这里统一脱敏；发生替换时追加哈希后缀，避免两个不同 doc_id
        落到同一个文件上互相覆盖。
        """
        return os.path.join(self.documents_dir, "{0}.json".format(safe_stem(doc_id, "doc")))

    def add_document(self, doc_id: str, text: str, category: str,
                     vector: Optional[List[float]] = None) -> bool:
        """添加（或覆盖）一个文档到索引。

        参数:
            doc_id: 文档唯一标识。
            text: 文档文本内容。
            category: 文档类别（用于过滤检索）。
            vector: 可选的向量表示；不传则用本地 hashing trick 生成。
                    外部向量必须与已有向量同维度，否则结果不可比。

        返回:
            成功返回 True，失败返回 False。
        """
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError("doc_id 必须为非空字符串")
        if not isinstance(text, str):
            raise TypeError("text 必须为字符串")
        if not isinstance(category, str) or not category:
            raise ValueError("category 必须为非空字符串")

        if vector is not None:
            if not isinstance(vector, list) or not vector:
                raise ValueError("vector 必须为非空数字列表")
            try:
                final_vec = [float(v) for v in vector]
            except (TypeError, ValueError):
                raise ValueError("vector 必须为非空数字列表")
        else:
            final_vec = self._build_vector(text)

        doc_file = self._doc_file_path(doc_id)
        doc_data = {
            "doc_id": doc_id,
            "text": text,
            "category": category,
            "vector": final_vec,
        }
        try:
            with open(doc_file, "w", encoding="utf-8") as f:
                json.dump(doc_data, f, ensure_ascii=False, indent=2)
        except (IOError, OSError) as e:
            print("[RAGEngine] 保存文档失败 {0}: {1}".format(doc_file, e))
            return False

        # 覆盖已有文档时先摘掉旧词项，避免脏数据残留
        old_freq = self._doc_term_freq.pop(doc_id, None)
        if old_freq is not None:
            for term in old_freq:
                postings = self._bm25_index.get(term)
                if not postings:
                    continue
                postings.pop(doc_id, None)
                if not postings:
                    del self._bm25_index[term]

        tokens = self._tokenize(text)
        freq: Dict[str, int] = defaultdict(int)
        for token in tokens:
            freq[token] += 1

        self._doc_texts[doc_id] = text
        self._doc_categories[doc_id] = category
        self._doc_vectors[doc_id] = final_vec
        self._doc_lengths[doc_id] = len(tokens)
        self._doc_term_freq[doc_id] = dict(freq)
        for term, count in freq.items():
            self._bm25_index.setdefault(term, {})[doc_id] = count

        self._num_docs = len(self._doc_texts)
        total_length = sum(self._doc_lengths.values())
        self._avg_doc_length = (total_length / self._num_docs) if self._num_docs else 0.0
        return True

    def remove_document(self, doc_id: str) -> bool:
        """从索引中移除一个文档。"""
        if doc_id not in self._doc_texts:
            return False

        doc_file = self._doc_file_path(doc_id)
        if os.path.exists(doc_file):
            try:
                os.remove(doc_file)
            except (IOError, OSError) as e:
                print("[RAGEngine] 删除文档失败 {0}: {1}".format(doc_file, e))

        for term in self._doc_term_freq.get(doc_id, {}):
            postings = self._bm25_index.get(term)
            if not postings:
                continue
            postings.pop(doc_id, None)
            if not postings:
                del self._bm25_index[term]

        self._doc_term_freq.pop(doc_id, None)
        self._doc_texts.pop(doc_id, None)
        self._doc_lengths.pop(doc_id, None)
        self._doc_categories.pop(doc_id, None)
        self._doc_vectors.pop(doc_id, None)

        self._num_docs = len(self._doc_texts)
        total_length = sum(self._doc_lengths.values())
        self._avg_doc_length = (total_length / self._num_docs) if self._num_docs else 0.0
        return True

    # ---- 检索 ----

    def _bm25_score(self, query_tokens: List[str], doc_id: str) -> float:
        """计算单个文档的 BM25 得分。"""
        if not query_tokens:
            return 0.0

        doc_length = self._doc_lengths.get(doc_id, 0)
        tf = self._doc_term_freq.get(doc_id)
        if tf is None or doc_length == 0:
            return 0.0

        avg_length = self._avg_doc_length or 1.0
        length_factor = doc_length / avg_length
        score = 0.0
        for q_token in query_tokens:
            tf_val = tf.get(q_token, 0)
            if tf_val == 0:
                continue
            doc_freq = len(self._bm25_index.get(q_token, {}))
            idf = math.log(max(1.0, float(self._num_docs) / (doc_freq + 1))) + 1.0
            tf_factor = (tf_val * (self.BM25_K1 + 1)) / (
                tf_val + self.BM25_K1 * (1 - self.BM25_B + self.BM25_B * length_factor)
            )
            score += idf * tf_factor
        return score

    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """计算两个向量的余弦相似度；维度不同时按较短长度截断。"""
        if not vec_a or not vec_b:
            return 0.0
        if len(vec_a) != len(vec_b):
            min_len = min(len(vec_a), len(vec_b))
            vec_a = vec_a[:min_len]
            vec_b = vec_b[:min_len]

        if _HAS_NUMPY:
            try:
                a = np.array(vec_a, dtype=np.float64)
                b = np.array(vec_b, dtype=np.float64)
                mag_a = float(np.linalg.norm(a))
                mag_b = float(np.linalg.norm(b))
                if mag_a == 0 or mag_b == 0:
                    return 0.0
                return float(np.dot(a, b)) / (mag_a * mag_b)
            except Exception as e:  # pragma: no cover - numpy 异常时兜底
                print("[RAGEngine] numpy 余弦计算失败，回退到纯 Python: {0}".format(e))

        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        mag_a = math.sqrt(sum(a * a for a in vec_a))
        mag_b = math.sqrt(sum(b * b for b in vec_b))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot_product / (mag_a * mag_b)

    def search(self, query: str, top_k: Optional[int] = None,
               category: Optional[str] = None,
               mode: str = "hybrid") -> List[Dict[str, Any]]:
        """检索相关文档。

        参数:
            query: 查询文本。
            top_k: 返回结果条数，默认为初始化时设置的值。
            category: 可选的类别过滤。
            mode: 检索模式："bm25" / "vector" / "hybrid"。

        返回:
            按相关性排序的结果列表。
        """
        if mode not in ("bm25", "vector", "hybrid"):
            raise ValueError("mode 必须是 'bm25' / 'vector' / 'hybrid' 之一，收到：{0}".format(mode))
        if not isinstance(query, str) or not query.strip():
            return []

        if top_k is None:
            top_k = self.top_k
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k 必须为正整数")

        candidates = self._doc_texts
        if category is not None:
            candidates = {k: v for k, v in candidates.items()
                          if self._doc_categories.get(k, "") == category}
        if not candidates:
            return []

        query_tokens = self._tokenize(query)
        query_vec = (self._build_vector(query)
                     if mode in ("vector", "hybrid") else None)

        # 一次性预计算 BM25，避免在循环里重复分词
        bm25_scores = {doc_id: self._bm25_score(query_tokens, doc_id)
                       for doc_id in candidates}
        max_bm25 = max(bm25_scores.values())
        min_bm25 = min(bm25_scores.values())

        results = []
        for doc_id in candidates:
            bm25_score = bm25_scores[doc_id]

            vector_score = 0.0
            if query_vec is not None:
                doc_vec = self._doc_vectors.get(doc_id) or [0.0] * self.VECTOR_DIM
                vector_score = self._cosine_similarity(query_vec, doc_vec)

            if mode == "bm25":
                combined = bm25_score
            elif mode == "vector":
                combined = vector_score
            else:
                if max_bm25 > min_bm25:
                    normalized_bm25 = (bm25_score - min_bm25) / (max_bm25 - min_bm25)
                else:
                    normalized_bm25 = 0.0
                combined = 0.6 * normalized_bm25 + 0.4 * max(0.0, vector_score)

            results.append({
                "doc_id": doc_id,
                "score": combined,
                "bm25_score": bm25_score,
                "vector_score": vector_score,
                "category": self._doc_categories.get(doc_id, "unknown"),
                "text": self._doc_texts.get(doc_id, ""),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        # 分数为 0 表示"完全没有命中"，不要把无关文档当结果返回
        results = [r for r in results if r["score"] > 0]
        return results[:top_k]

    # ---- 统计与清理 ----

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """按 doc_id 返回单个文档信息。"""
        if doc_id not in self._doc_texts:
            return None
        return {
            "doc_id": doc_id,
            "category": self._doc_categories.get(doc_id, "unknown"),
            "text": self._doc_texts[doc_id],
            "token_count": self._doc_lengths.get(doc_id, 0),
        }

    def all_documents(self) -> List[Dict[str, Any]]:
        """返回所有文档（doc_id / category / text / token_count）。"""
        return [self.get_document(doc_id) for doc_id in self._doc_texts]

    def get_category_stats(self) -> Dict[str, int]:
        """获取各分类的文档数量统计。"""
        stats: Dict[str, int] = defaultdict(int)
        for category in self._doc_categories.values():
            stats[category] += 1
        return dict(stats)

    def clear_all(self) -> int:
        """清空所有索引和文档，返回被清除的文档数量。"""
        count = len(self._doc_texts)
        self._clear_index()
        if os.path.exists(self.documents_dir):
            for filename in os.listdir(self.documents_dir):
                if filename.endswith(".json"):
                    try:
                        os.remove(os.path.join(self.documents_dir, filename))
                    except (IOError, OSError):
                        pass
        return count

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
