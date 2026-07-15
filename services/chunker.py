"""
自适应文档切分（Phase 8 升级 P0-2，借鉴 AgentCraft core/text_splitter.py）

为什么不直接用 RecursiveCharacterTextSplitter：
  - 项目 README 一直宣称 SemanticChunker，但 routers/documents.py 实际用的是 Recursive。
  - Recursive 用通用分隔符，对中文段落结构感知差；对 PDF 多列布局会在中文标点处错切。
  - AdaptiveChunker = 段落优先 → 句子兜底 → 标点兜底 → 硬切兜底，按文件类型自动选策略。

策略层级（每一级都先看是否触发，不触发就降级）：
  L1 段落：按 \\n\\n 切；段落总长 ≤ chunk_size 就保完整段落
  L2 句子：单段超 chunk_size 时，按 。！？；及英文句末标点切句
  L3 标点：单句还超 chunk_size 时，按 ，、：等次级标点继续切
  L4 硬切：标点也救不了的极长字符串，按 chunk_size 直接切

PDF 文件单独处理：用更保守的分隔符集（排除中文逗号），避免多列布局把一行拆成两段。
"""
import logging
import re
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


# ── 切分参数（写成模块常量，便于面试时讲"这些是经验值，可以通过 chunk_size sweep 调"）─
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_MIN_CHUNK_SIZE = 100

# 段落 / 句子 / 标点 三级 separator
_PARAGRAPH_SEP = re.compile(r"\n\s*\n")
_SENTENCE_SEP = re.compile(r"(?<=[。！？；])|(?<=[.!?;]\s)")
_PUNCT_SEP = re.compile(r"(?<=[，、：:,])")


class AdaptiveChunker:
    """自适应切分器。按文档类型选不同的 separator 策略。"""

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        min_chunk_size: int = DEFAULT_MIN_CHUNK_SIZE,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        # PDF 专用：去掉中文逗号、顿号、冒号等会被多列布局误触发的分隔符
        # 保留段落、换行、句末标点和空格
        self._pdf_fallback = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""],
        )

    # ── 公开入口 ───────────────────────────────────────────────────────────
    def split_documents(self, docs: list[Document]) -> list[Document]:
        """切分一组 Document，保留 metadata 并注入 chunk_index / total_chunks。"""
        result: list[Document] = []
        for doc in docs:
            file_type = (doc.metadata.get("file_type") or "").lower()

            # PDF：走更保守的 Recursive；其他类型走自适应语义切分
            if file_type == "pdf":
                pieces = self._pdf_fallback.split_text(doc.page_content)
            elif file_type == "image":
                # OCR 结果通常已经很短，不再切，避免破坏识别完整性
                pieces = [doc.page_content] if doc.page_content.strip() else []
            else:
                pieces = self._semantic_split(doc.page_content)

            for i, piece in enumerate(pieces):
                if not piece.strip():
                    continue
                result.append(Document(
                    page_content=piece,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "total_chunks": len(pieces),
                    },
                ))

        logger.info(
            f"[chunker] {len(docs)} doc(s) → {len(result)} chunks "
            f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return result

    # ── 三级 fallback 切分 ─────────────────────────────────────────────────
    def _semantic_split(self, text: str) -> list[str]:
        """L1 段落 → L2 句子 → L3 标点 → L4 硬切。"""
        text = self._normalize(text)
        if not text:
            return []

        chunks: list[str] = []
        buf = ""

        for para in self._split_paragraphs(text):
            # 段落本身超长 → 进 L2
            if len(para) > self.chunk_size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(self._split_long(para))
                continue

            # 段落短，但加进当前 buffer 会超 → 先封存 buffer，新开一块
            if len(buf) + len(para) + 2 > self.chunk_size:
                if buf:
                    chunks.append(buf)
                buf = para
            else:
                buf = f"{buf}\n\n{para}" if buf else para

        if buf:
            chunks.append(buf)

        return self._merge_tiny(chunks)

    def _split_long(self, paragraph: str) -> list[str]:
        """L2：按句子切；句子还超 → 进 L3。"""
        chunks: list[str] = []
        buf = ""

        for sent in _SENTENCE_SEP.split(paragraph):
            sent = sent.strip()
            if not sent:
                continue

            if len(sent) > self.chunk_size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(self._split_by_punct(sent))
                continue

            if len(buf) + len(sent) > self.chunk_size:
                if buf:
                    chunks.append(buf)
                buf = sent
            else:
                buf += sent

        if buf:
            chunks.append(buf)
        return chunks

    def _split_by_punct(self, text: str) -> list[str]:
        """L3：按次级标点切；还是超 → 进 L4 硬切。"""
        chunks: list[str] = []
        buf = ""

        for piece in _PUNCT_SEP.split(text):
            if not piece:
                continue
            if len(buf) + len(piece) > self.chunk_size:
                if buf:
                    chunks.append(buf)
                # L4 硬切：piece 自己就超长，直接按 chunk_size 截断
                if len(piece) > self.chunk_size:
                    for i in range(0, len(piece), self.chunk_size):
                        chunks.append(piece[i:i + self.chunk_size])
                    buf = ""
                else:
                    buf = piece
            else:
                buf += piece

        if buf:
            chunks.append(buf)
        return chunks

    # ── 辅助 ───────────────────────────────────────────────────────────────
    def _split_paragraphs(self, text: str) -> list[str]:
        return [p.strip() for p in _PARAGRAPH_SEP.split(text) if p.strip()]

    def _normalize(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("　", " ")
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def _merge_tiny(self, chunks: list[str]) -> list[str]:
        """合并 size < min_chunk_size 的 chunk，避免太碎影响向量召回质量。"""
        if not chunks:
            return []
        merged: list[str] = []
        cur = chunks[0]
        for nxt in chunks[1:]:
            if len(cur) < self.min_chunk_size:
                cur = f"{cur}\n\n{nxt}"
            else:
                merged.append(cur)
                cur = nxt
        merged.append(cur)
        return merged


# ── 全局单例（避免每次上传都重建实例）───────────────────────────────────────
default_chunker = AdaptiveChunker()
