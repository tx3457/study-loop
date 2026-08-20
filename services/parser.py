"""
多格式文档解析

支持格式：
  .pdf          → 文本型走 PyPDFLoader；扫描型自动 fallback 到 pdf2image + OCR
  .docx         → Docx2txtLoader
  .txt / .md    → TextLoader（utf-8 / GBK fallback）
  .png/.jpg/... → OCRBackend.recognize()

设计选择：
  1. 解析与切分解耦：parser 只产 Document，chunker 单独负责切块
  2. OCRBackend 抽象（Strategy 模式）：当前用 Tesseract，其他 backend
     可实现同一接口并通过配置切换
  3. PDF 两路径自动切换：
       PyPDFLoader 抽到的文本长度 < 阈值 → 判定为扫描型 → pdf2image + 每页 OCR
       否则 → 文本路径
  4. OCR 失败 graceful：返回带 error metadata 的 Document，不抛异常
     （避免单页失败拖垮整批上传）
"""

import asyncio
import io
import logging
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


SUPPORTED_TEXT_EXTS = {".txt", ".md"}
SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif"}

# 扫描型 PDF 判定：PyPDFLoader 平均每页抽到的字符数 < 此值 → 走 OCR
SCANNED_PDF_CHARS_PER_PAGE_THRESHOLD = 30

# OCR fallback 最大处理页数（避免 200 页扫描书把后端卡 5 分钟）
OCR_MAX_PAGES = 20


class UnsupportedFileError(ValueError):
    """文件类型不支持。继承 ValueError，由 main.py 的 handler 转 400。"""


class DocumentParseError(Exception):
    """The uploaded bytes cannot be safely parsed as the declared document type."""


# ── OCR Backend 抽象（Strategy 模式）─────────────────────────────────────────
class OCRBackend(ABC):
    """OCR 引擎抽象。切换 Tesseract → PaddleOCR/EasyOCR 时只换实现类。"""

    name: str = "base"
    available: bool = False

    @abstractmethod
    def recognize(self, image) -> str:
        """传入 PIL.Image.Image，返回识别文本（空字符串表示未识别到）。"""
        raise NotImplementedError


class TesseractBackend(OCRBackend):
    """Tesseract OCR（默认 backend）。

    依赖：
      pip 包：pytesseract、Pillow
      系统：tesseract-ocr + 语言包 (chi_sim/chi_tra/eng)
            apt install tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-chi-tra

    识别策略：chi_sim+eng → chi_sim → eng 三段 fallback，先非空就返回。
    """

    name = "tesseract"

    def __init__(self):
        self.available = False
        try:
            import pytesseract

            pytesseract.get_tesseract_version()
            self._pytesseract = pytesseract
            self.available = True
            logger.info("[parser] TesseractBackend ready")
        except Exception as e:
            logger.warning(
                "[parser] TesseractBackend unavailable: error_type=%s",
                type(e).__name__,
            )

    def recognize(self, image) -> str:
        if not self.available:
            return ""
        # 灰度化（提识别率）+ 限制最大边长 2000px（避免 OOM）
        if image.mode != "L":
            image = image.convert("L")
        max_size = 2000
        if max(image.size) > max_size:
            from PIL import Image as _Image

            ratio = max_size / max(image.size)
            image = image.resize(
                (int(image.size[0] * ratio), int(image.size[1] * ratio)),
                _Image.Resampling.LANCZOS,
            )
        # 多语言 fallback：第一个非空就返回
        for lang in ("chi_sim+eng", "chi_sim", "eng"):
            try:
                text = self._pytesseract.image_to_string(
                    image, lang=lang, config="--psm 3 --oem 3"
                ).strip()
                if text:
                    return text
            except Exception as e:
                logger.debug(
                    "[parser] OCR attempt failed: error_type=%s",
                    type(e).__name__,
                )
        return ""


# 全局 backend 单例（启动时初始化一次，避免每次解析都探测 tesseract 版本）
# 生产切换：把这行换成 PaddleOCRBackend() / EasyOCRBackend() 即可
_ocr_backend: OCRBackend = TesseractBackend()


# ── PIL 延迟导入：Pillow 没装时不直接 crash 模块加载 ──────────────────────────
# 解压炸弹防护:PIL 声明像素数 > 2×此阈值会抛 DecompressionBombError(被 _ocr_image_bytes 的 except 兜住)
_MAX_IMAGE_PIXELS = 50_000_000  # ~50 MP


def _open_image(image_bytes: bytes):
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = _MAX_IMAGE_PIXELS  # 收紧默认阈值,防恶意超大图 OOM
    return Image.open(io.BytesIO(image_bytes))


# ── 单张图片 OCR → Document ────────────────────────────────────────────────
def _ocr_image_bytes(
    image_bytes: bytes, source: str, page: Optional[int] = None
) -> Document:
    """OCR 一张图片字节流，返回单 Document。失败时返回 error-marked Document。"""
    if not _ocr_backend.available:
        return Document(
            page_content=f"图片 OCR 不可用：{_ocr_backend.name} 未安装",
            metadata={
                "source": source,
                "file_type": "image",
                "ocr_backend": _ocr_backend.name,
                "error": "ocr_unavailable",
                **({"page": page} if page is not None else {}),
            },
        )
    try:
        image = _open_image(image_bytes)
        text = _ocr_backend.recognize(image)
        if not text:
            return Document(
                page_content="图片中未识别到文字",
                metadata={
                    "source": source,
                    "file_type": "image",
                    "ocr_backend": _ocr_backend.name,
                    "error": "no_text_detected",
                    **({"page": page} if page is not None else {}),
                },
            )
        return Document(
            page_content=text,
            metadata={
                "source": source,
                "file_type": "image",
                "ocr_backend": _ocr_backend.name,
                **({"page": page} if page is not None else {}),
            },
        )
    except Exception as e:
        logger.error(
            "[parser] OCR fatal error: page=%s error_type=%s",
            page,
            type(e).__name__,
        )
        return Document(
            page_content="图片 OCR 失败，未提取到文本",
            metadata={
                "source": source,
                "file_type": "image",
                "ocr_backend": _ocr_backend.name,
                "error": "ocr_failed",
                **({"page": page} if page is not None else {}),
            },
        )


# ── 扫描型 PDF：转图 + 每页 OCR ───────────────────────────────────────────
def _ocr_scanned_pdf(file_bytes: bytes, filename: str) -> list[Document]:
    """扫描型 PDF fallback：用 pdf2image 把每页渲染成图片，逐页 OCR。

    依赖：
      pip: pdf2image
      系统: poppler-utils (apt install poppler-utils)

    安全限制：最多处理 OCR_MAX_PAGES 页（避免 200 页扫描书拖垮服务）。
    """
    try:
        from pdf2image import convert_from_bytes, pdfinfo_from_bytes
    except ImportError:
        logger.warning(
            "[parser] pdf2image 未安装，扫描型 PDF 无法处理。pip install pdf2image"
        )
        return [
            Document(
                page_content="扫描型 PDF 解析不可用：pdf2image 未安装",
                metadata={
                    "source": filename,
                    "file_type": "pdf",
                    "error": "pdf2image_missing",
                },
            )
        ]

    try:
        # 先探测真实总页数(用于 truncated 标记),再只渲染前 OCR_MAX_PAGES 页:
        # 避免恶意"页数炸弹"PDF 先把全部页渲染成位图导致 OOM
        try:
            total_pages = int(pdfinfo_from_bytes(file_bytes).get("Pages", 0)) or None
        except Exception:
            total_pages = None
        # dpi=200 平衡识别质量与转换开销；JPEG 控制中间图像体积。
        images = convert_from_bytes(
            file_bytes,
            dpi=200,
            fmt="jpeg",
            first_page=1,
            last_page=OCR_MAX_PAGES,
        )
    except Exception as e:
        logger.error(
            "[parser] pdf2image conversion failed: error_type=%s",
            type(e).__name__,
        )
        return [
            Document(
                page_content="PDF 转图失败，未提取到文本",
                metadata={
                    "source": filename,
                    "file_type": "pdf",
                    "error": "pdf_conversion_failed",
                },
            )
        ]

    if total_pages is None:
        total_pages = len(images)
    pages_to_process = images  # 渲染阶段已限制到 OCR_MAX_PAGES
    logger.info(
        f"[parser] scanned PDF detected: {total_pages} pages, OCR-ing first {len(pages_to_process)}"
    )

    docs: list[Document] = []
    for i, img in enumerate(pages_to_process):
        # 转 bytes 复用 _ocr_image_bytes 逻辑
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        doc = _ocr_image_bytes(buf.getvalue(), filename, page=i)
        doc.metadata["file_type"] = "pdf"  # 覆盖：实质是 PDF 不是 image
        doc.metadata["total_pages"] = total_pages
        doc.metadata["ocr_scanned_pdf"] = True
        if total_pages > OCR_MAX_PAGES:
            doc.metadata["truncated"] = True
        docs.append(doc)
    return docs


# ── PDF 主入口：文本型直走 PyPDFLoader，扫描型 fallback OCR ─────────────────
def _parse_pdf(file_bytes: bytes, filename: str) -> list[Document]:
    """PDF 两路径自动切换。

    判定逻辑：PyPDFLoader 抽出来的总字符数 / 页数 < 阈值 → 判定扫描型。
    （单纯检查"是否为空"不够稳，因为有些 PDF 有少量页眉页脚字但内容是图）
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        docs = PyPDFLoader(tmp_path).load()
        total_pages = max(len(docs), 1)
        total_chars = sum(len((d.page_content or "").strip()) for d in docs)
        avg_per_page = total_chars / total_pages

        if avg_per_page < SCANNED_PDF_CHARS_PER_PAGE_THRESHOLD:
            logger.info(
                f"[parser] PDF appears scanned (avg {avg_per_page:.0f} chars/page < "
                f"threshold {SCANNED_PDF_CHARS_PER_PAGE_THRESHOLD}), fallback to OCR"
            )
            return _ocr_scanned_pdf(file_bytes, filename)

        # 文本型：标注 metadata
        for d in docs:
            d.metadata["source"] = filename
            d.metadata["file_type"] = "pdf"
            d.metadata["total_pages"] = total_pages
        return docs
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _parse_sync(file_bytes: bytes, filename: str) -> list[Document]:
    """同步解析（在 asyncio.to_thread 里跑）。"""
    ext = Path(filename).suffix.lower()

    # 文本类：内存解析，零 I/O
    if ext in SUPPORTED_TEXT_EXTS:
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            # Windows 中文 GBK fallback
            text = file_bytes.decode("gbk", errors="replace")
        return [
            Document(
                page_content=text,
                metadata={"source": filename, "file_type": ext.lstrip(".")},
            )
        ]

    # 图片：单独走 OCR
    if ext in SUPPORTED_IMAGE_EXTS:
        return [_ocr_image_bytes(file_bytes, filename)]

    # PDF：两路径自动切换
    if ext == ".pdf":
        return _parse_pdf(file_bytes, filename)

    # DOCX
    if ext == ".docx":
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            docs = Docx2txtLoader(tmp_path).load()
            for d in docs:
                d.metadata["source"] = filename
                d.metadata["file_type"] = "docx"
            return docs
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    raise UnsupportedFileError(
        f"不支持的文件类型：{ext}。当前支持：.pdf .docx .txt .md "
        f"以及图片（{', '.join(sorted(SUPPORTED_IMAGE_EXTS))}）"
    )


async def parse_upload(file_bytes: bytes, filename: str) -> list[Document]:
    """异步入口。同步阻塞 I/O 扔到 thread pool 避免堵 event loop。"""
    try:
        return await asyncio.to_thread(_parse_sync, file_bytes, filename)
    except UnsupportedFileError:
        raise
    except Exception as exc:
        logger.error(
            "[parser] document parsing failed: type=%s bytes=%d error_type=%s",
            Path(filename).suffix.lower() or "unknown",
            len(file_bytes),
            type(exc).__name__,
        )
        raise DocumentParseError from exc


# 暴露给上层调用和单测
def get_ocr_backend_name() -> str:
    return _ocr_backend.name


def is_ocr_available() -> bool:
    return _ocr_backend.available
