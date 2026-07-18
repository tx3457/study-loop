"""Regression tests for document upload parsing and identifier boundaries."""

import tempfile
import unittest
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import chromadb
from chromadb.errors import NotFoundError
from fastapi.testclient import TestClient
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).parent.parent))

import routers.documents as documents_router
import services.vectorstore as vectorstore
from main import app


class TestDocumentUploadErrorBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app, raise_server_exceptions=False)

    def test_corrupt_pdf_returns_sanitized_json_without_indexing(self):
        index = AsyncMock()

        with patch.object(documents_router, "deal_document", index):
            response = self.client.post(
                "/documents/upload",
                files={"file": ("损坏 资料.pdf", b"not-a-pdf", "application/pdf")},
            )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            response.json(),
            {"detail": "文档无法解析，请确认文件未损坏且格式正确"},
        )
        self.assertNotIn("not-a-pdf", response.text)
        index.assert_not_awaited()

    def test_corrupt_docx_returns_sanitized_json_without_indexing(self):
        index = AsyncMock()

        with patch.object(documents_router, "deal_document", index):
            response = self.client.post(
                "/documents/upload",
                files={
                    "file": (
                        "损坏 讲义.docx",
                        b"not-a-zip-package",
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )
                },
            )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            response.json(),
            {"detail": "文档无法解析，请确认文件未损坏且格式正确"},
        )
        self.assertNotIn("zip", response.text.lower())
        index.assert_not_awaited()

    def test_parser_error_metadata_is_never_chunked_or_indexed(self):
        parser_result = [
            Document(
                page_content="PDF 转图失败：/srv/private/document.pdf",
                metadata={"error": "secret parser traceback"},
            )
        ]
        split = unittest.mock.MagicMock()
        index = AsyncMock()

        with (
            patch.object(
                documents_router, "parse_upload", AsyncMock(return_value=parser_result)
            ),
            patch.object(documents_router.default_chunker, "split_documents", split),
            patch.object(documents_router, "deal_document", index),
        ):
            response = self.client.post(
                "/documents/upload",
                files={"file": ("scan.pdf", b"%PDF-placeholder", "application/pdf")},
            )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            response.json(),
            {"detail": "文档无法解析，请确认文件未损坏且格式正确"},
        )
        self.assertNotIn("/srv/private", response.text)
        self.assertNotIn("secret parser traceback", response.text)
        split.assert_not_called()
        index.assert_not_awaited()

    def test_chinese_and_spaced_filename_remains_the_public_document_id(self):
        filename = "机器 学习讲义.md"
        parser_result = [Document(page_content="梯度下降", metadata={})]
        index = AsyncMock(return_value=1)

        with (
            patch.object(
                documents_router, "parse_upload", AsyncMock(return_value=parser_result)
            ),
            patch.object(documents_router, "deal_document", index),
        ):
            response = self.client.post(
                "/documents/upload",
                files={"file": (filename, "梯度下降".encode(), "text/markdown")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["document_id"], filename)
        self.assertEqual(response.json()["filename"], filename)
        index.assert_awaited_once_with(filename, filename, ["梯度下降"])

    def test_document_list_uses_public_id_from_collection_metadata(self):
        collection = SimpleNamespace(
            name="doc-0123456789abcdef",
            metadata={"source_document_id": "机器 学习讲义.md"},
        )
        with patch.object(
            documents_router,
            "get_all_document",
            AsyncMock(return_value=[collection]),
        ):
            response = self.client.get("/documents")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"documents": ["机器 学习讲义.md"]})

    def test_path_like_filename_is_rejected_before_parsing(self):
        parse = AsyncMock()
        with patch.object(documents_router, "parse_upload", parse):
            response = self.client.post(
                "/documents/upload",
                files={"file": ("../private.md", b"secret", "text/markdown")},
            )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            response.json(),
            {"detail": "文件名无效，请使用不含路径或控制字符的文件名"},
        )
        parse.assert_not_awaited()


class TestUnicodeDocumentStorage(unittest.IsolatedAsyncioTestCase):
    async def test_unicode_public_id_round_trips_through_safe_collection_name(self):
        filename = "机器 学习讲义.md"
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            client = chromadb.PersistentClient(path=str(Path(temp_dir) / "chroma"))
            embeddings = SimpleNamespace(
                data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])]
            )
            with (
                patch.object(vectorstore, "chromadb_client", client),
                patch.object(vectorstore, "_embed", AsyncMock(return_value=embeddings)),
            ):
                count = await vectorstore.deal_document(
                    filename, filename, ["梯度下降"]
                )
                collections = await vectorstore.get_all_document()

                self.assertEqual(count, 1)
                self.assertEqual(len(collections), 1)
                self.assertNotEqual(collections[0].name, filename)
                self.assertEqual(
                    collections[0].metadata["source_document_id"], filename
                )

                await vectorstore.ensure_document_available(filename)
                query_result = await vectorstore.query_document(filename, "梯度")
                self.assertEqual(query_result["documents"][0], ["梯度下降"])

                internal_id = collections[0].name
                with self.assertRaises(NotFoundError):
                    await vectorstore.ensure_document_available(internal_id)
                with self.assertRaises(NotFoundError):
                    await vectorstore.query_document(internal_id, "梯度")
                with self.assertRaises(NotFoundError):
                    await vectorstore.delete_document(internal_id)

                # Rejecting the internal alias must not affect the public id.
                await vectorstore.ensure_document_available(filename)

                await vectorstore.delete_document(filename)
                self.assertEqual(await vectorstore.get_all_document(), [])


if __name__ == "__main__":
    unittest.main()
