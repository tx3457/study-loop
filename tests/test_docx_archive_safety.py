"""DOCX archive validation must reject hostile ZIP metadata before loading it."""

import io
import stat
import sys
import unittest
import zipfile
import zlib
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

import routers.documents as documents_router
import services.docx_archive as docx_archive
import services.parser as parser
from main import app


def _zip_bytes(members: dict[str, bytes], *, compression: int = zipfile.ZIP_DEFLATED) -> bytes:
    return _zip_entries(list(members.items()), compression=compression)


def _zip_entries(
    members: list[tuple[str | zipfile.ZipInfo, bytes]],
    *,
    compression: int = zipfile.ZIP_DEFLATED,
) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=compression) as archive:
        for name, content in members:
            archive.writestr(name, content)
    return stream.getvalue()


def _zip_with_comment(comment: bytes) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", b"<Types />")
        archive.writestr("_rels/.rels", b"<Relationships />")
        archive.writestr("word/document.xml", b"<w:document />")
        archive.comment = comment
    return stream.getvalue()


class _NonSeekableBuffer(io.BytesIO):
    """Force zipfile's writer onto its legal data-descriptor code path."""

    def seekable(self) -> bool:
        return False

    def seek(self, *args, **kwargs):
        raise OSError("non-seekable fixture")


def _data_descriptor_docx() -> bytes:
    stream = _NonSeekableBuffer()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", b"<Types />")
        archive.writestr("_rels/.rels", b"<Relationships />")
        archive.writestr("word/document.xml", b"<w:document />")
    return stream.getvalue()


def _minimal_docx(
    *,
    document: bytes = b"<w:document />",
    compression: int = zipfile.ZIP_DEFLATED,
) -> bytes:
    return _zip_bytes(
        {
            "[Content_Types].xml": b"<Types />",
            "_rels/.rels": b"<Relationships />",
            "word/document.xml": document,
        },
        compression=compression,
    )


def _docx_with_extra_members(
    *members: tuple[str | zipfile.ZipInfo, bytes],
    document: bytes = b"<w:document />",
    compression: int = zipfile.ZIP_DEFLATED,
) -> bytes:
    return _zip_entries(
        [
            ("[Content_Types].xml", b"<Types />"),
            ("_rels/.rels", b"<Relationships />"),
            ("word/document.xml", document),
            *members,
        ],
        compression=compression,
    )


def _member_offsets(content: bytes, target_name: str) -> tuple[int, int]:
    """Return the central and local record offsets for one ordinary ZIP member."""
    eocd_offset = content.rfind(b"PK\x05\x06")
    assert eocd_offset >= 0
    central_offset = int.from_bytes(content[eocd_offset + 16 : eocd_offset + 20], "little")
    entries = int.from_bytes(content[eocd_offset + 10 : eocd_offset + 12], "little")
    cursor = central_offset
    for _ in range(entries):
        assert content[cursor : cursor + 4] == b"PK\x01\x02"
        flags = int.from_bytes(content[cursor + 8 : cursor + 10], "little")
        name_length = int.from_bytes(content[cursor + 28 : cursor + 30], "little")
        extra_length = int.from_bytes(content[cursor + 30 : cursor + 32], "little")
        comment_length = int.from_bytes(content[cursor + 32 : cursor + 34], "little")
        name = content[cursor + 46 : cursor + 46 + name_length].decode(
            "utf-8" if flags & 0x0800 else "cp437"
        )
        if name == target_name:
            return cursor, int.from_bytes(content[cursor + 42 : cursor + 46], "little")
        cursor += 46 + name_length + extra_length + comment_length
    raise AssertionError(f"member not found: {target_name}")


def _patch_member_metadata(
    content: bytes,
    target_name: str,
    *,
    central_crc: int | None = None,
    central_uncompressed: int | None = None,
    local_crc: int | None = None,
    local_uncompressed: int | None = None,
    local_extra_id: int | None = None,
) -> bytes:
    patched = bytearray(content)
    central, local = _member_offsets(content, target_name)
    if central_crc is not None:
        patched[central + 16 : central + 20] = central_crc.to_bytes(4, "little")
    if central_uncompressed is not None:
        patched[central + 24 : central + 28] = central_uncompressed.to_bytes(4, "little")
    if local_crc is not None:
        patched[local + 14 : local + 18] = local_crc.to_bytes(4, "little")
    if local_uncompressed is not None:
        patched[local + 22 : local + 26] = local_uncompressed.to_bytes(4, "little")
    if local_extra_id is not None:
        local_name_length = int.from_bytes(patched[local + 26 : local + 28], "little")
        local_extra_length = int.from_bytes(patched[local + 28 : local + 30], "little")
        assert local_extra_length >= 4
        extra_offset = local + 30 + local_name_length
        patched[extra_offset : extra_offset + 2] = local_extra_id.to_bytes(2, "little")
    return bytes(patched)


def _unicode_path_extra(raw_name: str, effective_name: str, *, field_id: int = 0x7075) -> bytes:
    payload = (
        b"\x01"
        + zlib.crc32(raw_name.encode("utf-8")).to_bytes(4, "little")
        + effective_name.encode("utf-8")
    )
    return field_id.to_bytes(2, "little") + len(payload).to_bytes(2, "little") + payload


def _set_zip_encrypted_flag(content: bytes) -> bytes:
    """Set ZIP's encryption bit in both local and central headers without data."""
    patched = bytearray(content)
    cursor = 0
    while cursor < len(patched):
        signature = bytes(patched[cursor : cursor + 4])
        if signature == b"PK\x03\x04":
            flag_offset = cursor + 6
            patched[flag_offset] |= 0x01
            name_len = int.from_bytes(patched[cursor + 26 : cursor + 28], "little")
            extra_len = int.from_bytes(patched[cursor + 28 : cursor + 30], "little")
            compressed_size = int.from_bytes(patched[cursor + 18 : cursor + 22], "little")
            cursor += 30 + name_len + extra_len + compressed_size
        elif signature == b"PK\x01\x02":
            patched[cursor + 8] |= 0x01
            cursor += (
                46
                + int.from_bytes(patched[cursor + 28 : cursor + 30], "little")
                + int.from_bytes(patched[cursor + 30 : cursor + 32], "little")
                + int.from_bytes(patched[cursor + 32 : cursor + 34], "little")
            )
        else:
            cursor += 1
    return bytes(patched)


def _patch_eocd(content: bytes, *, disk: int | None = None, entries: int | None = None) -> bytes:
    """Modify a small archive's classic EOCD fields without adding payload bytes."""
    patched = bytearray(content)
    offset = patched.rfind(b"PK\x05\x06")
    assert offset >= 0, "fixture must contain an EOCD"
    if disk is not None:
        patched[offset + 4 : offset + 6] = disk.to_bytes(2, "little")
    if entries is not None:
        encoded = entries.to_bytes(2, "little")
        patched[offset + 8 : offset + 10] = encoded
        patched[offset + 10 : offset + 12] = encoded
    return bytes(patched)


def _insert_zip64_locator(content: bytes) -> bytes:
    patched = bytearray(content)
    offset = patched.rfind(b"PK\x05\x06")
    assert offset >= 0, "fixture must contain an EOCD"
    patched[offset:offset] = b"PK\x06\x07" + (b"\x00" * 16)
    return bytes(patched)


class TestDocxArchiveValidation(unittest.TestCase):
    def assert_rejected(self, content: bytes, **limits: int) -> None:
        limiter = patch.multiple(docx_archive, **limits) if limits else nullcontext()
        with limiter:
            with self.assertRaises(docx_archive.DocxArchiveValidationError):
                docx_archive.validate_docx_archive(content)

    def assert_rejected_before_zipfile(self, content: bytes, **limits: int) -> None:
        limiter = patch.multiple(docx_archive, **limits) if limits else nullcontext()
        with limiter, patch.object(docx_archive.zipfile, "ZipFile") as zip_file:
            with self.assertRaises(docx_archive.DocxArchiveValidationError):
                docx_archive.validate_docx_archive(content)
        zip_file.assert_not_called()

    def test_normal_minimal_docx_is_accepted_and_boundary_sizes_are_inclusive(self):
        content = _minimal_docx(document=b"a" * 128)
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            infos = archive.infolist()
        largest = max(info.file_size for info in infos)
        total = sum(info.file_size for info in infos)
        max_ratio = max(info.file_size / max(info.compress_size, 1) for info in infos)

        with patch.multiple(
            docx_archive,
            DOCX_MAX_ENTRY_COUNT=len(infos),
            DOCX_MAX_ENTRY_UNCOMPRESSED_BYTES=largest,
            DOCX_MAX_TOTAL_UNCOMPRESSED_BYTES=total,
            DOCX_MAX_COMPRESSION_RATIO=max_ratio,
        ):
            docx_archive.validate_docx_archive(content)

    def test_bad_zip_and_non_docx_zip_are_rejected(self):
        self.assert_rejected(b"not a zip package")
        self.assert_rejected(_zip_bytes({"notes.txt": b"not a DOCX"}))

    def test_encrypted_member_is_rejected(self):
        self.assert_rejected(_set_zip_encrypted_flag(_minimal_docx()))

    def test_entry_count_single_entry_and_total_budget_are_rejected(self):
        self.assert_rejected(
            _minimal_docx(),
            DOCX_MAX_ENTRY_COUNT=2,
        )
        self.assert_rejected(
            _minimal_docx(document=b"x" * 128),
            DOCX_MAX_ENTRY_UNCOMPRESSED_BYTES=127,
        )
        self.assert_rejected(
            _minimal_docx(document=b"x" * 128),
            DOCX_MAX_TOTAL_UNCOMPRESSED_BYTES=128,
        )

    def test_high_compression_ratio_is_rejected_without_large_input(self):
        self.assert_rejected(
            _minimal_docx(document=b"0" * 4096),
            DOCX_MAX_COMPRESSION_RATIO=2,
        )

    def test_declared_entry_count_is_rejected_before_zipfile_construction(self):
        # A forged EOCD can advertise millions of entries while containing no
        # actual directory records.  The preflight must reject its declaration
        # before asking zipfile to enumerate anything.
        hostile = _patch_eocd(_zip_bytes({}), entries=2)
        with (
            patch.object(docx_archive, "DOCX_MAX_ENTRY_COUNT", 1),
            patch.object(docx_archive.zipfile, "ZipFile") as zip_file,
        ):
            with self.assertRaises(docx_archive.DocxArchiveValidationError):
                docx_archive.validate_docx_archive(hostile)
        zip_file.assert_not_called()

    def test_underdeclared_entry_count_is_rejected_before_zipfile_construction(self):
        # The central directory still has all three records, but EOCD claims
        # only one.  A bounded scan must reject the structural mismatch rather
        # than trusting the declaration and handing hidden records to a loader.
        self.assert_rejected_before_zipfile(_patch_eocd(_minimal_docx(), entries=1))

    def test_declared_central_directory_budget_is_rejected_before_zipfile(self):
        content = _minimal_docx()
        with (
            patch.object(docx_archive, "DOCX_MAX_CENTRAL_DIRECTORY_BYTES", 1),
            patch.object(docx_archive.zipfile, "ZipFile") as zip_file,
        ):
            with self.assertRaises(docx_archive.DocxArchiveValidationError):
                docx_archive.validate_docx_archive(content)
        zip_file.assert_not_called()

    def test_zip64_and_multi_disk_declarations_are_rejected_before_zipfile(self):
        zip64_sentinel = _patch_eocd(_zip_bytes({}), entries=0xFFFF)
        zip64_locator = _insert_zip64_locator(_zip_bytes({}))
        multi_disk = _patch_eocd(_zip_bytes({}), disk=1)
        for hostile in (zip64_sentinel, zip64_locator, multi_disk):
            with patch.object(docx_archive.zipfile, "ZipFile") as zip_file:
                with self.assertRaises(docx_archive.DocxArchiveValidationError):
                    docx_archive.validate_docx_archive(hostile)
            zip_file.assert_not_called()

    def test_zip_comment_with_fake_eocd_signature_is_rejected_before_loader(self):
        # CPython's loader uses the last EOCD marker rather than falling back
        # to an earlier one.  Accepting this archive would validate a different
        # directory from the one later given to docx2txt.
        comment = b"note PK\x05\x06" + (b"\x00" * 18) + b"still-a-comment"
        content = _zip_with_comment(comment)
        self.assert_rejected_before_zipfile(content)
        loader = unittest.mock.MagicMock()
        temporary_file = unittest.mock.MagicMock()
        with (
            patch.object(parser, "Docx2txtLoader", loader),
            patch.object(parser.tempfile, "NamedTemporaryFile", temporary_file),
        ):
            with self.assertRaises(parser.DocumentParseError):
                parser._parse_sync(content, "fake-eocd.docx")
        loader.assert_not_called()
        temporary_file.assert_not_called()

    def test_required_parts_paths_and_casefold_duplicates_are_rejected_preflight(self):
        for missing in (
            [("_rels/.rels", b"<Relationships />"), ("word/document.xml", b"x")],
            [("[Content_Types].xml", b"<Types />"), ("word/document.xml", b"x")],
            [("[Content_Types].xml", b"<Types />"), ("_rels/.rels", b"<Relationships />")],
        ):
            self.assert_rejected_before_zipfile(_zip_entries(missing))

        duplicate = _docx_with_extra_members(("Word/DOCUMENT.xml", b"duplicate"))
        self.assert_rejected_before_zipfile(duplicate)
        for name in ("../outside.xml", "C:/outside.xml"):
            self.assert_rejected_before_zipfile(_docx_with_extra_members((name, b"x")))

    def test_unsupported_compression_and_special_file_are_rejected_preflight(self):
        self.assert_rejected_before_zipfile(_minimal_docx(compression=zipfile.ZIP_BZIP2))
        symlink = zipfile.ZipInfo("word/link")
        symlink.create_system = 3
        symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
        self.assert_rejected_before_zipfile(_docx_with_extra_members((symlink, b"target")))

    def test_local_zip64_extra_and_local_crc_mismatch_are_rejected_preflight(self):
        # Keep the central extra field benign but turn its local counterpart
        # into ZIP64.  This catches loaders that trust local metadata.
        info = zipfile.ZipInfo("word/extra.bin")
        info.extra = b"\xfe\xca\x00\x00"
        local_zip64 = _patch_member_metadata(
            _docx_with_extra_members((info, b"x")),
            "word/extra.bin",
            local_extra_id=0x0001,
        )
        self.assert_rejected_before_zipfile(local_zip64)

        crc_mismatch = _patch_member_metadata(
            _minimal_docx(),
            "word/document.xml",
            local_crc=0,
        )
        self.assert_rejected_before_zipfile(crc_mismatch)

    def test_unicode_path_extra_cannot_change_loader_visible_member_names(self):
        raw_name = "word/safe.bin"
        effective_name = "word/header1.xml.evil"
        central_info = zipfile.ZipInfo(raw_name)
        central_info.extra = _unicode_path_extra(raw_name, effective_name)
        central_extra = _docx_with_extra_members((central_info, b"safe payload"))

        # Python's ZIP reader applies a central-directory Unicode Path extra
        # field, so docx2txt sees the header-like effective name rather than
        # the safe raw one inspected by a naive validator on Python 3.13+.
        # Python 3.11 ignores this field, but the validator must reject either
        # representation before passing it to the loader.
        with zipfile.ZipFile(io.BytesIO(central_extra)) as archive:
            names = archive.namelist()
            if sys.version_info >= (3, 13):
                self.assertIn(effective_name, names)
                self.assertNotIn(raw_name, names)
            else:
                self.assertIn(raw_name, names)
                self.assertNotIn(effective_name, names)

        loader = unittest.mock.MagicMock()
        temporary_file = unittest.mock.MagicMock()
        with (
            patch.object(parser, "Docx2txtLoader", loader),
            patch.object(parser.tempfile, "NamedTemporaryFile", temporary_file),
        ):
            with self.assertRaises(parser.DocumentParseError):
                parser._parse_sync(central_extra, "unicode-extra.docx")
        loader.assert_not_called()
        temporary_file.assert_not_called()

        # A local-header-only variant leaves ZipFile's central-directory name
        # unchanged today, but it remains ambiguous metadata that a downstream
        # loader must never receive.
        local_info = zipfile.ZipInfo(raw_name)
        local_info.extra = _unicode_path_extra(raw_name, effective_name, field_id=0xCAFE)
        local_extra = _patch_member_metadata(
            _docx_with_extra_members((local_info, b"safe payload")),
            raw_name,
            local_extra_id=0x7075,
        )
        with zipfile.ZipFile(io.BytesIO(local_extra)) as archive:
            self.assertIn(raw_name, archive.namelist())
            self.assertNotIn(effective_name, archive.namelist())
        self.assert_rejected_before_zipfile(local_extra)

    def test_declared_xml_budgets_are_rejected_before_inflate(self):
        content = _minimal_docx(document=b"x" * 256)
        self.assert_rejected_before_zipfile(content, DOCX_MAX_XML_ENTRY_UNCOMPRESSED_BYTES=255)
        self.assert_rejected_before_zipfile(content, DOCX_MAX_XML_TOTAL_UNCOMPRESSED_BYTES=255)

    def test_real_deflate_expansion_cannot_be_hidden_by_central_sizes_or_crc(self):
        # zipfile trusts the central directory's file_size and can return a
        # harmless-looking prefix.  The compressed bytes below still inflate to
        # one MiB, but all declared fields say ten bytes.
        document = b"x" * (1024 * 1024)
        forged = _patch_member_metadata(
            _minimal_docx(document=document),
            "word/document.xml",
            central_crc=zlib.crc32(b"x" * 10),
            central_uncompressed=10,
            local_crc=zlib.crc32(b"x" * 10),
            local_uncompressed=10,
        )
        with zipfile.ZipFile(io.BytesIO(forged)) as archive:
            self.assertEqual(archive.read("word/document.xml"), b"x" * 10)

        loader = unittest.mock.MagicMock()
        temporary_file = unittest.mock.MagicMock()
        with (
            patch.object(parser, "Docx2txtLoader", loader),
            patch.object(parser.tempfile, "NamedTemporaryFile", temporary_file),
            patch.object(docx_archive, "DOCX_MAX_ENTRY_UNCOMPRESSED_BYTES", 100),
            patch.object(docx_archive, "DOCX_MAX_TOTAL_UNCOMPRESSED_BYTES", 100),
        ):
            with self.assertRaises(parser.DocumentParseError):
                parser._parse_sync(forged, "forged-size.docx")
        loader.assert_not_called()
        temporary_file.assert_not_called()

    def test_docx2txt_header_footer_like_names_cannot_bypass_raw_inflate_budget(self):
        # docx2txt's header/footer matcher is intentionally broad.  A filename
        # with a suffix or wildcard character must therefore be validated as
        # parser material too, not merely by a strict `.xml` extension check.
        for member_name in (
            "word/header1.xml.evil",
            "word/header1Xxml",
            "word/footer1.xml.evil",
        ):
            forged = _patch_member_metadata(
                _docx_with_extra_members((member_name, b"h" * (256 * 1024))),
                member_name,
                central_crc=zlib.crc32(b"h" * 10),
                central_uncompressed=10,
                local_crc=zlib.crc32(b"h" * 10),
                local_uncompressed=10,
            )
            loader = unittest.mock.MagicMock()
            temporary_file = unittest.mock.MagicMock()
            with (
                patch.object(parser, "Docx2txtLoader", loader),
                patch.object(parser.tempfile, "NamedTemporaryFile", temporary_file),
                patch.object(docx_archive, "DOCX_MAX_ENTRY_UNCOMPRESSED_BYTES", 100),
                patch.object(docx_archive, "DOCX_MAX_TOTAL_UNCOMPRESSED_BYTES", 100),
            ):
                with self.assertRaises(parser.DocumentParseError):
                    parser._parse_sync(forged, "header-footer-bypass.docx")
            loader.assert_not_called()
            temporary_file.assert_not_called()

    def test_doctype_entity_and_nul_interleaved_declarations_are_rejected(self):
        boundary_doctype = b"x" * (1024 * 1024 - 4) + b"<!DOCTYPE doc>"
        nul_interleaved_entity = b"<\x00!\x00E\x00N\x00T\x00I\x00T\x00Y\x00 x\x00>\x00"
        # The repetitive prefix would normally hit the compression-ratio guard.
        # Raise that ceiling so this regression proves the DTD scan carries its
        # marker across the 64 KiB raw-inflate chunk boundary.
        self.assert_rejected(
            _minimal_docx(document=boundary_doctype),
            DOCX_MAX_COMPRESSION_RATIO=100_000,
        )
        self.assert_rejected(_minimal_docx(document=nul_interleaved_entity))

    def test_corrupt_crc_and_truncated_archives_are_rejected(self):
        content = bytearray(_minimal_docx(document=b"x" * 2048))
        _, local = _member_offsets(content, "word/document.xml")
        name_length = int.from_bytes(content[local + 26 : local + 28], "little")
        extra_length = int.from_bytes(content[local + 28 : local + 30], "little")
        data_offset = local + 30 + name_length + extra_length
        content[data_offset] ^= 0xFF
        self.assert_rejected(bytes(content))
        self.assert_rejected(_minimal_docx()[:-8])

    def test_empty_member_and_directory_entries_are_legal(self):
        for compression in (zipfile.ZIP_DEFLATED, zipfile.ZIP_STORED):
            self.assertIsNone(
                docx_archive.validate_docx_archive(
                    _docx_with_extra_members(
                        ("word/media/", b""),
                        ("word/empty.bin", b""),
                        compression=compression,
                    )
                )
            )

    def test_legal_data_descriptor_docx_is_accepted(self):
        content = _data_descriptor_docx()
        central, _ = _member_offsets(content, "word/document.xml")
        flags = int.from_bytes(content[central + 8 : central + 10], "little")
        self.assertTrue(flags & 0x0008)
        self.assertIsNone(docx_archive.validate_docx_archive(content))

    def test_parse_upload_rejects_before_docx_loader_or_tempfile_write(self):
        loader = unittest.mock.MagicMock()
        temporary_file = unittest.mock.MagicMock()

        with (
            patch.object(parser, "Docx2txtLoader", loader),
            patch.object(parser.tempfile, "NamedTemporaryFile", temporary_file),
            patch.object(docx_archive, "DOCX_MAX_ENTRY_COUNT", 2),
        ):
            with self.assertRaises(parser.DocumentParseError):
                parser._parse_sync(_minimal_docx(), "hostile.docx")

        loader.assert_not_called()
        temporary_file.assert_not_called()


class TestDocxArchiveUploadBoundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app, raise_server_exceptions=False)

    def test_hostile_docx_returns_fixed_error_before_loader_or_indexing(self):
        loader = unittest.mock.MagicMock()
        index = AsyncMock()
        with (
            patch.object(parser, "Docx2txtLoader", loader),
            patch.object(documents_router, "deal_document", index),
            patch.object(docx_archive, "DOCX_MAX_ENTRY_COUNT", 2),
        ):
            response = self.client.post(
                "/documents/upload",
                files={
                    "file": (
                        "hostile.docx",
                        _minimal_docx(),
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )
                },
            )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            response.json(),
            {"detail": "文档无法解析，请确认文件未损坏且格式正确"},
        )
        loader.assert_not_called()
        index.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
