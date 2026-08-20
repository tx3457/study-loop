"""Bounded, dependency-free DOCX ZIP archive validation."""

import os
import stat
import struct
import unicodedata
import zipfile
import zlib


# DOCX 是 ZIP 容器。上传端同样使用 MAX_UPLOAD_MB（默认 20 MiB）限制压缩包
# 本身；以下预算限制其 central directory 与解压后的资源消耗，必须在交给
# Docx2txtLoader 前执行，避免由 loader 先展开恶意归档。
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "20")) * 1024 * 1024
DOCX_MAX_ARCHIVE_BYTES = MAX_UPLOAD_BYTES
DOCX_MAX_ENTRY_COUNT = 2_048
DOCX_MAX_CENTRAL_DIRECTORY_BYTES = 2 * 1024 * 1024
DOCX_MAX_ENTRY_UNCOMPRESSED_BYTES = 32 * 1024 * 1024
DOCX_MAX_TOTAL_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
DOCX_MAX_XML_ENTRY_UNCOMPRESSED_BYTES = 16 * 1024 * 1024
DOCX_MAX_XML_TOTAL_UNCOMPRESSED_BYTES = 16 * 1024 * 1024
DOCX_MAX_COMPRESSION_RATIO = 200
_DOCX_XML_READ_CHUNK_BYTES = 64 * 1024

_ZIP_EOCD_SIGNATURE = b"PK\x05\x06"
_ZIP_CENTRAL_DIRECTORY_SIGNATURE = b"PK\x01\x02"
_ZIP_LOCAL_FILE_SIGNATURE = b"PK\x03\x04"
_ZIP64_END_OF_CENTRAL_DIRECTORY_LOCATOR_SIGNATURE = b"PK\x06\x07"
_ZIP64_EXTRA_FIELD_ID = 0x0001
_UNICODE_PATH_EXTRA_FIELD_ID = 0x7075
_UNSAFE_DOCX_EXTRA_FIELD_IDS = frozenset({_ZIP64_EXTRA_FIELD_ID, _UNICODE_PATH_EXTRA_FIELD_ID})
_DOCX_REQUIRED_MEMBERS = frozenset({"[Content_Types].xml", "_rels/.rels", "word/document.xml"})


class DocxArchiveValidationError(Exception):
    """The uploaded DOCX archive is unsafe or malformed."""


def _reject_docx_archive() -> None:
    """Raise the stable, deliberately detail-free DOCX parse failure."""
    raise DocxArchiveValidationError()


def _find_zip_eocd(archive: bytes) -> tuple[int, tuple[int, ...]]:
    """Locate and minimally validate the classic ZIP end-of-central-directory record."""
    eocd_size = 22
    if len(archive) < eocd_size:
        _reject_docx_archive()

    # Match CPython's ZIP reader: it trusts the last signature in the permitted
    # comment suffix.  Falling back to an earlier marker would validate a
    # different central-directory view than the loader later consumes.
    search_start = max(0, len(archive) - eocd_size - 0xFFFF)
    offset = archive.rfind(_ZIP_EOCD_SIGNATURE, search_start)
    if offset < 0 or len(archive) - offset < eocd_size:
        _reject_docx_archive()
    fields = struct.unpack_from("<4s4H2LH", archive, offset)
    comment_length = fields[-1]
    if offset + eocd_size + comment_length != len(archive):
        _reject_docx_archive()
    # Drop the signature; callers only need the seven numeric fields.
    return offset, fields[1:]


def _has_unsafe_docx_extra_field(extra: bytes) -> bool:
    """Reject metadata that changes ZIP addressing or the loader-visible name."""
    cursor = 0
    while cursor < len(extra):
        if len(extra) - cursor < 4:
            _reject_docx_archive()
        field_id, field_length = struct.unpack_from("<HH", extra, cursor)
        cursor += 4
        if field_length > len(extra) - cursor:
            _reject_docx_archive()
        if field_id in _UNSAFE_DOCX_EXTRA_FIELD_IDS:
            return True
        cursor += field_length
    return False


def _validate_docx_member_name(raw_name: bytes, flags: int) -> str:
    """Reject names that are ambiguous or unsafe for a package loader to handle."""
    if not raw_name or b"\x00" in raw_name or len(raw_name) > 1024:
        _reject_docx_archive()
    try:
        name = raw_name.decode("utf-8" if flags & 0x0800 else "cp437")
    except UnicodeDecodeError:
        _reject_docx_archive()

    if (
        not name
        or "\\" in name
        or name.startswith("/")
        or name.startswith("\\")
        or any(ord(char) < 32 for char in name)
    ):
        _reject_docx_archive()
    parts = name.rstrip("/").split("/")
    if not parts or ":" in parts[0] or any(part in {"", ".", ".."} for part in parts):
        _reject_docx_archive()
    return name


def _validate_docx_xml_payload(
    archive: bytes,
    *,
    data_start: int,
    compressed_size: int,
    uncompressed_size: int,
    compression_method: int,
    expected_crc32: int,
) -> None:
    """Bounded raw inflation and CRC validation for XML read by ``docx2txt``."""
    data_end = data_start + compressed_size
    output_size = 0
    crc32 = 0
    previous_scan = b""

    def consume(output: bytes) -> None:
        nonlocal output_size, crc32, previous_scan
        output_size += len(output)
        if output_size > uncompressed_size:
            _reject_docx_archive()
        crc32 = zlib.crc32(output, crc32)
        scan_window = previous_scan + output.replace(b"\x00", b"").upper()
        if b"<!DOCTYPE" in scan_window or b"<!ENTITY" in scan_window:
            _reject_docx_archive()
        previous_scan = scan_window[-8:]

    if compression_method == zipfile.ZIP_STORED:
        if compressed_size != uncompressed_size:
            _reject_docx_archive()
        for offset in range(data_start, data_end, _DOCX_XML_READ_CHUNK_BYTES):
            consume(archive[offset : min(offset + _DOCX_XML_READ_CHUNK_BYTES, data_end)])
    else:
        try:
            decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
            for offset in range(data_start, data_end, _DOCX_XML_READ_CHUNK_BYTES):
                compressed_chunk = archive[
                    offset : min(offset + _DOCX_XML_READ_CHUNK_BYTES, data_end)
                ]
                while compressed_chunk:
                    before_size = len(compressed_chunk)
                    output = decompressor.decompress(
                        compressed_chunk,
                        min(
                            _DOCX_XML_READ_CHUNK_BYTES,
                            uncompressed_size - output_size + 1,
                        ),
                    )
                    compressed_chunk = decompressor.unconsumed_tail
                    consume(output)
                    if decompressor.eof:
                        if (
                            compressed_chunk
                            or decompressor.unused_data
                            or offset + _DOCX_XML_READ_CHUNK_BYTES < data_end
                        ):
                            _reject_docx_archive()
                        break
                    if not output and len(compressed_chunk) == before_size:
                        _reject_docx_archive()
            if not decompressor.eof:
                _reject_docx_archive()
        except zlib.error:
            _reject_docx_archive()

    if output_size != uncompressed_size or (crc32 & 0xFFFFFFFF) != expected_crc32:
        _reject_docx_archive()


def validate_docx_archive(file_bytes: bytes) -> None:
    """Validate the ZIP directory before bounded loader-relevant member checks.

    ``ZipFile.infolist()`` itself allocates one object per central-directory entry,
    so the compact central directory is parsed directly.  This is intentionally
    strict: DOCX files submitted here must be single-volume classic ZIP packages
    using stored or deflated members only.
    """
    if not file_bytes or len(file_bytes) > DOCX_MAX_ARCHIVE_BYTES:
        _reject_docx_archive()

    eocd_offset, eocd = _find_zip_eocd(file_bytes)
    (
        disk_number,
        central_directory_disk,
        entries_on_disk,
        entry_count,
        central_directory_size,
        central_directory_offset,
        _comment_length,
    ) = eocd

    # ZIP64 and multi-disk archives have indirections which defeat the bounded,
    # single-buffer validation below. They are neither needed nor expected in a
    # <=20 MiB DOCX upload.
    if (
        disk_number != 0
        or central_directory_disk != 0
        or entries_on_disk != entry_count
        or entry_count == 0xFFFF
        or central_directory_size == 0xFFFFFFFF
        or central_directory_offset == 0xFFFFFFFF
        or file_bytes[max(0, eocd_offset - 20) : eocd_offset - 16]
        == _ZIP64_END_OF_CENTRAL_DIRECTORY_LOCATOR_SIGNATURE
    ):
        _reject_docx_archive()
    if (
        entry_count > DOCX_MAX_ENTRY_COUNT
        or central_directory_size > DOCX_MAX_CENTRAL_DIRECTORY_BYTES
        or central_directory_size < entry_count * 46
        or central_directory_offset + central_directory_size != eocd_offset
    ):
        _reject_docx_archive()

    central_directory_end = central_directory_offset + central_directory_size
    if central_directory_offset < 0 or central_directory_end > eocd_offset:
        _reject_docx_archive()

    cursor = central_directory_offset
    total_uncompressed_size = 0
    declared_xml_total_size = 0
    seen_names: set[str] = set()
    seen_exact_names: set[str] = set()
    local_member_ranges: list[tuple[int, int]] = []
    xml_members: list[tuple[int, int, int, int, int]] = []
    for _ in range(entry_count):
        if central_directory_end - cursor < 46:
            _reject_docx_archive()
        fields = struct.unpack_from("<4s6H3L5H2L", file_bytes, cursor)
        (
            signature,
            version_made_by,
            _version_needed,
            flags,
            compression_method,
            _modified_time,
            _modified_date,
            _crc32,
            compressed_size,
            uncompressed_size,
            name_length,
            extra_length,
            comment_length,
            member_disk_number,
            _internal_attributes,
            external_attributes,
            local_header_offset,
        ) = fields
        if signature != _ZIP_CENTRAL_DIRECTORY_SIGNATURE:
            _reject_docx_archive()

        member_size = 46 + name_length + extra_length + comment_length
        if member_size > central_directory_end - cursor:
            _reject_docx_archive()
        name_start = cursor + 46
        name_end = name_start + name_length
        extra_end = name_end + extra_length
        raw_name = file_bytes[name_start:name_end]
        extra = file_bytes[name_end:extra_end]
        name = _validate_docx_member_name(raw_name, flags)
        normalized_name = unicodedata.normalize("NFC", name.rstrip("/")).casefold()
        # docx2txt's header/footer match is a permissive, unanchored regex.
        # Include those prefixes even if the suffix is unusual, so a crafted
        # filename cannot make the loader inflate an unvalidated stream.
        is_xml_member = normalized_name.endswith((".xml", ".rels")) or normalized_name.startswith(
            ("word/header", "word/footer")
        )
        allowed_flags = 0x0808  # data descriptor + UTF-8 names
        if compression_method == zipfile.ZIP_DEFLATED:
            allowed_flags |= 0x0006  # normal deflate compression options

        if (
            member_disk_number != 0
            or flags & ~allowed_flags
            or compression_method not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
            or _has_unsafe_docx_extra_field(extra)
            or compressed_size == 0xFFFFFFFF
            or uncompressed_size == 0xFFFFFFFF
            or local_header_offset == 0xFFFFFFFF
            or normalized_name in seen_names
        ):
            _reject_docx_archive()
        if (version_made_by >> 8) == 3:
            unix_file_type = stat.S_IFMT(external_attributes >> 16)
            if unix_file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
                _reject_docx_archive()
        if name.endswith("/") and uncompressed_size:
            _reject_docx_archive()
        if (
            uncompressed_size > DOCX_MAX_ENTRY_UNCOMPRESSED_BYTES
            or uncompressed_size > compressed_size * DOCX_MAX_COMPRESSION_RATIO
            or total_uncompressed_size > DOCX_MAX_TOTAL_UNCOMPRESSED_BYTES - uncompressed_size
        ):
            _reject_docx_archive()
        if is_xml_member and (
            uncompressed_size > DOCX_MAX_XML_ENTRY_UNCOMPRESSED_BYTES
            or declared_xml_total_size > DOCX_MAX_XML_TOTAL_UNCOMPRESSED_BYTES - uncompressed_size
        ):
            _reject_docx_archive()
        total_uncompressed_size += uncompressed_size
        if is_xml_member:
            declared_xml_total_size += uncompressed_size
        seen_names.add(normalized_name)
        seen_exact_names.add(name)

        # Validate the referenced local header too. This catches corrupt offsets,
        # mismatched metadata and overlapping member data before the loader reads
        # any member body.
        if local_header_offset + 30 > central_directory_offset:
            _reject_docx_archive()
        local_fields = struct.unpack_from("<4s5H3L2H", file_bytes, local_header_offset)
        (
            local_signature,
            _local_version_needed,
            local_flags,
            local_compression_method,
            _local_modified_time,
            _local_modified_date,
            _local_crc32,
            local_compressed_size,
            local_uncompressed_size,
            local_name_length,
            local_extra_length,
        ) = local_fields
        local_name_start = local_header_offset + 30
        local_data_start = local_name_start + local_name_length + local_extra_length
        local_extra_start = local_name_start + local_name_length
        local_extra_end = local_extra_start + local_extra_length
        if (
            local_signature != _ZIP_LOCAL_FILE_SIGNATURE
            or local_flags != flags
            or local_compression_method != compression_method
            or local_data_start > central_directory_offset
            or file_bytes[local_name_start : local_name_start + local_name_length] != raw_name
            or local_data_start + compressed_size > central_directory_offset
            or _has_unsafe_docx_extra_field(file_bytes[local_extra_start:local_extra_end])
            or (
                not flags & 0x0008
                and (
                    _local_crc32 != _crc32
                    or local_compressed_size != compressed_size
                    or local_uncompressed_size != uncompressed_size
                )
            )
        ):
            _reject_docx_archive()
        local_member_ranges.append((local_header_offset, local_data_start + compressed_size))
        if is_xml_member:
            xml_members.append(
                (
                    local_data_start,
                    compressed_size,
                    uncompressed_size,
                    compression_method,
                    _crc32,
                )
            )
        cursor += member_size

    if cursor != central_directory_end or not _DOCX_REQUIRED_MEMBERS.issubset(seen_exact_names):
        _reject_docx_archive()

    local_member_ranges.sort()
    if any(
        current_start < previous_end
        for (_, previous_end), (current_start, _) in zip(
            local_member_ranges, local_member_ranges[1:]
        )
    ):
        _reject_docx_archive()

    # Only package XML/relationship files are material to the document parser.
    # Validate their raw member streams with a bounded inflater, rather than
    # ZipExtFile: the latter trusts central-directory sizes and can silently hide
    # a stream whose real output is larger than its declared ``file_size``.
    for (
        data_start,
        compressed_size,
        uncompressed_size,
        compression_method,
        crc32,
    ) in xml_members:
        _validate_docx_xml_payload(
            file_bytes,
            data_start=data_start,
            compressed_size=compressed_size,
            uncompressed_size=uncompressed_size,
            compression_method=compression_method,
            expected_crc32=crc32,
        )
