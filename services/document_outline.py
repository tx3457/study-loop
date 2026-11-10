"""Derive a section outline from a document's indexed chunks (no model call).

PDF chunks carry a page number and never cross a page, so the first line of a
page's first chunk is that page's title candidate; consecutive pages with the
same title form one section. Documents without pages (Markdown, text, DOCX)
contribute explicit heading lines only.

The outline is a map for navigation, not evidence: titles are heuristics over
untrusted document text and are never citable.
"""

from __future__ import annotations

import re
import unicodedata

MAX_TITLE_CHARS = 30
# A section whose title reappears after at most this many short sections
# (an example slide, a figure page) resumes instead of starting a new section.
_RESUME_LOOKBACK = 2
_RESUME_MAX_PAGES = 2

_CJK = "一-鿿"
_LEADING_NOISE = re.compile(r"^[^\w(\[【《\"“'#]+")
_SPACES = re.compile(r"\s+")
_CJK_GAP = re.compile(rf"(?<=[{_CJK}]) (?=[{_CJK}])")
_DATE = re.compile(r"\d{4}[/-]\d{1,2}[/-]\d{1,2}")
_CJK_CHARS = re.compile(rf"[{_CJK}]")
_LATIN_WORD = re.compile(r"[A-Za-z]{4,}")
_SECTION_NUMBER = re.compile(r"^\d+(\.\d+)*\.?\s+\S")
_TRAILING_NUMBER = re.compile(r"\s+(\d{1,4})$")
_MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+(\S.*)$")
_CHAPTER_HEADING = re.compile(r"^第\s*[一二三四五六七八九十百零\d]+\s*[章节篇部]")
_CODE_FENCE = re.compile(r"^(```|~~~)")


def _clean(line: str) -> str:
    text = unicodedata.normalize("NFKC", line)
    text = _LEADING_NOISE.sub("", text)
    return _CJK_GAP.sub("", _SPACES.sub(" ", text)).strip()


def _is_running_header(title: str, deck_title: str, page_number: int) -> bool:
    """The deck title repeated as a page header, possibly with the printed page number.

    Only a substantial prefix counts, so a short real section ("操作系统") inside a
    deck titled "操作系统原理与实践" is kept.
    """
    if not title or not deck_title:
        return False
    trailing = _TRAILING_NUMBER.search(title)
    if trailing and int(trailing.group(1)) == page_number:
        title = title[: trailing.start()].rstrip()
    return deck_title.startswith(title) and len(title) * 2 >= len(deck_title)


def _is_title(text: str) -> bool:
    """Short, word-like, and not a sentence fragment, code line or footer."""
    if not 2 <= len(text) <= MAX_TITLE_CHARS:
        return False
    if text[-1] in ",:;。、=(":  # NFKC already folded the full-width forms
        return False
    if any(mark in text for mark in ";{}<>") or text.startswith("//"):
        return False
    if _DATE.search(text) or "," in text or "。" in text:  # a footer or a sentence
        return False
    cjk = len(_CJK_CHARS.findall(text))
    if cjk >= 2 or (cjk == 1 and _SECTION_NUMBER.match(text)):  # "2.1 栈"
        return True
    # No CJK: require a real word and mostly letters, which rejects formula debris.
    letters = sum(ch.isalpha() for ch in text)
    return bool(_LATIN_WORD.search(text)) and letters / len(text) >= 0.6


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line
    return ""


def _paged_sections(rows: list[dict]) -> list[dict]:
    sections: list[dict] = []
    deck_title = ""
    seen_pages: set[int] = set()
    for row in rows:
        page = row["page_number"]
        if page in seen_pages:
            continue
        seen_pages.add(page)
        title = _clean(_first_line(row["text"]))
        if not sections:
            deck_title = title
            sections.append({
                "title": title[:MAX_TITLE_CHARS] or "(无标题)",
                "pages": [page, page],
                "chunk_index": row["chunk_index"],
            })
            continue
        current = sections[-1]
        running_header = _is_running_header(title, deck_title, page)
        if not _is_title(title) or running_header or title == current["title"]:
            current["pages"][1] = page
            continue
        resumed = _resumed_section(sections, title)
        if resumed is not None:
            del sections[resumed + 1:]
            sections[-1]["pages"][1] = page
            continue
        sections.append({"title": title, "pages": [page, page], "chunk_index": row["chunk_index"]})
    return sections


def _resumed_section(sections: list[dict], title: str) -> int | None:
    """Index of a recent section this title continues, past only short interruptions."""
    for back in range(2, _RESUME_LOOKBACK + 2):
        if len(sections) < back or sections[-back]["title"] != title:
            continue
        between = sections[len(sections) - back + 1:]
        if all(s["pages"][1] - s["pages"][0] + 1 <= _RESUME_MAX_PAGES for s in between):
            return len(sections) - back
    return None


def _heading_sections(rows: list[dict]) -> list[dict]:
    sections: list[dict] = []
    in_fence = False
    for row in rows:
        for line in row["text"].splitlines():
            stripped = line.strip()
            if _CODE_FENCE.match(stripped):
                in_fence = not in_fence  # "# comment" lines inside code are not headings
                continue
            if in_fence:
                continue
            markdown = _MARKDOWN_HEADING.match(stripped)
            if markdown:
                title = _clean(markdown.group(1))
            elif _CHAPTER_HEADING.match(stripped) and len(stripped) <= MAX_TITLE_CHARS:
                title = _clean(stripped)
            else:
                continue
            title = title[:MAX_TITLE_CHARS]
            if title and (not sections or sections[-1]["title"] != title):
                sections.append({"title": title, "pages": None, "chunk_index": row["chunk_index"]})
    return sections


def build_outline(rows: list[dict]) -> list[dict]:
    """Sections in document order: ``{title, pages: [first, last] | None, chunk_index}``.

    ``rows`` are ``get_document_chunks`` items. ``chunk_index`` is the first
    chunk of the section, the position to start reading from.
    """
    if not rows:
        return []
    if all(row.get("page_number") is not None for row in rows):
        return _paged_sections(rows)
    return _heading_sections(rows)
