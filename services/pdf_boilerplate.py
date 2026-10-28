"""PDF 页码行和目录清理。

PyPDFLoader 按页抽文本，页眉页脚会混进每一页的正文。一本车主手册里，
"20 MODEL 3 车主手册" 这类页脚出现在两成以上的切片里，检索和引用都带着它。

只删能确认带着本页页码的页边行，以及目录条目；拿不准的不动。

页码行：每页只看首尾各两行非空行。印刷页码 = 物理页码 + 全书统一的偏移，
印刷页码通常从封面、目录之后才开始数，所以偏移因书而异。偏移按两种证据确定，
先看第一种：
1. 纯页码行，如 "12"、"- 12 -"、"Page 12 of 30"、"第 12 页 共 30 页"。
   页码前后还带课程名等文字的（"Course Notes, page 12"），模板要在足够多页
   出现才算，免得 "见第 3 页的图" 这种正文被当成页码。
2. 页码加标题的行，如 "20 MODEL 3 车主手册"、"18 Chapter 2 Getting Started"。
   每页一道题的习题集里，"第3题""Problem 3" 也随页递增，单看数字分不开。
   所以要求同一偏移下至少两种标题各在两页以上出现（书名、章名交替），而且
   一页只该有一个页码：较多页面同时有两行这样的候选（"第3题" 加页脚），
   就当作拿不准，不删。
偏移由哪种证据确定，就只删哪种行。被删的行把页码换成 "#" 后，还必须在
两页以上出现：正文里偶然出现本页页码的一句话不会被删。

目录：同一页至少 5 行是 "标题……页码" 点线格式，这些页码不递减、不超过全书
页数，并且这一页或与它相连的目录页上有 "目录"/"内容"/"Contents" 标题，才删掉
这些条目和标题。升序的点线表格（扭矩表、平方表）没有这个标题，保留。删完
整页为空就丢掉整页。

不处理：不带页码的重复行，因为代码里的 #include、"解：" 这类标签和真正的
页眉分不开；也不处理索引页。
"""

import math
import re
from collections import Counter, defaultdict
from typing import NamedTuple

# 每页首尾各看几行非空行
_EDGE_LINES = 2
# 页数太少时"随页递增"没有统计意义，不删页码行
_MIN_PAGES = 4
# 同一个偏移至少要在这么多页上成立
_OFFSET_MIN_PAGE_FRACTION = 0.3
# 标题类证据至少要有几种各自重复的标题
_TITLED_MIN_TEMPLATES = 2
# 同时有两行标题类候选的页面达到这个比例，就认为页码有歧义
_TITLED_MAX_MULTI_FRACTION = 0.1
# 被删的行（页码换成 #）至少在几页上出现
_TEMPLATE_MIN_PAGES = 2
# 目录页至少有几条点线条目
_TOC_MIN_ENTRIES = 5

# 独立的 1～4 位数：小数、千分位、日期里的数不算（"3.14"、"1,024"、"2026-10-28"）
_NUMBER = re.compile(r"(?<!\d)(?<!\d[.,:\-])\d{1,4}(?!\d)(?![.,:\-]\d)")
_SPACES = re.compile(r"\s+")
# 纯页码模板："#"、"- # -"、"Page # of 30"、"第 # 页 共 30 页"。括号不算装饰：
# "(3)" "[3]" 是小题号、公式号、参考文献号
_BARE_FOLIO = re.compile(
    r"[\s\-–—|·•/]*(?:page|pg\.?|p\.|第)?\s*#\s*页?\s*"
    r"(?:(?:of|/|共)\s*\d+\s*(?:页|pages)?)?[\s\-–—|·•/]*",
    re.IGNORECASE,
)
# 写明了"第几页"的模板，前后可以带课程名、日期
_WORDED_FOLIO = re.compile(r"\bpage\s*#|第\s*#\s*页", re.IGNORECASE)
# "# of 30"、"#/30" 里的总页数；比页码还小的是分数或别的东西
_FOLIO_TOTAL = re.compile(r"#\s*页?\s*(?:of|/|共)\s*(\d+)", re.IGNORECASE)
_TOC_ENTRY = re.compile(r"(?:\.{4,}|…{2,}|·{4,}|(?:\. ){4,})\s*(\d{1,4})\s*$")
_TOC_HEADINGS = {"目录", "内容", "contents", "tableofcontents"}


class PdfCleanup(NamedTuple):
    # 与输入逐页对齐；None 表示整页只剩被删的内容，应当丢掉
    pages: list[str | None]
    removed_lines: int
    dropped_pages: int


class _Candidate(NamedTuple):
    page_index: int
    line_index: int
    template: str
    # "bare"：纯页码；"worded"：带文字但写明是页码；"titled"：页码加标题
    kind: str


def _edge_indexes(lines: list[str]) -> list[int]:
    non_empty = [i for i, line in enumerate(lines) if line.strip()]
    edges = non_empty[:_EDGE_LINES] + non_empty[-_EDGE_LINES:]
    return sorted(set(edges))


def _kind(template: str, number: int) -> str:
    total = _FOLIO_TOTAL.search(template)
    if total and int(total.group(1)) < number:
        return "titled"
    if _BARE_FOLIO.fullmatch(template):
        return "bare"
    if _WORDED_FOLIO.search(template):
        return "worded"
    return "titled"


def _candidates_by_offset(
    page_lines: list[list[str]],
) -> dict[int, list[_Candidate]]:
    """页边行里每个独立数字都假设是页码，按"数字 - 物理页码"归组。"""
    by_offset: dict[int, list[_Candidate]] = defaultdict(list)
    for page_index, lines in enumerate(page_lines):
        for line_index in _edge_indexes(lines):
            line = lines[line_index]
            for match in _NUMBER.finditer(line):
                number = int(match.group())
                template = _SPACES.sub(
                    " ", f"{line[:match.start()]}#{line[match.end():]}"
                ).strip()
                by_offset[number - (page_index + 1)].append(
                    _Candidate(page_index, line_index, template, _kind(template, number))
                )
    return by_offset


def _recurring(candidates: list[_Candidate], min_pages: int) -> list[_Candidate]:
    """只留模板在至少 min_pages 页上出现的候选。"""
    pages: dict[str, set[int]] = defaultdict(set)
    for candidate in candidates:
        pages[candidate.template].add(candidate.page_index)
    return [c for c in candidates if len(pages[c.template]) >= min_pages]


def _covered_pages(candidates: list[_Candidate]) -> int:
    return len({c.page_index for c in candidates})


def _page_number_lines(
    page_lines: list[list[str]],
) -> tuple[int | None, set[tuple[int, int]]]:
    """返回 (偏移, 要删的 (页, 行) 集合)；证据不足或有歧义时偏移为 None、不删。"""
    if len(page_lines) < _MIN_PAGES:
        return None, set()
    needed = max(
        _MIN_PAGES, math.ceil(_OFFSET_MIN_PAGE_FRACTION * len(page_lines))
    )
    by_offset = _candidates_by_offset(page_lines)

    def chosen(evidence: dict[int, list[_Candidate]]):
        qualified = [o for o, found in evidence.items() if _covered_pages(found) >= needed]
        if len(qualified) != 1:
            return None
        offset = qualified[0]
        return offset, {(c.page_index, c.line_index) for c in evidence[offset]}

    plain = {
        offset: _recurring([c for c in found if c.kind == "bare"], _TEMPLATE_MIN_PAGES)
        + _recurring([c for c in found if c.kind == "worded"], needed)
        for offset, found in by_offset.items()
    }
    result = chosen(plain)
    if result is not None:
        return result

    titled = {}
    for offset, found in by_offset.items():
        recurring = _recurring(
            [c for c in found if c.kind == "titled"], _TEMPLATE_MIN_PAGES
        )
        per_page = Counter(c.page_index for c in recurring)
        multi = sum(count >= 2 for count in per_page.values())
        if len({c.template for c in recurring}) >= _TITLED_MIN_TEMPLATES and multi < max(
            2, math.ceil(_TITLED_MAX_MULTI_FRACTION * len(per_page))
        ):
            titled[offset] = recurring
    result = chosen(titled)
    return result if result is not None else (None, set())


def _toc_lines(page_lines: list[list[str]], max_page: int) -> dict[int, set[int]]:
    """{页: 要删的目录条目和标题行号}。

    一页至少 _TOC_MIN_ENTRIES 条点线条目，页码不递减、不超过 max_page，且这一页
    或与它相连的同类页面上有目录标题。
    """
    entries: dict[int, set[int]] = {}
    headings: dict[int, set[int]] = {}
    for page_index, lines in enumerate(page_lines):
        found = []
        for i, line in enumerate(lines):
            match = _TOC_ENTRY.search(line)
            if match:
                found.append((i, int(match.group(1))))
        numbers = [number for _, number in found]
        if (
            len(found) < _TOC_MIN_ENTRIES
            or numbers != sorted(numbers)
            or numbers[-1] > max_page
        ):
            continue
        entries[page_index] = {i for i, _ in found}
        headings[page_index] = {
            i
            for i, line in enumerate(lines)
            if _SPACES.sub("", line).lower() in _TOC_HEADINGS
        }

    toc: dict[int, set[int]] = {}
    for page_index in sorted(entries):
        if page_index in toc:
            continue
        run = [page_index]
        while run[-1] + 1 in entries:
            run.append(run[-1] + 1)
        if any(headings[p] for p in run):
            for p in run:
                toc[p] = entries[p] | headings[p]
    return toc


def strip_pdf_boilerplate(pages: list[str]) -> PdfCleanup:
    """按页去掉页码行和目录条目；清理后什么都不剩时原样返回。"""
    page_lines = [page.splitlines() for page in pages]
    offset, folio_lines = _page_number_lines(page_lines)
    toc = _toc_lines(page_lines, len(pages) + max(offset or 0, 0))

    cleaned: list[str | None] = []
    removed_lines = 0
    dropped_pages = 0
    for page_index, lines in enumerate(page_lines):
        drop = toc.get(page_index, set()) | {
            line_index for p, line_index in folio_lines if p == page_index
        }
        removed_lines += len(drop)
        kept = [line for i, line in enumerate(lines) if i not in drop]
        if drop and not any(line.strip() for line in kept):
            cleaned.append(None)
            dropped_pages += 1
        else:
            cleaned.append("\n".join(kept))

    if not any(page and page.strip() for page in cleaned):
        return PdfCleanup(list(pages), 0, 0)
    return PdfCleanup(cleaned, removed_lines, dropped_pages)
