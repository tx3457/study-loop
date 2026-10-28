"""PDF 页码行和目录清理：只删能确认带着本页页码的页边行和点线目录条目。"""

from unittest.mock import patch

from langchain_core.documents import Document

import services.parser as parser
from services.parser import pdf_page_number
from services.pdf_boilerplate import strip_pdf_boilerplate

_TOPICS = [
    "轮胎", "刹车", "座椅", "车门", "车窗", "灯光", "雨刷", "空调",
    "电池", "充电", "后视镜", "方向盘", "喇叭", "天窗", "门锁", "仪表",
]


def _body(index: int) -> str:
    """每页不同、不含数字的两行正文，避免正文本身被当成页码行。"""
    topic = _TOPICS[index]
    return f"{topic}的检查方法。\n{topic}的注意事项。"


def _manual_pages(count: int) -> list[str]:
    """仿车主手册：前两页无页码，之后偶数页页脚是手册名、奇数页是章名，印刷页码 = 物理页码 - 2。"""
    pages = []
    for physical in range(1, count + 1):
        body = _body(physical - 1)
        if physical <= 2:
            pages.append(body)
        elif physical % 2 == 0:
            pages.append(f"{body}\n{physical - 2} MODEL 3 车主手册")
        else:
            pages.append(f"{body}\n{physical - 2}座椅和安全约束系统")
    return pages


def _toc(entries: list[tuple[str, int]]) -> str:
    return "\n".join(f"{title}{'.' * 20}{page}" for title, page in entries)


def test_titled_page_number_footers_with_an_offset_are_removed():
    result = strip_pdf_boilerplate(_manual_pages(10))

    assert result.removed_lines == 8
    assert result.pages == [_body(i) for i in range(10)]


def test_bare_and_worded_page_numbers_are_removed():
    pages = [
        f"Course Notes, page {i + 1} of 6, Spring 2026\n{_body(i)}\n- {i + 1} -"
        for i in range(6)
    ]

    result = strip_pdf_boilerplate(pages)

    assert result.pages == [_body(i) for i in range(6)]


def test_one_numbered_question_per_page_is_not_mistaken_for_page_numbers():
    for label in ("第{}题", "Problem {}", "习题 {}"):
        pages = [f"{label.format(i + 1)}\n{_body(i)}" for i in range(10)]

        result = strip_pdf_boilerplate(pages)

        assert result.pages == pages, label


def test_question_numbers_survive_when_the_page_numbers_have_another_offset():
    shifted = [f"第{i + 1}题\n{_body(i)}\n{i + 3}" for i in range(10)]
    with_cover = [_body(0)] + [f"第{i}题\n{_body(i)}\n{i + 1}" for i in range(1, 10)]

    assert strip_pdf_boilerplate(shifted).pages == [
        f"第{i + 1}题\n{_body(i)}" for i in range(10)
    ]
    assert strip_pdf_boilerplate(with_cover).pages == [_body(0)] + [
        f"第{i}题\n{_body(i)}" for i in range(1, 10)
    ]


def test_body_line_that_happens_to_contain_the_page_number_is_kept():
    pages = [f"{_body(i)}\n{i + 1}" for i in range(16)]
    pages[6] = f"{_body(6)}\n所以 x = 7\n7"
    pages[13] = f"{_body(13)}\nπ ≈ 3.14\n14"
    pages[9] = f"{_body(9)}\n2026-10-10 更新\n10"

    result = strip_pdf_boilerplate(pages)

    assert "所以 x = 7" in result.pages[6]
    assert "π ≈ 3.14" in result.pages[13]
    assert "2026-10-10 更新" in result.pages[9]
    assert result.removed_lines == 16


def test_two_numbered_lines_per_page_are_treated_as_ambiguous():
    with_footer = [f"第{i + 1}题\n{_body(i)}\n高等数学练习 {i + 1}" for i in range(10)]
    two_labels = [f"例题 {i + 1}\n{_body(i)}\n练习 {i + 1}" for i in range(10)]

    assert strip_pdf_boilerplate(with_footer).pages == with_footer
    assert strip_pdf_boilerplate(two_labels).pages == two_labels


def test_bracketed_labels_and_fractions_are_not_page_numbers():
    labels = [f"({i + 1})\n{_body(i)}" for i in range(10)]
    equations = [f"{_body(i)}\n{i + 1}" for i in range(12)]
    equations[3] = f"{_body(3)}\n(4)\n4"
    equations[6] = f"{_body(6)}\n(7)\n7"
    fractions = [f"{_body(i)}\n{i + 1}" for i in range(12)]
    fractions[4] = f"{_body(4)}\n5/4\n5"
    fractions[8] = f"{_body(8)}\n9/4\n9"

    assert strip_pdf_boilerplate(labels).pages == labels
    assert "(4)" in strip_pdf_boilerplate(equations).pages[3]
    assert "(7)" in strip_pdf_boilerplate(equations).pages[6]
    assert "5/4" in strip_pdf_boilerplate(fractions).pages[4]
    assert "9/4" in strip_pdf_boilerplate(fractions).pages[8]


def test_sentence_mentioning_its_own_page_on_a_few_pages_is_kept():
    pages = [f"{_body(i)}\n{i + 1}" for i in range(16)]
    pages[2] = f"{_body(2)}\nSee the figure on page 3\n3"
    pages[7] = f"{_body(7)}\nSee the figure on page 8\n8"

    result = strip_pdf_boilerplate(pages)

    assert "See the figure on page 3" in result.pages[2]
    assert "See the figure on page 8" in result.pages[7]


def test_repeated_lines_without_the_page_number_are_kept():
    pages = [f"#include <stdio.h>\n{_body(i)}\nreturn 0;" for i in range(6)]

    result = strip_pdf_boilerplate(pages)

    assert result.pages == pages


def test_citation_markers_next_to_the_page_number_are_kept():
    # pypdf 常把引用标号抽到页尾，紧挨着页码
    pages = [f"{_body(i)}\n20\n21\n{i + 1}" for i in range(6)]

    result = strip_pdf_boilerplate(pages)

    assert result.pages == [f"{_body(i)}\n20\n21" for i in range(6)]


def test_table_of_contents_entries_are_removed_and_an_empty_page_dropped():
    toc = "目录\n" + _toc([(f"第{i}章", i * 2) for i in range(1, 7)])
    handout = "课程讲义\n目录\n" + _toc([(f"第{i}节", i) for i in range(1, 6)]) + "\n本讲义介绍二叉树。"

    bodies = [_body(i) for i in range(12)]

    result = strip_pdf_boilerplate([toc, handout, *bodies])

    assert result.pages == [None, "课程讲义\n本讲义介绍二叉树。", *bodies]
    assert result.dropped_pages == 1


def test_dotted_pages_that_are_not_contents_are_kept():
    index = _toc([("二叉树", 12), ("哈希表", 3), ("堆", 40), ("链表", 7), ("栈", 21)])
    specs = _toc([("整车长度", 4694), ("整车宽度", 1850), ("轴距", 2875), ("高度", 1443), ("重量", 1765)])
    answers = _toc([("1. 7 + 3 =", 10), ("2. 9 - 4 =", 5), ("3. 6 + 6 =", 12), ("4. 8 - 7 =", 1), ("5. 2 + 2 =", 4)])
    # 升序、不超过总页数的点线表格，只差一个目录标题
    squares = "平方表\n" + _toc([(str(n), n * n) for n in range(1, 8)])
    pages = [index, specs, answers, squares, *(["正文内容。"] * 56)]

    result = strip_pdf_boilerplate(pages)

    assert result.pages == pages


def test_short_documents_keep_their_page_numbers():
    pages = [f"{_body(i)}\n{i + 1}" for i in range(3)]

    result = strip_pdf_boilerplate(pages)

    assert result.pages == pages


def test_cleanup_that_would_empty_the_document_returns_it_unchanged():
    toc = "Contents\n" + _toc([(f"Chapter {i}", i) for i in range(1, 9)])

    result = strip_pdf_boilerplate([toc] * 8)

    assert result.pages == [toc] * 8
    assert (result.removed_lines, result.dropped_pages) == (0, 0)


def test_parse_pdf_drops_toc_pages_and_keeps_physical_page_numbers():
    pages = _manual_pages(8)
    pages[2] = "目录\n" + _toc([(f"第{i}章", i) for i in range(1, 9)])
    loaded = [Document(page_content=text, metadata={"page": i}) for i, text in enumerate(pages)]

    with patch.object(parser, "PyPDFLoader") as loader:
        loader.return_value.load.return_value = loaded
        docs = parser._parse_pdf(b"%PDF-1.4", "manual.pdf")

    assert [pdf_page_number(d.metadata) for d in docs] == [1, 2, 4, 5, 6, 7, 8]
    assert [d.page_content for d in docs] == [_body(i) for i in (0, 1, 3, 4, 5, 6, 7)]
    assert all(d.metadata["total_pages"] == 8 for d in docs)
