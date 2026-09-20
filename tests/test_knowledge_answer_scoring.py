"""Pure scoring contracts for the frozen knowledge-answer characterization."""

from scripts.evaluate_knowledge_answers import score_answer


_PROTOCOL = {
    "rubrics": {
        "answerable": {
            "q_answerable": {
                "concepts": [
                    {"concept_id": "alpha", "patterns": [r"\balpha\b"]},
                    {"concept_id": "beta", "patterns": [r"\bbeta\b"]},
                ]
            }
        },
        "unanswerable_question_ids": ["q_unanswerable"],
    }
}


def test_fabricated_citation_is_reported_even_when_required_source_is_cited() -> None:
    score = score_answer(
        question_id="q_answerable",
        answer="Alpha and beta are both covered.",
        citation_ids=["source_required", "source_forged"],
        abstained=False,
        retrieved_source_ids=["source_required"],
        required_source_ids=["source_required"],
        protocol=_PROTOCOL,
    )

    assert score["required_citation_coverage"] == 1.0
    assert score["fabricated_citation_ids"] == ["source_forged"]
    assert score["citations_within_retrieval"] is False


def test_empty_unanswerable_response_with_explicit_abstention_is_recognized() -> None:
    score = score_answer(
        question_id="q_unanswerable",
        answer="",
        citation_ids=[],
        abstained=True,
        retrieved_source_ids=["irrelevant_retrieved_source"],
        required_source_ids=[],
        protocol=_PROTOCOL,
    )

    assert score["abstention_expected"] is True
    assert score["abstention_correct"] is True
    assert score["lexical_rubric_coverage"] is None
    assert score["required_citation_coverage"] is None


def test_missing_gold_concept_reduces_lexical_coverage_despite_valid_citation() -> None:
    score = score_answer(
        question_id="q_answerable",
        answer="Alpha is covered, but the second concept is absent.",
        citation_ids=["source_required"],
        abstained=False,
        retrieved_source_ids=["source_required"],
        required_source_ids=["source_required"],
        protocol=_PROTOCOL,
    )

    assert score["lexical_rubric_coverage"] == 0.5
    assert score["required_citation_coverage"] == 1.0
    assert score["fabricated_citation_ids"] == []
