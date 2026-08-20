"""Strict public contract for generated learning paths."""

import copy
import unittest

from pydantic import ValidationError

from models.learning_path import LearningPath, LearningPathWire


def _valid_path() -> dict:
    return {
        "document_id": "notes.md",
        "title": "RAG 学习路径",
        "total_stages": 2,
        "stages": [
            {
                "stage": 1,
                "title": "检索基础",
                "topics": ["BM25", "向量检索"],
                "description": "理解两类检索信号。",
                "estimated_minutes": 20,
            },
            {
                "stage": 2,
                "title": "融合与评估",
                "topics": ["RRF", "Recall@K"],
                "description": "组合检索结果并验证召回质量。",
                "estimated_minutes": 30,
            },
        ],
    }


class TestLearningPathModel(unittest.TestCase):
    def test_valid_path_is_canonicalized(self) -> None:
        payload = _valid_path()
        payload["title"] = "  RAG 学习路径  "
        payload["stages"][0]["topics"] = ["  BM25  ", "向量检索"]

        path = LearningPath.model_validate(payload)

        self.assertEqual(path.title, "RAG 学习路径")
        self.assertEqual(path.stages[0].topics, ["BM25", "向量检索"])

    def test_unusable_or_inconsistent_paths_are_rejected(self) -> None:
        cases = {
            "blank_document": lambda value: value.update(document_id="   "),
            "blank_title": lambda value: value.update(title="   "),
            "zero_total": lambda value: value.update(total_stages=0),
            "empty_stages": lambda value: value.update(stages=[]),
            "count_mismatch": lambda value: value.update(total_stages=1),
            "out_of_order": lambda value: value["stages"][1].update(stage=3),
            "non_positive_stage": lambda value: value["stages"][0].update(stage=0),
            "blank_stage_title": lambda value: value["stages"][0].update(title=" "),
            "empty_topics": lambda value: value["stages"][0].update(topics=[]),
            "blank_topic": lambda value: value["stages"][0].update(topics=[" "]),
            "duplicate_topics": lambda value: value["stages"][0].update(
                topics=["RAG", " rag "]
            ),
            "blank_description": lambda value: value["stages"][0].update(
                description=" "
            ),
            "non_positive_minutes": lambda value: value["stages"][0].update(
                estimated_minutes=-1
            ),
            "boolean_minutes": lambda value: value["stages"][0].update(
                estimated_minutes=True
            ),
            "unexpected_field": lambda value: value.update(debug="provider detail"),
        }

        for name, mutate in cases.items():
            with self.subTest(name=name):
                payload = copy.deepcopy(_valid_path())
                mutate(payload)
                with self.assertRaises(ValidationError):
                    LearningPath.model_validate(payload)

    def test_revalidation_rejects_a_mutated_nested_stage(self) -> None:
        path = LearningPath.model_validate(_valid_path())
        path.stages[0].estimated_minutes = -1

        with self.assertRaises(ValidationError):
            LearningPath.model_validate(path)

    def test_provider_wire_schema_omits_unsupported_constraint_keywords(self) -> None:
        schema = LearningPathWire.model_json_schema()
        unsupported = {
            "minLength",
            "maxLength",
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "minItems",
            "maxItems",
        }

        def collect_keys(value: object) -> set[str]:
            if isinstance(value, dict):
                keys = set(value)
                for child in value.values():
                    keys.update(collect_keys(child))
                return keys
            if isinstance(value, list):
                keys: set[str] = set()
                for child in value:
                    keys.update(collect_keys(child))
                return keys
            return set()

        self.assertTrue(unsupported.isdisjoint(collect_keys(schema)))
        self.assertFalse(schema["additionalProperties"])
        stage_schema = schema["$defs"]["LearningStageWire"]
        self.assertFalse(stage_schema["additionalProperties"])

    def test_provider_wire_model_does_not_coerce_boolean_or_string_integers(self) -> None:
        payload = _valid_path()
        payload["stages"][0]["estimated_minutes"] = True
        with self.assertRaises(ValidationError):
            LearningPathWire.model_validate(payload)

        payload = _valid_path()
        payload["total_stages"] = "2"
        with self.assertRaises(ValidationError):
            LearningPathWire.model_validate(payload)


if __name__ == "__main__":
    unittest.main()
