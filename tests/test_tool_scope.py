"""Pure request-scope contracts shared by tool-enabled endpoints."""

import unittest

from services.tool_scope import build_business_tool_scope_guard


class TestBusinessToolScopeGuard(unittest.TestCase):
    def test_rejects_non_object_arguments(self):
        guard = build_business_tool_scope_guard(user_id="u", document_id="d")

        self.assertEqual(guard("search_document", []), "invalid_tool_arguments")

    def test_rejects_malformed_explicit_scope_values(self):
        guard = build_business_tool_scope_guard(user_id="u", document_id=None)

        for arguments in (
            {"user_id": None},
            {"user_id": " "},
            {"user_id": "u\nother"},
            {"document_id": None},
            {"document_id": 1},
            {"document_id": " "},
            {"document_id": "doc\tother"},
            {"document_id": "x" * 513},
        ):
            with self.subTest(arguments=arguments):
                self.assertEqual(
                    guard("search_document", arguments),
                    "invalid_tool_arguments",
                )

    def test_rejects_explicit_cross_user_and_cross_document_arguments(self):
        guard = build_business_tool_scope_guard(
            user_id="selected-user",
            document_id="selected.md",
        )

        self.assertEqual(
            guard("get_user_profile", {"user_id": "other-user"}),
            "user_scope_mismatch",
        )
        self.assertEqual(
            guard("search_document", {"document_id": "other.md"}),
            "document_scope_mismatch",
        )

    def test_allows_matching_scope_and_requires_opt_in_for_unbound_selection(self):
        bound = build_business_tool_scope_guard(
            user_id="selected-user",
            document_id="selected.md",
        )
        denied_unbound = build_business_tool_scope_guard(
            user_id="selected-user",
            document_id=None,
        )
        allowed_unbound = build_business_tool_scope_guard(
            user_id="selected-user",
            document_id=None,
            allow_unbound_document_selection=True,
        )

        self.assertIsNone(bound(
            "update_learning_profile",
            {"user_id": "selected-user", "document_id": "selected.md"},
        ))
        self.assertEqual(denied_unbound(
            "search_document",
            {"document_id": "model-selected.md"},
        ), "document_scope_unbound")
        self.assertIsNone(allowed_unbound(
            "search_document",
            {"document_id": "model-selected.md"},
        ))


if __name__ == "__main__":
    unittest.main(verbosity=2)
