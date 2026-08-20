"""Request-bound scope guards for model-selected business tools."""

from collections.abc import Callable


BusinessToolScopeGuard = Callable[[str, dict], str | None]


def _valid_scope_value(value: object, *, max_length: int) -> bool:
    return bool(
        isinstance(value, str)
        and bool(value.strip())
        and len(value) <= max_length
        and not any(ord(character) < 32 or ord(character) == 127 for character in value)
    )


def build_business_tool_scope_guard(
    *,
    user_id: str,
    document_id: str | None,
    allow_unbound_document_selection: bool = False,
) -> BusinessToolScopeGuard:
    """Bind model-supplied identity fields to the enclosing API request.

    An unbound request denies document-scoped calls by default. Callers with
    an established dynamic-selection contract must opt in explicitly. Once a
    document is bound, an explicit tool argument cannot escape it.
    """

    def guard(_tool_name: str, arguments: dict) -> str | None:
        if not isinstance(arguments, dict):
            return "invalid_tool_arguments"
        if "user_id" in arguments:
            supplied_user_id = arguments["user_id"]
            if not _valid_scope_value(supplied_user_id, max_length=128):
                return "invalid_tool_arguments"
            if supplied_user_id != user_id:
                return "user_scope_mismatch"
        if "document_id" in arguments:
            supplied_document_id = arguments["document_id"]
            if not _valid_scope_value(supplied_document_id, max_length=512):
                return "invalid_tool_arguments"
            if document_id is None and not allow_unbound_document_selection:
                return "document_scope_unbound"
            if document_id is not None and supplied_document_id != document_id:
                return "document_scope_mismatch"
        return None

    return guard
