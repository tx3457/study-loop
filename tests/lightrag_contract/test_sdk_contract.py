from __future__ import annotations

import inspect
import importlib
from importlib.metadata import version

from .support import (
    EXPECTED_DIST_VERSION,
    REQUIRED_METHOD_PARAMETERS,
    REQUIRED_POSTGRES_STORAGES,
)


def test_pinned_lightrag_distribution_is_installed() -> None:
    assert version("lightrag-hku") == EXPECTED_DIST_VERSION


def test_approved_postgres_storage_classes_are_available() -> None:
    from lightrag.kg import STORAGES

    missing: list[str] = []
    for name in REQUIRED_POSTGRES_STORAGES:
        module_name = STORAGES.get(name)
        if module_name is None:
            missing.append(name)
            continue
        module = importlib.import_module(module_name, package="lightrag")
        if not hasattr(module, name):
            missing.append(name)
    assert not missing, (
        "approved storage contract is unavailable: missing "
        f"{missing}; PGGraphStorage is not an authorized substitute for "
        "PGTableGraphStorage"
    )


def test_required_async_sdk_methods_keep_the_approved_parameters() -> None:
    from lightrag import LightRAG

    mismatches: list[str] = []
    for method_name, required_parameters in REQUIRED_METHOD_PARAMETERS.items():
        method = getattr(LightRAG, method_name, None)
        if method is None:
            mismatches.append(f"{method_name}: method absent")
            continue
        actual = inspect.signature(method).parameters
        missing = [name for name in required_parameters if name not in actual]
        if missing:
            mismatches.append(f"{method_name}: missing {missing}")
    assert not mismatches, "approved LightRAG API mismatch: " + "; ".join(mismatches)
