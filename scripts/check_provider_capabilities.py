"""Run explicit, potentially billable checks against configured providers."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.provider_capabilities import ProviderCapabilityChecker


def main() -> int:
    result = asyncio.run(ProviderCapabilityChecker().check())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
