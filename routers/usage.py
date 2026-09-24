"""Model usage recorded by this backend process, grouped by product feature."""

from fastapi import APIRouter

from services.usage import usage_ledger


router = APIRouter()


@router.get("/usage")
async def model_usage():
    """模型用量：provider 返回的 token 数，按触发它的功能归类。

    Registered behind the authentication gate: usage reveals what the learner
    has been doing. Counts cover this process since it started; the separate
    knowledge-base service is excluded and no currency amount is estimated.
    """
    return usage_ledger.snapshot()
