from fastapi import APIRouter

from backend.app.core.config import DATASETS

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("")
async def get_datasets():
    return DATASETS
