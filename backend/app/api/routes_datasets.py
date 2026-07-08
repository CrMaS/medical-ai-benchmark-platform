from fastapi import APIRouter

from backend.app.core.config import DATASETS

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("")
def get_datasets():
    return DATASETS
