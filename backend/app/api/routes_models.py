from fastapi import APIRouter

from backend.app.core.config import MODELS

router = APIRouter(prefix="/models", tags=["models"])


@router.get("")
def get_models():
    return MODELS
