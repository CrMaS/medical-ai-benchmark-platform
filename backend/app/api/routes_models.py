from fastapi import APIRouter

from backend.app.core.config import MODELS
from backend.app.ml.inference import get_classifier

router = APIRouter(prefix="/models", tags=["models"])


@router.get("")
async def get_models():
    models = [dict(model) for model in MODELS]
    medical_model = next(
        model for model in models if model["id"] == "skin-lesion-pytorch"
    )
    model_status = get_classifier().status()
    medical_model["status"] = (
        "loaded"
        if model_status["loaded"]
        else "available"
        if model_status["configured"]
        else "requires_configuration"
    )
    return models
