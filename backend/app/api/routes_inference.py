from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from backend.app.core.config import MAX_IMAGE_BYTES
from backend.app.ml.inference import (
    InvalidImageError,
    MedicalImageClassifier,
    ModelLoadError,
    ModelNotConfiguredError,
    ModelOutputError,
    get_classifier_dependency,
)
from backend.app.schemas.inference import InferenceResponse, ModelStatusResponse

router = APIRouter(prefix="/inference", tags=["inference"])
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


@router.get("/status", response_model=ModelStatusResponse)
async def inference_status(
    classifier: MedicalImageClassifier = Depends(get_classifier_dependency),
):
    return classifier.status()


@router.post("/skin-lesion", response_model=InferenceResponse)
async def classify_skin_lesion(
    image: UploadFile = File(description="JPEG, PNG, or WebP dermoscopic image"),
    classifier: MedicalImageClassifier = Depends(get_classifier_dependency),
):
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only JPEG, PNG, and WebP images are supported",
        )

    try:
        image_bytes = await image.read(MAX_IMAGE_BYTES + 1)
    finally:
        await image.close()
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Image exceeds the {MAX_IMAGE_BYTES}-byte upload limit",
        )

    try:
        return classifier.predict(image_bytes, image.filename)
    except ModelNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except InvalidImageError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelLoadError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ModelOutputError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
