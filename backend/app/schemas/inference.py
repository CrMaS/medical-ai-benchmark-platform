from pydantic import BaseModel, Field


class ImageMetadata(BaseModel):
    format: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class ClassPrediction(BaseModel):
    class_id: str
    display_name: str
    confidence: float = Field(ge=0, le=1)


class ClassProbability(BaseModel):
    class_id: str
    display_name: str
    probability: float = Field(ge=0, le=1)


class InferenceResponse(BaseModel):
    model_id: str
    model_sha256: str
    filename: str | None
    image: ImageMetadata
    prediction: ClassPrediction
    probabilities: list[ClassProbability]
    disclaimer: str


class ModelStatusResponse(BaseModel):
    model_id: str
    configured: bool
    loaded: bool
    artifact: str
    artifact_sha256: str | None
    device: str
    input_size: int
    class_names: list[str]
    preprocessing: dict
