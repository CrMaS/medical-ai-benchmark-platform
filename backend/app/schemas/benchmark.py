from pydantic import BaseModel, Field


class RandomBenchmarkRequest(BaseModel):
    num_samples: int = Field(
        default=200,
        ge=10,
        le=10000,
        description="Number of synthetic samples to evaluate.",
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducible benchmark runs.",
    )
