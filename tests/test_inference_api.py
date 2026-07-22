import httpx
import pytest

from backend.app.main import app
from backend.app.ml.inference import MedicalImageClassifier, get_classifier_dependency
from tests.test_inference import (
    CLASS_NAMES,
    SevenClassFixture,
    make_png,
    save_exported_model,
)


def override_classifier(classifier):
    async def dependency():
        return classifier

    return dependency


@pytest.mark.anyio
async def test_image_inference_api(tmp_path):
    model_path = tmp_path / "classifier.pt2"
    save_exported_model(model_path, SevenClassFixture())
    classifier = MedicalImageClassifier(
        model_path=model_path,
        class_names=CLASS_NAMES,
        input_size=16,
        device="cpu",
    )
    app.dependency_overrides[get_classifier_dependency] = override_classifier(classifier)

    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            status_response = await client.get("/inference/status")
            assert status_response.status_code == 200
            assert status_response.json()["configured"] is True

            response = await client.post(
                "/inference/skin-lesion",
                files={"image": ("lesion.png", make_png(), "image/png")},
            )
            assert response.status_code == 200
            assert response.json()["prediction"]["class_id"] == "vasc"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_image_inference_api_rejects_invalid_upload(tmp_path):
    classifier = MedicalImageClassifier(
        model_path=tmp_path / "missing.ts",
        class_names=CLASS_NAMES,
        input_size=16,
        device="cpu",
    )
    app.dependency_overrides[get_classifier_dependency] = override_classifier(classifier)

    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            unsupported = await client.post(
                "/inference/skin-lesion",
                files={"image": ("notes.txt", b"text", "text/plain")},
            )
            assert unsupported.status_code == 415

            invalid = await client.post(
                "/inference/skin-lesion",
                files={"image": ("fake.png", b"not an image", "image/png")},
            )
            assert invalid.status_code == 422

            missing_model = await client.post(
                "/inference/skin-lesion",
                files={"image": ("lesion.png", make_png(), "image/png")},
            )
            assert missing_model.status_code == 503
    finally:
        app.dependency_overrides.clear()
