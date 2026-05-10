"""Smoke tests for the FastAPI endpoints.

These intentionally avoid loading actual ML models – they validate routing,
request validation and edge cases. Real-model end-to-end tests should be added
once a `.keras` file is dropped in `models/`.
"""

import io
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import main
from main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# ---------------------------------------------------------------------------
# /
# ---------------------------------------------------------------------------

def test_root(client: TestClient) -> None:
    res = client.get("/")
    assert res.status_code == 200
    assert res.json() == {"message": "Plant Disease Prediction API is running"}


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_shape(client: TestClient) -> None:
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    for key in (
        "model_loaded",
        "leaf_model_loaded",
        "classes_count",
        "disease_info_loaded",
        "thresholds",
    ):
        assert key in body, f"missing key {key!r} in /health response"
    assert {"leaf", "confidence", "entropy"}.issubset(body["thresholds"].keys())


# ---------------------------------------------------------------------------
# /supported-classes
# ---------------------------------------------------------------------------

def test_supported_classes(client: TestClient) -> None:
    res = client.get("/supported-classes")
    assert res.status_code == 200
    body = res.json()
    assert "by_species" in body
    assert "limitations" in body
    assert body["total_classes"] == len(main.class_names)


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------

def _make_png_bytes(size: tuple[int, int] = (32, 32)) -> bytes:
    img = Image.new("RGB", size, color=(0, 128, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_predict_no_file(client: TestClient) -> None:
    res = client.post("/predict")
    assert res.status_code == 422  # FastAPI validation error


def test_predict_non_image(client: TestClient) -> None:
    res = client.post(
        "/predict",
        files={"file": ("foo.txt", b"not an image", "text/plain")},
    )
    # If model not loaded -> 503; if loaded -> 400 due to non-image content type.
    assert res.status_code in (400, 503)


def test_predict_with_image_when_model_missing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force "model not loaded" code path
    monkeypatch.setattr(main, "disease_model", None)
    monkeypatch.setattr(main, "class_names", [])
    res = client.post(
        "/predict",
        files={"file": ("leaf.png", _make_png_bytes(), "image/png")},
    )
    assert res.status_code == 503


# ---------------------------------------------------------------------------
# /qna
# ---------------------------------------------------------------------------

def test_qna_validation_missing_fields(client: TestClient) -> None:
    res = client.post("/qna", json={})
    assert res.status_code == 422


def test_qna_without_openrouter_key(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "OPENROUTER_API_KEY", "")
    payload: Dict[str, Any] = {
        "disease_name": "Tomato – Early blight",
        "question": "How do I treat this organically?",
    }
    res = client.post("/qna", json=payload)
    assert res.status_code == 503
    assert "OPENROUTER_API_KEY" in res.json()["detail"]
