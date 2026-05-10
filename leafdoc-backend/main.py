"""
LeafDoc FastAPI backend.

Endpoints
---------
GET  /                    – health ping
GET  /health              – detailed health (model_loaded, classes_count, ...)
GET  /supported-classes   – list of classes the custom model can predict, grouped by species
POST /predict             – classify an uploaded image; returns AnalysisResult or rejection
POST /qna                 – OpenRouter-backed Q&A about a previously identified disease
                            (uses the OpenAI SDK pointed at OpenRouter's free-tier endpoint)

Inference flow for /predict:
  1. Run leaf-vs-not-leaf gate. If leaf_prob < LEAF_THRESHOLD -> respond status="not_a_leaf".
  2. Run main 38-class disease classifier.
  3. If top confidence < CONFIDENCE_THRESHOLD OR entropy > ENTROPY_THRESHOLD
     -> respond status="out_of_scope".
  4. Otherwise enrich with disease_info.json and return status="ok".
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "models/plant_disease_model.keras")
LEAF_CLASSIFIER_PATH = os.getenv("LEAF_CLASSIFIER_PATH", "models/leaf_classifier.keras")
CLASS_INDICES_PATH = os.getenv("CLASS_INDICES_PATH", "models/class_indices.json")
DISEASE_INFO_PATH = os.getenv("DISEASE_INFO_PATH", "models/disease_info.json")
PORT = int(os.getenv("PORT", "8000"))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# Thresholds (tunable via .env). The Colab notebook prints a calibrated value
# for LEAF_THRESHOLD. CONFIDENCE_THRESHOLD and ENTROPY_THRESHOLD are heuristic;
# tweak them if the model rejects too aggressively or not aggressively enough.
LEAF_THRESHOLD = float(os.getenv("LEAF_THRESHOLD", "0.5"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))
ENTROPY_THRESHOLD = float(os.getenv("ENTROPY_THRESHOLD", "2.5"))  # max possible for 38 classes ≈ 3.64

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_TEXT_MODEL = os.getenv(
    "OPENROUTER_TEXT_MODEL", "nvidia/nemotron-3-super-120b-a12b:free"
)

IMG_SIZE = (224, 224)

# ----------------------------------------------------------------------------
# Module-level state (populated on startup)
# ----------------------------------------------------------------------------

disease_model: Optional[tf.keras.Model] = None
leaf_model: Optional[tf.keras.Model] = None
class_names: List[str] = []
disease_info: Dict[str, Any] = {}

# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------

app = FastAPI(
    title="LeafDoc Plant Disease Prediction API",
    description="Custom MobileNetV3 classifier for plant leaf diseases + OpenRouter-backed Q&A.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------------
# Startup
# ----------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event() -> None:
    global disease_model, leaf_model, class_names, disease_info

    if os.path.exists(MODEL_PATH):
        try:
            logger.info("Loading disease classifier from %s ...", MODEL_PATH)
            disease_model = tf.keras.models.load_model(MODEL_PATH)
            logger.info("Disease classifier loaded.")
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load disease classifier: %s", exc)
    else:
        logger.warning(
            "Disease model not found at %s. Train on Colab and drop the file in models/.",
            MODEL_PATH,
        )

    if os.path.exists(LEAF_CLASSIFIER_PATH):
        try:
            logger.info("Loading leaf gate from %s ...", LEAF_CLASSIFIER_PATH)
            leaf_model = tf.keras.models.load_model(LEAF_CLASSIFIER_PATH)
            logger.info("Leaf gate loaded.")
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load leaf gate: %s", exc)
    else:
        logger.warning(
            "Leaf classifier not found at %s. /predict will skip the leaf gate.",
            LEAF_CLASSIFIER_PATH,
        )

    if os.path.exists(CLASS_INDICES_PATH):
        with open(CLASS_INDICES_PATH, "r") as f:
            class_names = json.load(f)
        logger.info("Loaded %d class names.", len(class_names))
    else:
        logger.warning("Class indices missing at %s.", CLASS_INDICES_PATH)

    if os.path.exists(DISEASE_INFO_PATH):
        with open(DISEASE_INFO_PATH, "r") as f:
            disease_info = json.load(f)
        logger.info("Loaded disease info for %d entries.", len(disease_info) - 1)  # minus _meta
    else:
        logger.warning("Disease info JSON missing at %s.", DISEASE_INFO_PATH)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _decode_image(image_bytes: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Invalid image format") from exc
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)


def _entropy(probs: np.ndarray) -> float:
    eps = 1e-12
    return float(-np.sum(probs * np.log(probs + eps)))


def _is_healthy(class_name: str) -> bool:
    return class_name.lower().endswith("___healthy") or class_name.lower().endswith("_healthy")


def _humanize(class_name: str) -> str:
    """`Apple___Apple_scab` -> `Apple - Apple scab`."""
    if "___" in class_name:
        plant, disease = class_name.split("___", 1)
        return f"{plant.replace('_', ' ').strip()} – {disease.replace('_', ' ').strip()}"
    return class_name.replace("_", " ").strip()


def _info_for(class_name: str) -> Dict[str, Any]:
    """Look up enrichment data; gracefully handle missing entries."""
    entry = disease_info.get(class_name)
    if not entry:
        return {
            "description": "No additional information is available for this class.",
            "symptoms": [],
            "treatment": ["Consult an agricultural extension officer."],
            "prevention": ["Maintain good agricultural practices."],
            "severity_baseline": 50,
            "progression": [],
            "environmentalFactors": {
                "temperature": "Unknown",
                "humidity": "Unknown",
                "sunlight": "Unknown",
                "watering": "Unknown",
            },
        }
    return entry


def _build_analysis_result(
    class_name: str, confidence: float, top_predictions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    info = _info_for(class_name)
    healthy = _is_healthy(class_name)
    severity = 0.0 if healthy else round(info.get("severity_baseline", 50) * confidence, 1)
    return {
        "status": "ok",
        "isHealthy": healthy,
        "diseaseName": _humanize(class_name) if not healthy else "Healthy",
        "rawClassName": class_name,
        "confidence": round(confidence * 100, 2),
        "description": info.get("description", ""),
        "symptoms": info.get("symptoms", []),
        "treatment": info.get("treatment", []),
        "prevention": info.get("prevention", []),
        "severity": severity,
        "progression": info.get("progression", []),
        "environmentalFactors": info.get("environmentalFactors", {}),
        "topPredictions": top_predictions,
    }


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------

@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Plant Disease Prediction API is running"}


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "model_loaded": disease_model is not None,
        "leaf_model_loaded": leaf_model is not None,
        "classes_count": len(class_names),
        "disease_info_loaded": bool(disease_info),
        "thresholds": {
            "leaf": LEAF_THRESHOLD,
            "confidence": CONFIDENCE_THRESHOLD,
            "entropy": ENTROPY_THRESHOLD,
        },
    }


@app.get("/supported-classes")
def supported_classes() -> Dict[str, Any]:
    """Group class_indices.json by plant species so the frontend can render a coverage list."""
    by_species: Dict[str, List[str]] = {}
    for cls in class_names:
        if "___" in cls:
            plant, disease = cls.split("___", 1)
            plant_label = plant.replace("_", " ").strip()
            disease_label = disease.replace("_", " ").strip()
        else:
            plant_label, disease_label = "Other", cls
        by_species.setdefault(plant_label, []).append(disease_label)

    return {
        "total_classes": len(class_names),
        "species_count": len(by_species),
        "by_species": by_species,
        "limitations": [
            "Pests (insects, mites) other than two-spotted spider mite on tomato are not directly identified.",
            "Nutrient deficiencies (N, P, K, micronutrients) are not detected.",
            "Environmental damage (sunburn, frost, herbicide, drought stress) is not detected.",
            "Only 14 plant species are supported. Anything outside this list will be flagged as out-of-scope.",
            "Whole-plant or fruit-only conditions are typically not detected from a single leaf image.",
        ],
        "fallback_advice": "For images outside the supported list, use the Gemini provider for open-ended analysis.",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    if disease_model is None or not class_names:
        raise HTTPException(
            status_code=503,
            detail=(
                "Disease model not loaded. Train it on Colab and place "
                "plant_disease_model.keras + class_indices.json in models/."
            ),
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    img_array = _decode_image(contents)

    # Gate 1: leaf-vs-not-leaf
    leaf_prob: Optional[float] = None
    if leaf_model is not None:
        try:
            leaf_pred = leaf_model.predict(img_array, verbose=0)
            leaf_prob = float(leaf_pred.flatten()[0])
            if leaf_prob < LEAF_THRESHOLD:
                return {
                    "status": "not_a_leaf",
                    "isHealthy": False,
                    "diseaseName": "Not a leaf",
                    "rawClassName": None,
                    "confidence": round((1 - leaf_prob) * 100, 2),
                    "leafProbability": round(leaf_prob * 100, 2),
                    "message": (
                        "The image doesn't appear to contain a plant leaf. "
                        "Upload a clearer leaf photo, or use the Gemini provider for general image analysis."
                    ),
                }
        except Exception as exc:  # noqa: BLE001
            logger.error("Leaf gate inference failed: %s", exc)

    # Gate 2 + 3: disease classifier with confidence/entropy thresholds
    try:
        probabilities = disease_model.predict(img_array, verbose=0)[0]
    except Exception as exc:  # noqa: BLE001
        logger.exception("Disease model inference failed.")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    top_idx = int(np.argmax(probabilities))
    top_confidence = float(probabilities[top_idx])
    top_class = class_names[top_idx]

    top_3 = probabilities.argsort()[-3:][::-1]
    top_predictions = [
        {
            "className": class_names[i],
            "label": _humanize(class_names[i]),
            "probability": round(float(probabilities[i]) * 100, 2),
        }
        for i in top_3
    ]

    entropy = _entropy(probabilities)

    if top_confidence < CONFIDENCE_THRESHOLD or entropy > ENTROPY_THRESHOLD:
        reason = []
        if top_confidence < CONFIDENCE_THRESHOLD:
            reason.append(f"low confidence ({top_confidence * 100:.1f}%)")
        if entropy > ENTROPY_THRESHOLD:
            reason.append(f"high prediction entropy ({entropy:.2f})")
        return {
            "status": "out_of_scope",
            "isHealthy": False,
            "diseaseName": "Could not confidently identify",
            "rawClassName": None,
            "confidence": round(top_confidence * 100, 2),
            "entropy": round(entropy, 3),
            "leafProbability": round(leaf_prob * 100, 2) if leaf_prob is not None else None,
            "message": (
                f"This image looks like a leaf but the model couldn't confidently match any of the "
                f"{len(class_names)} supported classes ({'; '.join(reason)}). "
                "Try the Gemini provider for open-ended analysis."
            ),
            "topPredictions": top_predictions,
        }

    result = _build_analysis_result(top_class, top_confidence, top_predictions)
    if leaf_prob is not None:
        result["leafProbability"] = round(leaf_prob * 100, 2)
    return result


# ----------------------------------------------------------------------------
# Q&A endpoint (OpenRouter via OpenAI SDK)
# ----------------------------------------------------------------------------

class QnaMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str


class QnaRequest(BaseModel):
    disease_name: str
    question: str
    history: List[QnaMessage] = []


@app.post("/qna")
async def qna(req: QnaRequest) -> Dict[str, Any]:
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your_openrouter_api_key_here":
        raise HTTPException(
            status_code=503,
            detail="OPENROUTER_API_KEY is not configured in the backend .env file.",
        )

    # Lazy import so the backend still boots without the SDK installed
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise HTTPException(
            status_code=500,
            detail="openai package is not installed. Run `pip install -r requirements.txt`.",
        ) from exc

    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            # OpenRouter analytics headers (optional but recommended)
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "LeafDoc-Backend",
        },
    )

    system_instruction = (
        "You are LeafDoc, a friendly and knowledgeable plant pathology assistant. "
        f"The user has just been told their plant has: '{req.disease_name}'. "
        "Answer their follow-up questions about this disease — symptoms, treatment options, "
        "prevention, severity, environmental factors, organic alternatives, regional considerations, etc. "
        "Keep answers concise (2–4 short paragraphs), practical, and grounded in established agricultural science. "
        "If the user asks about something outside plant pathology, briefly redirect them. "
        "Use simple bullet points when listing actions."
    )

    # OpenAI chat-completions format. History is plain {role, content} pairs;
    # both "user" and "assistant" map directly.
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_instruction}]
    for msg in req.history:
        role = msg.role if msg.role in ("user", "assistant") else "user"
        messages.append({"role": role, "content": msg.content})
    messages.append({"role": "user", "content": req.question})

    try:
        response = client.chat.completions.create(
            model=OPENROUTER_TEXT_MODEL,
            messages=messages,  # type: ignore[arg-type]
        )
        answer = ((response.choices[0].message.content or "") if response.choices else "").strip()
        if not answer:
            answer = "I couldn't generate a response. Please try rephrasing your question."
        return {"answer": answer}
    except Exception as exc:  # noqa: BLE001
        logger.exception("OpenRouter Q&A failed.")
        raise HTTPException(status_code=502, detail=f"OpenRouter call failed: {exc}") from exc


# ----------------------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
