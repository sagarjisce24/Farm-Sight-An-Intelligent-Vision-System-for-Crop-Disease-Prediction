# LeafDoc Backend

FastAPI service that wraps a **MobileNetV3** plant-disease classifier and a
**leaf-vs-not-leaf** gate trained on Google Colab, plus an OpenRouter-backed Q&A
endpoint (using free-tier models via the OpenAI SDK) for follow-up questions
about a detected disease.

> **Heads up:** training has moved to Google Colab (free T4 GPU). See
> [`colab/README.md`](colab/README.md) for the full training walkthrough.
> The local `train_model.py` is kept only as a fallback.

---

## Architecture

```
        ┌──────────────────────────────────────────────┐
        │  POST /predict (image)                       │
        │    1. leaf_classifier.keras  →  is it a leaf?│
        │    2. plant_disease_model.keras → which class?│
        │    3. confidence + entropy thresholds        │
        │    4. enrich with disease_info.json          │
        │    →  AnalysisResult | not_a_leaf | out_of_scope
        └──────────────────────────────────────────────┘

        POST /qna  (disease_name, question, history[])
                       │
                       └──►  OpenRouter (server-side, OpenAI SDK)

        GET  /health             health + thresholds
        GET  /supported-classes  list of classes the model can predict
```

---

## Prerequisites

- Python 3.10+
- A trained model produced by [`colab/train_on_colab.ipynb`](colab/train_on_colab.ipynb)
- (Optional) An OpenRouter API key for the `/qna` endpoint — free tier works fine.
  Get one at <https://openrouter.ai/keys>.

---

## Install

```bash
cd leafdoc-backend
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Important:** after running the Colab notebook for the first time, replace the
> bare `tensorflow` line in `requirements.txt` with the exact version printed in
> the notebook (e.g. `tensorflow==2.20.0`). This avoids `.keras` load errors.

---

## Drop in the trained models

After running the Colab notebook, place these three files inside `models/`:

```
models/
├── plant_disease_model.keras   # 38-class disease classifier
├── class_indices.json          # ordered class name list
├── leaf_classifier.keras       # binary leaf-vs-not-leaf gate
└── disease_info.json           # already shipped with the repo
```

---

## Configure `.env`

```env
MODEL_PATH=models/plant_disease_model.keras
LEAF_CLASSIFIER_PATH=models/leaf_classifier.keras
CLASS_INDICES_PATH=models/class_indices.json
DISEASE_INFO_PATH=models/disease_info.json

LEAF_THRESHOLD=0.5            # use the value the Colab notebook prints
CONFIDENCE_THRESHOLD=0.60
ENTROPY_THRESHOLD=2.5

ALLOWED_ORIGINS=*

# OpenRouter (free tier) – get your key at https://openrouter.ai/keys
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_TEXT_MODEL=nvidia/nemotron-3-super-120b-a12b:free
```

---

## Run

```bash
uvicorn main:app --reload
```

The API is now at `http://localhost:8000`. Interactive docs at `/docs`.

---

## Endpoints

### `GET /health`

```json
{
  "model_loaded": true,
  "leaf_model_loaded": true,
  "classes_count": 38,
  "disease_info_loaded": true,
  "thresholds": { "leaf": 0.5, "confidence": 0.6, "entropy": 2.5 }
}
```

### `GET /supported-classes`

Lists what the custom model can recognize, grouped by plant species, plus an
explicit list of limitations.

### `POST /predict`

Multipart upload `file=<image>`. Returns one of three response shapes:

**1. `status: "ok"` — successful classification**

```json
{
  "status": "ok",
  "isHealthy": false,
  "diseaseName": "Tomato – Early blight",
  "rawClassName": "Tomato___Early_blight",
  "confidence": 92.4,
  "description": "...",
  "symptoms": ["..."],
  "treatment": ["..."],
  "prevention": ["..."],
  "severity": 50.8,
  "progression": [{"stage": "Early", "timeline": "Days 1-7..."}],
  "environmentalFactors": { "temperature": "...", "humidity": "...", "sunlight": "...", "watering": "..." },
  "topPredictions": [{"className": "...", "label": "...", "probability": 92.4}, ...],
  "leafProbability": 99.3
}
```

**2. `status: "not_a_leaf"` — leaf gate rejected**

```json
{
  "status": "not_a_leaf",
  "diseaseName": "Not a leaf",
  "leafProbability": 12.4,
  "message": "..."
}
```

**3. `status: "out_of_scope"` — confidence/entropy threshold rejected**

```json
{
  "status": "out_of_scope",
  "diseaseName": "Could not confidently identify",
  "confidence": 38.2,
  "entropy": 2.91,
  "message": "..."
}
```

The frontend uses `status` to render three distinct UIs.

### `POST /qna`

Body:

```json
{
  "disease_name": "Tomato – Early blight",
  "question": "Are there organic treatment options?",
  "history": [
    { "role": "user", "content": "Earlier user question" },
    { "role": "assistant", "content": "Earlier assistant answer" }
  ]
}
```

Returns `{ "answer": "..." }`. Calls OpenRouter server-side using `OPENROUTER_API_KEY`.

---

## Tests

```bash
pytest tests/
```

The bundled tests are smoke-level — they don't require the `.keras` files to
exist. Add real-model tests once you've placed the trained artifacts.

---

## Threshold tuning

If the API returns `out_of_scope` too often:
- Lower `CONFIDENCE_THRESHOLD` (e.g. `0.45`)
- Raise `ENTROPY_THRESHOLD` (e.g. `3.0`)

If it returns `not_a_leaf` too often on real leaves:
- Lower `LEAF_THRESHOLD` (e.g. `0.3`)

If it confidently mis-classifies obvious non-leaves:
- Raise `LEAF_THRESHOLD` (e.g. `0.7`)

The Colab notebook prints a calibrated starting value for `LEAF_THRESHOLD`.

---

## Coverage limitations

The custom model is trained on the PlantVillage "New Plant Diseases" dataset
which contains **38 classes across 14 plant species**:

> Apple, Blueberry, Cherry, Corn (maize), Grape, Orange, Peach, Bell Pepper,
> Potato, Raspberry, Soybean, Squash, Strawberry, Tomato.

It **cannot** detect:
- Plants outside this list (mango, banana, rice, wheat, etc.)
- Pests other than tomato two-spotted spider mite
- Nutrient deficiencies (N, P, K, micros)
- Environmental damage (sunburn, frost, herbicide, drought)
- Whole-plant or fruit-only conditions from a leaf photo

The leaf gate + confidence threshold + entropy threshold combine to *reject*
out-of-scope inputs rather than confidently lying. For anything outside the
supported list, use the **OpenRouter provider** in the frontend.
