# Farm-Sight-An-Intelligent-Vision-System-for-Crop-Disease-Prediction

Identify plant diseases from a leaf photo using either:
- **OpenRouter (free AI)** — open-ended, recognizes most plants/conditions
- **Custom MobileNetV3 classifier** — trained on Google Colab, served locally via FastAPI; covers 38 classes across 14 plant species, with a leaf-vs-not-leaf gate to reject obviously out-of-scope images

Plus an OpenRouter-backed **Q&A chatbot** for follow-up questions about whatever disease was detected.

---

## Architecture

```
┌─ Frontend (Next.js, Server Actions only) ──────────────────────────┐
│                                                                    │
│   ┌──────────────────────┐                                         │
│   │  Provider toggle     │                                         │
│   │  ── OpenRouter ──────┼──►  Server Action (SSR)                 │
│   │                      │       │                                 │
│   │                      │       └──►  OpenRouter API (server-side)│
│   │                      │                                         │
│   │  ── Custom Model ────┼──►  Server Action (SSR)                 │
│   │                      │       │                                 │
│   └──────────────────────┘       └──►  FastAPI /predict            │
│                                          │                         │
│   ┌──────────────────────┐               ├─► leaf gate model       │
│   │  Q&A Chat            │               ├─► disease classifier    │
│   │  about detected      │               └─► confidence/entropy    │
│   │  disease             │                   thresholds            │
│   │                      │                                         │
│   │                      ├──►  Server Action (SSR)                 │
│   │                      │       │                                 │
│   │                      │       └──►  FastAPI /qna ─► OpenRouter  │
│   └──────────────────────┘                                         │
└────────────────────────────────────────────────────────────────────┘
```

**SSR posture:** every OpenRouter API call and every backend URL reference is
confined to Next.js Server Actions (`"use server"`). The browser never sees
the OpenRouter key or the backend URL.

---

## Repo Layout

```
.
├── leafdoc-backend/              FastAPI + ML models (Python)
│   ├── colab/                    Google Colab training notebook + guide
│   │   ├── train_on_colab.ipynb
│   │   ├── README.md             ← read this BEFORE training
│   │   └── requirements-colab.txt
│   ├── models/
│   │   ├── disease_info.json     ships with the repo (enrichment data)
│   │   ├── class_indices.json    written by Colab
│   │   ├── plant_disease_model.keras   ← drop here after Colab
│   │   └── leaf_classifier.keras       ← drop here after Colab
│   ├── tests/
│   ├── main.py                   FastAPI app (/predict, /qna, /health, ...)
│   ├── train_model.py            DEPRECATED local trainer (kept as fallback)
│   ├── requirements.txt
│   ├── .env                      configuration (placeholders by default)
│   └── README.md
│
├── leafdoc-frontend/             Next.js app
│   ├── app/
│   │   ├── actions/              Server Actions (SSR boundary)
│   │   │   ├── analyze.ts          provider router (OpenRouter | Custom)
│   │   │   ├── qna.ts              proxies to FastAPI /qna
│   │   │   ├── health.ts           proxies to FastAPI /health
│   │   │   ├── supported-classes.ts proxies to FastAPI /supported-classes
│   │   │   └── recommend.ts        crop recommendations (existing)
│   │   ├── page.tsx              main page (provider toggle + upload)
│   │   └── recommendations/      existing crop-recommendations page
│   ├── components/
│   │   ├── provider-toggle.tsx   OpenRouter ↔ Custom Model
│   │   ├── disease-chat.tsx      follow-up Q&A chat
│   │   ├── supported-classes-info.tsx   coverage info dialog
│   │   ├── analysis-result.tsx   3-state result UI (ok/not_a_leaf/out_of_scope)
│   │   └── …
│   ├── lib/types.ts             shared TYPES only (no SDK calls here)
│   └── .env.local                placeholders by default
│
└── README.md                     (this file)
```

---

## Quickstart

You have two paths depending on whether you want to use the custom local model.

### Path A — Demo with OpenRouter only (no Colab, no backend)

1. **Install Bun** (or use npm/yarn/pnpm).
2. ```bash
   cd leafdoc-frontend
   bun install
   ```
3. Edit `leafdoc-frontend/.env.local` — replace `your_openrouter_api_key_here` with a real key from <https://openrouter.ai/keys> (free tier works).
4. ```bash
   bun dev
   ```
5. Open <http://localhost:3000> → keep the **OpenRouter** toggle selected → upload a leaf photo.

The custom-model toggle will show "Backend unreachable" — that's expected without the backend. The Q&A chat below the result also requires the backend (since Q&A is proxied through FastAPI), so for a pure OpenRouter-only demo the chat won't work.

### Path B — Full setup with the custom model

1. **Train on Colab** — follow [`leafdoc-backend/colab/README.md`](leafdoc-backend/colab/README.md). Expect ~45–75 minutes.
2. Drop the three artifacts (`plant_disease_model.keras`, `class_indices.json`, `leaf_classifier.keras`) into `leafdoc-backend/models/`.
3. **Set up the backend:**
   ```bash
   cd leafdoc-backend
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   Pin `tensorflow==<version>` in `requirements.txt` to whatever version the Colab notebook printed in section 1.
4. Edit `leafdoc-backend/.env` — replace `your_openrouter_api_key_here` with your OpenRouter key (used for the `/qna` chatbot). Also update `LEAF_THRESHOLD` to the value the notebook suggested in section 10.
5. ```bash
   uvicorn main:app --reload
   ```
6. **Set up the frontend:**
   ```bash
   cd ../leafdoc-frontend
   bun install
   ```
   Edit `leafdoc-frontend/.env.local` — set `OPENROUTER_API_KEY` (only needed if you also want to use the OpenRouter provider toggle) and confirm `BACKEND_API_URL=http://localhost:8000`.
7. ```bash
   bun dev
   ```
8. Open <http://localhost:3000>. Toggle **Custom Model** → upload a leaf → see the rich result + chat with OpenRouter about it.

---

## Provider behavior

| Toggle | Image classification | Q&A chat |
|---|---|---|
| OpenRouter | Server Action → OpenRouter API directly | Server Action → FastAPI `/qna` → OpenRouter |
| Custom Model | Server Action → FastAPI `/predict` (leaf gate + 38-class classifier) | Server Action → FastAPI `/qna` → OpenRouter |

The chat is always backend-routed — the frontend never calls Gemini directly for chat.

---

## Custom-model coverage & limitations

The custom model is trained on PlantVillage's "New Plant Diseases (Augmented)" dataset:

**14 plant species:** Apple, Blueberry, Cherry, Corn (maize), Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato.

**38 total classes** including healthy variants.

The custom-model **CANNOT** detect:
- Plants outside the 14-species list (mango, banana, rice, wheat, …)
- Pests other than tomato two-spotted spider mite
- Nutrient deficiencies (N, P, K, micros)
- Environmental damage (sunburn, frost, herbicide, drought)
- Whole-plant or fruit-only conditions from a single leaf photo

To prevent confidently-wrong predictions, the backend applies three rejection layers in order:

1. **Leaf gate** — `leaf_classifier.keras` returns `not_a_leaf` if the image probably isn't a leaf
2. **Confidence threshold** — `out_of_scope` if top class confidence < `CONFIDENCE_THRESHOLD` (default 0.60)
3. **Entropy threshold** — `out_of_scope` if predictions are spread too uniformly (default 2.5; max ≈ 3.64)

The frontend renders distinct UIs for each case and offers a one-click **"Re-analyze with OpenRouter"** button.

The full coverage list and limitations are also exposed at:
- Frontend → click "What can the custom model detect?" beside the toggle
- Backend → `GET http://localhost:8000/supported-classes`

---

## Tuning thresholds

Edit `leafdoc-backend/.env`:

| Variable | Default | What it does |
|---|---|---|
| `LEAF_THRESHOLD` | `0.5` | Below this, `not_a_leaf`. Calibrated by the Colab notebook |
| `CONFIDENCE_THRESHOLD` | `0.60` | Below this, `out_of_scope` |
| `ENTROPY_THRESHOLD` | `2.5` | Above this, `out_of_scope` |

Lower thresholds → fewer rejections, more risk of confidently wrong predictions.
Higher thresholds → more rejections, safer but more "use OpenRouter instead" prompts.

---

## Running tests

```bash
cd leafdoc-backend
source .venv/bin/activate
pytest tests/
```

Smoke tests only — they don't require the trained `.keras` files.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| Frontend shows "Backend unreachable" for Custom Model | Run `uvicorn main:app --reload` in `leafdoc-backend` |
| `/predict` returns 503 "Disease model not loaded" | The `.keras` files aren't in `leafdoc-backend/models/`. Train on Colab and drop them in |
| `tf.keras.models.load_model` errors locally | Your local TF version doesn't match Colab's. Pin it in `requirements.txt` to the version the notebook printed |
| Q&A chat fails with "OPENROUTER_API_KEY is not configured" | Set `OPENROUTER_API_KEY` in `leafdoc-backend/.env` |
| Custom model rejects everything as `out_of_scope` | Lower `CONFIDENCE_THRESHOLD` (e.g. `0.45`) or raise `ENTROPY_THRESHOLD` (e.g. `3.0`) |
| Custom model rejects real leaves as `not_a_leaf` | Lower `LEAF_THRESHOLD` (e.g. `0.3`) |
| Colab session disconnects mid-training | The `ModelCheckpoint` saved the best weights to Drive. Re-run from where you stopped |

---

## Manual model setup (no Colab)

If you already have a `.keras` file (from a colleague or a previous training run):

1. Place it as `leafdoc-backend/models/plant_disease_model.keras`.
2. Place the matching `class_indices.json` next to it.
3. (Optional) Place a `leaf_classifier.keras`. If absent, the backend skips the leaf gate but the confidence/entropy thresholds still apply.
4. Make sure your local TensorFlow version matches whatever produced the file.
5. Start the backend.

---

## Security notes

- Both `.env` files contain placeholders — replace them before deploying anywhere.
- All OpenRouter calls and the backend URL are kept off the client by routing through Next.js Server Actions.
- CORS is currently `*`. Tighten it (`ALLOWED_ORIGINS=http://localhost:3000`) if you expose the backend publicly.
- The OpenRouter API key in `leafdoc-frontend/.env.local` is used by the OpenRouter provider's image-analysis path (server-side). The Q&A chat uses the backend's separate `OPENROUTER_API_KEY`.
