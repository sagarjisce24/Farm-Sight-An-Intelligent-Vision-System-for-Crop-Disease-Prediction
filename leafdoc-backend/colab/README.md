# Training LeafDoc on Google Colab

This guide walks you through training the **plant disease classifier** AND the **leaf-vs-not-leaf gate** on Google Colab's free T4 GPU, then bringing the trained artifacts back to your local FastAPI backend.

> **Why Colab?** The PlantVillage dataset has ~70k images; training MobileNetV3Large on CPU/integrated graphics takes hours and uses lots of RAM. Colab gives you a free T4 GPU that finishes the full two-stage training in ~45–75 minutes.

---

## 1. Prerequisites

- A Google account (for Colab + Drive).
- A Kaggle account.
- ~1 GB free Drive space (for the dataset cache + model artifacts).

---

## 2. Get a Kaggle API token

1. Sign in at https://www.kaggle.com/.
2. Click your avatar → **Settings**.
3. Scroll to the **API** section → **Create New API Token**. A `kaggle.json` file downloads.
4. Keep this file handy — you will upload it to Colab in step 4.

> Treat `kaggle.json` like a password. Don't commit it.

---

## 3. Open the notebook in Colab

You have two ways:

**Option A — From your local clone:**
1. Go to https://colab.research.google.com/.
2. **File → Upload notebook** → choose `leafdoc-backend/colab/train_on_colab.ipynb`.

**Option B — From GitHub (after you push):**
- `File → Open notebook → GitHub` and paste your repo URL, then pick `leafdoc-backend/colab/train_on_colab.ipynb`.

Once open, set the runtime:

- **Runtime → Change runtime type → Hardware accelerator: GPU (T4)**.

---

## 4. Run the notebook top-to-bottom

The notebook has 12 sections. Run them in order with `Shift+Enter`. Highlights:

| Section | What it does | Approx. time |
|---|---|---|
| 1 | Verify GPU + print TF/Keras versions | <1 min |
| 2 | Mount Google Drive at `/content/drive/MyDrive/leafdoc/` | <1 min |
| 3 | Upload `kaggle.json` when prompted | <1 min |
| 4 | Download + unzip the dataset (~700 MB) | 2–5 min |
| 5 | Build dataset pipeline + disease model | <1 min |
| 6 | **Stage 1 training** — 10 epochs, frozen base | ~25–35 min |
| 7 | **Stage 2 training** — 5 epochs, fine-tune | ~10–15 min |
| 8 | Evaluate (validation accuracy + top-3) | ~1 min |
| 9 | Build + train the leaf-vs-not-leaf gate | ~5–10 min |
| 10 | Calibrate the `LEAF_THRESHOLD` value | <1 min |
| 11 | Sanity inference on one validation image | <1 min |
| 12 | Trigger downloads of the three artifacts | depends on connection |

> **Don't close the tab during training.** Colab will pause your runtime if it thinks you've stepped away. Keep the browser tab focused or use a tab-keepalive extension.

---

## 5. Note the values the notebook prints

Two values printed by the notebook need to land in your local config:

1. **TensorFlow version** (printed in section 1). Open `leafdoc-backend/requirements.txt` and replace the `tensorflow` line with the exact pinned version. This guarantees the `.keras` file you trained loads cleanly locally.

2. **Suggested `LEAF_THRESHOLD`** (printed in section 10). Open `leafdoc-backend/.env` and update:

   ```
   LEAF_THRESHOLD=<the suggested value>
   ```

---

## 6. Bring the artifacts to your local backend

After section 12 runs, three files download automatically:

- `plant_disease_model.keras`
- `class_indices.json`
- `leaf_classifier.keras`

They are also saved in your Drive at `MyDrive/leafdoc/` for next time.

Place all three into:

```
leafdoc-backend/models/
├── plant_disease_model.keras
├── class_indices.json
└── leaf_classifier.keras
```

(`disease_info.json` is already in that folder — it ships with the repo.)

Then start the backend:

```bash
cd leafdoc-backend
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Hit `http://localhost:8000/health` to confirm both models loaded.

---

## 7. Iterating

If you want to retrain later:
- Re-run only sections 6–7 (Stage 1+2) for a new disease classifier.
- Re-run section 9 for a new leaf gate.
- Drive holds your previous artifacts — they get overwritten by `ModelCheckpoint` only when validation accuracy improves.

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `nvidia-smi` reports no GPU | Runtime → Change runtime type → GPU |
| "Disconnected" mid-training | Colab session timed out. Re-run from section 5; checkpoints in Drive are intact |
| Kernel keeps restarting / "could not allocate pinned host of size" warnings in logs | Host RAM OOM (Colab free tier ~12 GB). The notebook already keeps the pipeline lean (no `.shuffle()` buffer, `.prefetch(buffer_size=2)`, no `val_ds.cache()`). If it still OOMs, lower `BATCH_SIZE` from 32 to 16 in section 5, then `Runtime → Restart session` and re-run |
| `RandomFlip` / `RandomRotation` error: `module 'keras' has no attribute 'KerasTensor'` | TF 2.20 + Keras 3.13 bug. Already worked around in this notebook by moving augmentation into the `tf.data` pipeline. If you still hit it, make sure you re-ran the dataset-prep cell after pulling latest |
| Out-of-memory during training | Lower `BATCH_SIZE` from 32 to 16 in section 5; restart the runtime |
| Kaggle download fails with 403 | Re-create your `kaggle.json` token; ensure you accepted the dataset's competition rules on its Kaggle page |
| Locally `tf.keras.models.load_model` fails | Your local TF version doesn't match Colab's. Pin it in `requirements.txt` exactly as printed in notebook section 1 |
| Final accuracy is poor (<80%) | Run more epochs, or unfreeze deeper layers in stage 2 (`FINETUNE_UNFREEZE_FROM = -60`) |

---

## 9. Manual fallback

If you can't use Colab and you've trained the model elsewhere (or downloaded a pretrained `.keras` file from a colleague), just drop the three files into `leafdoc-backend/models/`. The backend doesn't care where they came from — it only needs the file shape and class_indices to match.
