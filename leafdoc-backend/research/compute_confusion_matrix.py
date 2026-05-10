"""
LeafDoc — Confusion Matrix Computer
====================================

Loads the trained plant_disease_model.keras and runs inference over the full
validation set to produce a real 38×38 confusion matrix.

Outputs
-------
    leafdoc-backend/research/output/confusion_matrix.json

    Schema:
    {
        "matrix":        [[int, ...], ...],   // 38×38
        "labels":        ["Apple___Apple_scab", ...],
        "short_labels":  ["Apple scab", ...],
        "per_class": [
            {"label": "...", "short": "...",
             "precision": float, "recall": float, "f1": float, "support": int},
            ...
        ],
        "overall_accuracy":  float,           // 0-1
        "macro_f1":          float,
        "weighted_f1":       float,
        "total_images":      int,
        "total_correct":     int,
        "class_order_source": "class_indices.json"
    }

Usage
-----
    # Mac — uses built-in default path, no argument needed:
    python research/compute_confusion_matrix.py

    # Windows — pass the validation folder explicitly:
    python research/compute_confusion_matrix.py --valid-dir "C:\\Users\\ankur\\...\\valid"

    # All overrides (any platform):
    python research/compute_confusion_matrix.py \\
        --valid-dir "C:\\path\\to\\valid" \\
        --model     "C:\\path\\to\\plant_disease_model.keras" \\
        --output    "C:\\path\\to\\output\\confusion_matrix.json"

    # Show help:
    python research/compute_confusion_matrix.py --help

Runtime
-------
    ~3-5 minutes on CPU (17,572 validation images, batch=32).
    GPU (if available) cuts this to ~30 seconds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Fixed paths (relative to this script) — overridable via CLI args below
# ---------------------------------------------------------------------------

SCRIPT_DIR   = Path(__file__).resolve().parent
BACKEND_DIR  = SCRIPT_DIR.parent

_DEFAULT_VALID_DIR = (
    BACKEND_DIR
    / "new-plant-diseases-dataset"
    / "New Plant Diseases Dataset(Augmented)"
    / "New Plant Diseases Dataset(Augmented)"
    / "valid"
)
_DEFAULT_MODEL_PATH  = BACKEND_DIR / "models" / "plant_disease_model.keras"
_DEFAULT_OUTPUT_DIR  = SCRIPT_DIR / "output"

IMG_SIZE   = (224, 224)
BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LeafDoc — compute 38×38 confusion matrix from the validation set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Mac (default paths — no arguments needed):
  python research/compute_confusion_matrix.py

  # Windows:
  python research\\compute_confusion_matrix.py --valid-dir "C:\\Users\\ankur\\datasets\\valid"

  # Override everything:
  python research/compute_confusion_matrix.py \\
      --valid-dir /data/valid \\
      --model     /data/models/plant_disease_model.keras \\
      --output    /data/output/confusion_matrix.json
        """,
    )
    p.add_argument(
        "--valid-dir",
        type=Path,
        default=_DEFAULT_VALID_DIR,
        metavar="PATH",
        help=(
            "Path to the validation dataset folder (contains one sub-folder per class). "
            f"Default: {_DEFAULT_VALID_DIR}"
        ),
    )
    p.add_argument(
        "--model",
        type=Path,
        default=_DEFAULT_MODEL_PATH,
        metavar="PATH",
        help=(
            "Path to plant_disease_model.keras. "
            f"Default: {_DEFAULT_MODEL_PATH}"
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Where to write confusion_matrix.json. "
            "Default: <script-dir>/output/confusion_matrix.json"
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Short label builder — all 38 are unique
# ---------------------------------------------------------------------------

def _make_short_labels(labels: list[str]) -> list[str]:
    """
    Strip the species prefix (everything up to and including '___').
    For classes where the disease part alone would be duplicated across
    species (i.e. 'healthy'), prepend the species name to keep uniqueness.

    Example:
        'Apple___Apple_scab'            →  'Apple scab'
        'Apple___healthy'               →  'Apple healthy'
        'Blueberry___healthy'           →  'Blueberry healthy'
        'Corn_(maize)___Common_rust_'   →  'Common rust'
    """
    def _clean(s: str) -> str:
        return s.replace("_", " ").strip()

    # Split each label into (species, disease)
    split = []
    for lbl in labels:
        if "___" in lbl:
            species, disease = lbl.split("___", 1)
        else:
            species, disease = "", lbl
        split.append((_clean(species), _clean(disease)))

    # Find disease strings that appear more than once
    from collections import Counter
    disease_counts = Counter(d for _, d in split)

    short = []
    for species, disease in split:
        if disease_counts[disease] > 1:
            short.append(f"{species} {disease}")
        else:
            short.append(disease)

    # Sanity-check: all must be unique
    assert len(set(short)) == len(short), (
        f"Short labels are not all unique: {[s for s in short if short.count(s) > 1]}"
    )
    return short


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # Resolve paths from args
    MODEL_PATH         = args.model.resolve()
    VALID_DIR          = args.valid_dir.resolve()
    CLASS_INDICES_PATH = MODEL_PATH.parent / "class_indices.json"

    if args.output is not None:
        CM_JSON_PATH = args.output.resolve()
        CM_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = _DEFAULT_OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        CM_JSON_PATH = out_dir / "confusion_matrix.json"

    print("=" * 60)
    print("LeafDoc — Confusion Matrix Computer")
    print("=" * 60)
    print(f"  Model     : {MODEL_PATH}")
    print(f"  Valid dir : {VALID_DIR}")
    print(f"  Output    : {CM_JSON_PATH}")
    print()

    # ------------------------------------------------------------------
    # Pre-flight checks
    # ------------------------------------------------------------------
    if not MODEL_PATH.exists():
        sys.exit(f"ERROR: model not found at {MODEL_PATH}\n"
                 "Run the Colab notebook first and drop the .keras file into models/.")

    if not CLASS_INDICES_PATH.exists():
        sys.exit(f"ERROR: class_indices.json not found at {CLASS_INDICES_PATH}")

    if not VALID_DIR.exists():
        sys.exit(f"ERROR: validation directory not found:\n  {VALID_DIR}")

    # ------------------------------------------------------------------
    # Load class labels
    # ------------------------------------------------------------------
    print("Loading class_indices.json …")
    with CLASS_INDICES_PATH.open() as f:
        raw = json.load(f)

    # class_indices.json may be a list or a {name: index} dict
    if isinstance(raw, list):
        class_labels: list[str] = raw
    elif isinstance(raw, dict):
        # Sort by integer value to get ordered list
        class_labels = [k for k, v in sorted(raw.items(), key=lambda kv: kv[1])]
    else:
        sys.exit("ERROR: unrecognised class_indices.json format")

    n_classes = len(class_labels)
    print(f"  {n_classes} classes loaded")

    # ------------------------------------------------------------------
    # Verify validation directory matches class_indices.json
    # ------------------------------------------------------------------
    disk_classes = sorted(d.name for d in VALID_DIR.iterdir() if d.is_dir())
    json_sorted  = sorted(class_labels)

    if disk_classes != json_sorted:
        missing_in_json = set(disk_classes) - set(json_sorted)
        missing_on_disk = set(json_sorted)  - set(disk_classes)
        msg = "ERROR: mismatch between validation folders and class_indices.json\n"
        if missing_in_json:
            msg += f"  On disk but not in JSON: {missing_in_json}\n"
        if missing_on_disk:
            msg += f"  In JSON but not on disk: {missing_on_disk}\n"
        sys.exit(msg)

    print(f"  Validation folder matches class_indices.json ✓")

    # ------------------------------------------------------------------
    # Load TensorFlow (late import — gives a clear error if not installed)
    # ------------------------------------------------------------------
    print("Importing TensorFlow …")
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # suppress C++ info logs
        import tensorflow as tf                                # type: ignore
    except ImportError:
        sys.exit(
            "ERROR: TensorFlow is not installed.\n"
            "Activate the venv and run:  pip install -r requirements.txt"
        )
    print(f"  TensorFlow {tf.__version__}")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"Loading model from {MODEL_PATH.name} …")
    t0 = time.time()
    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    print(f"  loaded in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # Build validation dataset
    # CRITICAL: shuffle=False so label order matches filesystem traversal
    # image_dataset_from_directory uses alphabetical class ordering by default
    # which matches our class_indices.json (also alphabetical).
    # ------------------------------------------------------------------
    print("Building validation dataset …")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        str(VALID_DIR),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        label_mode="int",
    )

    # Grab the class names as discovered by Keras (alphabetical)
    keras_class_names: list[str] = val_ds.class_names

    # Verify Keras order matches class_indices.json order
    if keras_class_names != class_labels:
        # If class_indices.json happened to be stored differently, remap
        print("  WARNING: Keras class order differs from class_indices.json — remapping …")
        # Build a mapping: keras index → class_indices index
        keras_to_json = [class_labels.index(name) for name in keras_class_names]
    else:
        keras_to_json = None   # identical — no remapping needed

    print(f"  {len(keras_class_names)} classes, using labels from class_indices.json")

    # ------------------------------------------------------------------
    # Run inference
    # ------------------------------------------------------------------
    total_images = sum(1 for _ in VALID_DIR.rglob("*") if _.suffix.lower() in {".jpg",".jpeg",".png"})
    print(f"Running inference on {total_images:,} images (batch={BATCH_SIZE}) …")

    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    processed = 0
    t_start = time.time()

    for images, true_labels_keras in val_ds:
        preds = model.predict(images, verbose=0)          # (batch, 38) softmax
        pred_indices = np.argmax(preds, axis=1)           # (batch,)
        true_indices = true_labels_keras.numpy()          # (batch,)

        if keras_to_json is not None:
            pred_indices = np.array([keras_to_json[i] for i in pred_indices])
            true_indices = np.array([keras_to_json[i] for i in true_indices])

        for true_i, pred_i in zip(true_indices, pred_indices):
            confusion[true_i, pred_i] += 1

        processed += len(true_indices)
        elapsed = time.time() - t_start
        eta = (elapsed / processed) * (total_images - processed) if processed else 0
        print(f"  {processed:>6,} / {total_images:,}  "
              f"({processed/total_images*100:.1f}%)  "
              f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s",
              end="\r", flush=True)

    print()  # newline after \r progress

    elapsed_total = time.time() - t_start
    print(f"Inference complete in {elapsed_total:.1f}s")

    # ------------------------------------------------------------------
    # Compute per-class precision, recall, F1
    # ------------------------------------------------------------------
    print("Computing metrics …")

    tp  = np.diag(confusion).astype(float)
    fp  = confusion.sum(axis=0) - tp          # column sum minus diagonal
    fn  = confusion.sum(axis=1) - tp          # row sum minus diagonal
    support = confusion.sum(axis=1).astype(int)

    precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    recall    = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
    f1        = np.where(precision + recall > 0,
                         2 * precision * recall / (precision + recall),
                         0.0)

    total_correct = int(tp.sum())
    total_imgs    = int(confusion.sum())
    overall_acc   = total_correct / total_imgs

    macro_f1    = float(f1.mean())
    weighted_f1 = float(np.sum(f1 * support) / total_imgs)

    print(f"  Overall accuracy : {overall_acc*100:.2f}%")
    print(f"  Macro F1         : {macro_f1*100:.2f}%")
    print(f"  Weighted F1      : {weighted_f1*100:.2f}%")
    print(f"  Total correct    : {total_correct:,} / {total_imgs:,}")

    # ------------------------------------------------------------------
    # Build short labels
    # ------------------------------------------------------------------
    short_labels = _make_short_labels(class_labels)
    print(f"  Short label uniqueness check passed ✓")

    # ------------------------------------------------------------------
    # Serialize
    # ------------------------------------------------------------------
    per_class = []
    for i, lbl in enumerate(class_labels):
        per_class.append({
            "label":     lbl,
            "short":     short_labels[i],
            "precision": round(float(precision[i]), 6),
            "recall":    round(float(recall[i]),    6),
            "f1":        round(float(f1[i]),        6),
            "support":   int(support[i]),
        })

    payload = {
        "matrix":           confusion.tolist(),
        "labels":           class_labels,
        "short_labels":     short_labels,
        "per_class":        per_class,
        "overall_accuracy": round(overall_acc, 6),
        "macro_f1":         round(macro_f1,    6),
        "weighted_f1":      round(weighted_f1, 6),
        "total_images":     total_imgs,
        "total_correct":    total_correct,
        "class_order_source": "class_indices.json",
    }

    CM_JSON_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved → {CM_JSON_PATH}")
    print("Run generate_paper_assets.py to include pages 18 & 19 in the PDF.")


if __name__ == "__main__":
    main()
