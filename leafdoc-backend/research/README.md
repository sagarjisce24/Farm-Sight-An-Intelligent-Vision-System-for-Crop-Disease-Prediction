# Research Paper Asset Generator

`generate_paper_assets.py` produces a multi-page PDF + per-figure PNGs + machine-readable tables that you can drop into a research paper about LeafDoc.

The numbers are extracted from the actual Colab training run (TensorFlow 2.20.0, Keras 3.13.2, Tesla T4, May 2026). They are NOT regenerated — the script does not depend on TensorFlow being installed.

## Run

```bash
# from repo root
uv pip install -p leafdoc-backend/.venv/bin/python matplotlib pygments
leafdoc-backend/.venv/bin/python leafdoc-backend/research/generate_paper_assets.py
open leafdoc-backend/research/output/leafdoc_research_assets.pdf
```

## Output

```
leafdoc-backend/research/output/
├── leafdoc_research_assets.pdf      # 17-page PDF, primary deliverable
├── training_history.json            # parsed numbers, machine-readable
├── tables.csv                       # all tables in CSV form
└── figures/
    ├── 01_title.png
    ├── 02_abstract.png
    ├── ...
    └── 17_end_to_end_system.png
```

All figures are 300 DPI, suitable for direct insertion into LaTeX/Word.

## Limitations

- No confusion matrix — would require per-class validation predictions, which the script can't generate without the trained `.keras` files and a working TF environment. Listed as future work in the PDF.
- The leaf-gate score-distribution histogram (page 7) is synthesized from published summary statistics (mean, min, max from the calibration cell). The figure caption discloses this.
