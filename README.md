# Multilingual Health Question Answering in Low-Resource African Languages

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ReneNtabana/health-qa-african-languages/blob/main/notebooks/African_Health_QA_Challenge_colab_ready.ipynb)

Final project for the **Multilingual Health Question Answering in Low-Resource African Languages Challenge** (Zindi, co-organized by ITU, HASH, and Makerere University Centre for AI).

## Overview

This project builds a question-answering system for maternal, sexual, and reproductive health questions across English, Akan, Luganda, Swahili, and Amharic. Rather than fine-tuning a generative model from scratch, the core approach is **retrieval-based**: a per-language hybrid index combines TF-IDF (lexical) and sentence-embedding (semantic) similarity to find the most similar previously-answered question for any new query, then returns that answer.

A supplementary experiment fine-tunes `google/mt5-small` as a generative alternative, to compare retrieval against generation under the same evaluation protocol.

**Why retrieval first:** with a training set of real, pre-written, expert-style health answers, copying the answer to the most similar existing question is a strong, fast, low-risk baseline — before paying the cost (and instability risk) of fine-tuning a multilingual generative model on a relatively small dataset.

## Results Summary

| Experiment | Change | Val ROUGE-1 | Val ROUGE-L |
|---|---|---|---|
| E01 | TF-IDF only baseline | 0.393 | 0.336 |
| E02 | Semantic-only retrieval | 0.413 | 0.361 |
| E03 | Fixed-weight hybrid (0.35 TF-IDF / 0.65 semantic) | 0.433 | 0.377 |
| E04 | Per-subset tuned hybrid weights | 0.460 | 0.407 |
| E05 | Exact-match lookup + tuned fallback | 0.460 | 0.407 |
| E06 | Deduplicated training corpus | 0.460 | 0.407 |
| **E07** | **Char-level TF-IDF for low-resource subsets** | **0.461** | **0.408** |
| E08 | Cross-subset fallback for low-resource languages | 0.459 | 0.406 |
| E09 | mT5-small fine-tune (2 epochs, 25% data) | 0.080 | 0.071 |
| E10 | Final comparison — selects best config | — | — |
| E12 | Encoder swap to `multilingual-e5-large` | ~0.49 | ~0.44 |

**Final submission:** Experiment E07 (char-level TF-IDF hybrid retrieval) — Zindi leaderboard score: **0.544**

Full experiment log with rationale and insights: [`outputs/experiment_tracker.csv`](outputs/experiment_tracker.csv)

## Repository Structure

```
health-qa-african-languages/
├── README.md
├── requirements.txt
├── notebooks/
│   └── African_Health_QA_Challenge_colab_ready.ipynb   # main pipeline, runs on Colab or Kaggle
├── outputs/
│   ├── submission_E01.csv ... submission_E08.csv       # per-experiment submissions
│   ├── submission_FINAL.csv                            # best config (E07)
│   └── experiment_tracker.csv                          # full experiment log
├── screenshots/
│   └── leaderboard_*.png                               # Zindi score progression
└── report/
    └── (final PDF report — submitted via course platform)
```

## How to Reproduce

### Option 1 — Google Colab (recommended for grading)
1. Click the **Open in Colab** badge above.
2. Upload `Train.csv`, `Val.csv`, `Test.csv`, `SampleSubmission.csv` (from the [Zindi competition data page](https://zindi.africa)) to any folder in your Google Drive.
3. Run all cells in order. The notebook auto-detects Colab, mounts your Drive, and locates the CSVs automatically — no path editing needed.
4. Outputs (submissions, experiment log) are written to `/content/outputs/`.

### Option 2 — Kaggle
1. Upload the notebook to a new Kaggle notebook.
2. Add the competition CSVs as a Kaggle Dataset, then attach it via **Add Input**.
3. Enable a GPU accelerator (T4 x2 recommended) for the E09 fine-tuning experiment — all retrieval experiments (E01–E08) run fine on CPU.
4. Run all cells in order.

### Option 3 — Local
```bash
git clone https://github.com/ReneNtabana/health-qa-african-languages.git
cd health-qa-african-languages
pip install -r requirements.txt
jupyter notebook notebooks/African_Health_QA_Challenge_colab_ready.ipynb
```
Place the four competition CSVs in the repo root (or any subfolder) before running — the notebook's `find_data_dir()` will locate them automatically.

**Note on data:** competition CSVs are not included in this repository (Zindi's terms prohibit redistribution). Download them directly from the [competition page](https://zindi.africa) after joining.

## Methodology Summary

- **Preprocessing:** Whitespace normalization, null/empty-row filtering. Minimal text cleaning was deliberately chosen to avoid destroying diacritics and script-specific characters in Akan and Amharic.
- **Retrieval architecture:** Per-subset (language) hybrid index — TF-IDF (word n-grams for most subsets, character n-grams for Amharic/Akan) blended with sentence-embedding cosine similarity (`paraphrase-multilingual-MiniLM-L12-v2`).
- **Weight tuning:** Per-subset grid search over the TF-IDF/semantic blend ratio, evaluated on the official `Val.csv` split (never trained on).
- **Fine-tuning experiment:** `google/mt5-small`, single epoch on a 25% data sample, greedy decoding — deliberately constrained to fit available compute, included to satisfy the project's fine-tuning demonstration requirement rather than as the primary strategy.
- **Evaluation:** ROUGE-1 F1 and ROUGE-L F1 with whitespace tokenization (matching the competition's official scoring approach), computed on the held-out `Val.csv`.

## Ethical Considerations

This system retrieves answers from a fixed corpus of training data rather than generating novel medical advice, which limits — but does not eliminate — the risk of returning an inaccurate or out-of-context answer for a genuinely novel question. Health information in low-resource languages carries real consequences for misinformation; this system is a research prototype and is not validated for clinical or public deployment.

## Acknowledgments

- Zindi, ITU, HASH, and Makerere University Centre for AI for organizing the competition and providing the dataset.
- The official competition starter notebook, which informed the language-code mapping and initial baseline approach.
