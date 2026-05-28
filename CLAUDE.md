# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single coursework lab for *Tópicos en Inteligencia Artificial*: compare an MLP
against a CNN on the [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
dataset (6 classes, ~14k train, ~3k test). All work lives in one notebook —
`CNN_vs_MLP.ipynb`. `TIA_CNN (1).pdf` is the assignment statement (in Spanish).

## Environment

The virtualenv is **one level up**, shared across labs: `../../.venv` (i.e.
`labs2/.venv`). Python 3.11, TensorFlow 2.21 (CPU-only on macOS), plus
scikit-learn, seaborn, pandas, matplotlib, pillow, kagglehub.

The notebook is bound to a registered Jupyter kernel named `tia-labs`
(display name "Python (TIA labs)") backed by that venv. When running cells
non-interactively, use that interpreter directly:

```bash
../.venv/bin/python                                    # the interpreter
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1800 "CNN_vs_MLP.ipynb"
../.venv/bin/jupyter lab                               # interactive editing
```

## Workflow: local + Colab + Kaggle + GitHub

This lab uses a **multi-actor workflow** because the local Mac has no GPU
and `tf.data` deadlocks on CPU when iterating `image_dataset_from_directory`
batches. Pieces:

- **Mac local** → notebook editing, dataset exploration, NumPy preprocessing,
  Section 7 (no compute).
- **GitHub** `f3r21/cnn-vs-mlp-intel` → versioned notebook with outputs.
- **Kaggle** (via `kagglehub`, public dataset, no auth) → dataset (~350 MB).
- **Colab T4** → training of MLP, CNN base, CNN improved.

The first cell of the notebook detects `IS_COLAB = 'google.colab' in sys.modules`
and adapts paths/installs accordingly. The dataset cache lives in `.cache_npz/`
(gitignored) — same path in both environments.

## Notebook structure

The notebook is graded by section; preserve the question/answer ordering. It has
seven parts plus a setup cell:

0. **Setup** — env detection, global constants (SEED, IMG_SIZE, BATCH_SIZE).
1. **Dataset** — load with `image_dataset_from_directory` (for show), count
   images, sample resolutions, visualize per-class.
2. **Preprocesamiento** — `load_dataset_numpy` (PIL + cache), normalize, flatten,
   `to_categorical`. Benchmark 64/128/150 in Colab.
3. **MLP base** — `Dense(512) → Dense(256) → Dense(128) → Dense(6)` (~25M params).
4. **CNN base** — 3× `Conv + MaxPool` → `Dense(128) → Dropout(0.5) → Dense(6)` (~700K params).
5. **Entrenamiento y Evaluación** — comparison table, confusion matrices,
   correct/incorrect samples, training curves.
6. **Optimización** — CNN + BatchNorm + Data Augmentation (the two chosen
   improvements). Final comparison table `[MLP, CNN, CNN+]`.
7. **Investigación** — ResNet50: architecture, params, advantages, cost,
   applications. No training.

Each section ends with markdown cells answering the assignment's conceptual
questions in Spanish prose. The answers must reference real numbers/plots from
the preceding code cells.

## Architecture conventions to keep consistent

- **Working resolution**: 128×128 (mid-point of the PDF's 64/128/150 range,
  divisible by 2 three times — fits the 3 Conv+Pool layers cleanly).
- **Loss**: `categorical_crossentropy` with `to_categorical` labels. **This
  differs from Lab 1**, which used `sparse_categorical_crossentropy` with
  integer labels. The PDF explicitly requires `categorical_crossentropy`.
- **Preprocessing**: centralized in `load_dataset_numpy(split_dir, img_size, cache=True)`
  (S2), `normalize(X)`, `flatten_for_mlp(X)`. Reuse these — do not reshape inline.
- **Helpers reused across sections** (defined inline, no `.py` module):
  - `train_with_timer(model, ...)` → S3, S4, S6
  - `plot_training_history(history, title)` → S3, S4, S6
  - `evaluate_and_confusion(model, X, y, names, title)` → S5, S6
  - `show_examples(model, X, y, names, kind, n)` → S5
- **Base models** mandated by the PDF — do not change shapes:
  - MLP: `49152 → 512 → 256 → 128 → 6`, ReLU, softmax
  - CNN: 3 conv blocks with filters `[32, 64, 128]`, kernel `3×3`, ReLU,
    MaxPool 2×2 after each conv, then `Flatten → Dense(128) → Dropout(0.5) → Dense(6)`
- **Optimizer**: Adam (default lr). Same across all 3 models for a fair compare.
- **Reproducibility**: every model cell calls `tf.random.set_seed(42)` and
  `np.random.seed(42)` before building. Keep this when adding or editing
  model cells.
- `CLASS_NAMES` (defined in S0) is the canonical label list — index it rather
  than hardcoding strings elsewhere.
- **Saved weights** (`cnn_base.weights.h5`, `cnn_plus.weights.h5`) are
  gitignored. Download from Colab and put in `labs2/2/` to skip retraining
  in S5/S7.

## Working notes

- Answer text is Spanish and intentionally written without accents in places —
  match the surrounding markdown style.
- When editing code cells, also refresh their stored outputs (re-execute) so
  the submitted notebook stays self-consistent; stale outputs are a grading
  hazard.
- Target notebook size: <8 MB. If it grows, reduce `figsize` of grids,
  set `plt.rcParams['figure.dpi'] = 80`, and limit `n=9` in `show_examples`.
- `tf.data` (image_dataset_from_directory + .take()) **deadlocks** on this
  Mac in CPU. Use NumPy in-memory (`load_dataset_numpy`) for any real
  iteration. Keep the `image_dataset_from_directory` call in S1 only as a
  cardinality probe (do not iterate from it).
