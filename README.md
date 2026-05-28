# Lab 2 — CNN vs MLP con Intel Image Classification

Práctica de la materia *Tópicos en Inteligencia Artificial* (UCSP).
Comparación entre un MLP y una CNN sobre el dataset
[Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
(6 clases: buildings, forest, glacier, mountain, sea, street).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/f3r21/cnn-vs-mlp-intel/blob/main/CNN_vs_MLP.ipynb)

## Estructura

Todo el trabajo vive en un único notebook: `CNN_vs_MLP.ipynb`. Sigue las 7 secciones
de la guía (`TIA_CNN (1).pdf`):

1. Dataset — carga y exploración
2. Preprocesamiento — resize, normalización, flatten
3. MLP base
4. CNN base
5. Entrenamiento y evaluación comparada
6. Optimización (BatchNorm + Data Augmentation)
7. Investigación (ResNet50)

## Cómo correr

### Opción A — Colab (recomendado para entrenamiento)

Haz clic en el badge de arriba. La primera celda detecta Colab e instala las
dependencias automáticamente. Activa GPU T4: `Runtime → Change runtime type → T4 GPU`.

### Opción B — Local

```bash
# venv compartido un nivel arriba (Python 3.11)
cd "/Users/99/2026-1/Tópicos en Inteligencia Artificial/labs2/2"
../.venv/bin/pip install -r requirements.txt
../.venv/bin/jupyter lab
```

El entrenamiento de la CNN puede tardar horas en CPU; recomendado solo para
exploración (S1-S2) y la sección 7 (sin cómputo).

## Workflow

- **Mac local** → edición y exploración liviana
- **GitHub** → versionado del notebook
- **Kaggle** → host del dataset (descarga vía `kagglehub`, sin auth, ~350MB)
- **Colab T4** → entrenamiento de los 3 modelos

## Créditos

- Dataset: [Puneet Bansal — Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
- Curso: Tópicos en Inteligencia Artificial — Departamento de Computación, UCSP
