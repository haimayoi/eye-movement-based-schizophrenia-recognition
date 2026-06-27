# 🧠 Eye Movement-Based Schizophrenia Recognition

## Hierarchical 5-Tier Framework for Eye Movement Analysis

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org)
[![PyG](https://img.shields.io/badge/PyG-2.4%2B-green.svg)](https://pyg.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Schizophrenia recognition from free-viewing eye movement data (EMS dataset, 208 subjects) using a 5-tier hierarchical framework: spatiotemporal graph preprocessing → contextual feature engineering → tabular baseline (Tier 3) → deep sequential model (Tier 4) → meta-learner ensemble (Tier 5). Target: exceed MSNet SOTA (ACC=81.25%, AUC=88.54%).

---

## 📋 Table of Contents

- [Framework Overview](#-framework-overview)
- [Architecture (5-Tier)](#-architecture-5-tier)
- [Dataset](#-dataset-ems)
- [Installation](#-installation)
- [Directory Structure](#-directory-structure)
- [Usage](#-usage)
- [Results & Evaluation](#-results--evaluation)
- [Explainability](#-explainability)
- [Reproducibility](#-reproducibility)
- [References](#-references)

---

## 🎯 Framework Overview

```
╔══════════════════════════════════════════════════════════════════════╗
║  TIER 1  │ Preprocessing — Excel → Spatial/Temporal Filter → Parquet║
║          │ FIX_PUPIL preserved (pupil diameter as biomarker)        ║
╠══════════════════════════════════════════════════════════════════════╣
║  TIER 2  │ Feature Engineering                                       ║
║          │ • Stimulus-level (15 features per trial)                  ║
║          │ • Contextual Delta Features (Δ_Social-Natural, etc.)      ║
║          │   → features_subject_level.csv  [NOW USED in training]   ║
╠══════════════════════════════════════════════════════════════════════╣
║  TIER 3  │ Tabular Baseline                                          ║
║          │ XGBoost / LightGBM / CatBoost on [stimulus + delta feats]║
║          │ GroupKFold(4) → OOF P_tab for Tier 5                     ║
╠══════════════════════════════════════════════════════════════════════╣
║  TIER 4  │ Deep Sequential Model                                     ║
║          │ BiCA-HS: Transformer + Bidirectional Cross-Attention      ║
║          │ OR: GNN-CEFAM: GAT + Cross-Attention Enhanced Fusion      ║
║          │ Input: [stimulus + delta feats] + raw fixation sequence  ║
║          │ GroupKFold(4) → OOF P_bica for Tier 5                    ║
╠══════════════════════════════════════════════════════════════════════╣
║  TIER 5  │ Meta-Learner Ensemble [NEW]                               ║
║          │ Input: [P_tab, P_bica] (OOF only — no leakage)          ║
║          │ Strategy A: Optuna-optimised weighted average             ║
║          │ Strategy B: L2-regularised Logistic Regression           ║
║          │ → Final P(SZ) + SHAP attribution + calibration curves    ║
╚══════════════════════════════════════════════════════════════════════╝
```

### Comparison with SOTA

| Method | ACC | AUC | Notes |
|---|:---:|:---:|---|
| MSNet (SOTA, 2025) | 81.25% | 88.54% | CNN + Mean-Shift clustering |
| **Tier 3: Tabular Baseline** | OOF result | OOF result | XGBoost + delta features |
| **Tier 4: BiCA-HS / GNN-CEFAM** | OOF result | OOF result | Deep sequential model |
| **Tier 5: Meta-Learner** | OOF result | OOF result | Ensemble fusion |

> ⚠️ Claimed results (ACC=90%, AUC=94.98%) referenced in earlier drafts were post-hoc threshold-optimised on the validation set. All metrics above are reported at threshold=0.5 on OOF predictions.

---

## 🏗 Architecture (5-Tier)

### Tier 1 — Preprocessing

```
Raw Excel (.xlsx)
    ↓
Spatial filter:  remove FIX_X ∉ [0,1024) or FIX_Y ∉ [0,768)
Temporal filter: remove FIX_DURATION < 50 ms
    ↓
clean_fixations.parquet (snappy-compressed, preserves FIX_PUPIL)
```

**Key design choice**: Loading from raw `.xlsx` preserves `FIX_PUPIL` (pupil diameter), which is lost in the original EMS text-file conversion pipeline.

### Tier 2 — Feature Engineering

**Tier 2A — Stimulus-Level Features** (15 per trial):

| Group | Features |
|---|---|
| Fixation Statistics | Count, Duration mean/median/std |
| Saccade Dynamics | Amplitude $A=\sqrt{(\Delta x)^2+(\Delta y)^2}$, Velocity, Turning angle |
| Scanpath Geometry | Total length, Convex hull area, Center bias, Spatial entropy |
| Pupil Dynamics | Mean, Std, CV of FIX_PUPIL |

**Tier 2B — Contextual Delta Features** (60 per subject, cross-category contrasts):

| Delta pair | Clinical meaning |
|---|---|
| $\Delta_{\text{Social-Natural}}$ | Social scene processing deficit |
| $\Delta_{\text{Manipulated-Natural}}$ | Cognitive manipulation response |
| $\Delta_{\text{Social-Synthetic}}$ | Social vs abstract contrast |
| $\Delta_{\text{Manipulated-Synthetic}}$ | Manipulation awareness contrast |

These 60 delta features and 60 category-mean features (120 features total) are saved to `features_subject_level.csv` and are **broadcast to every trial** of each subject during Tier 3 and Tier 4 training. Feature dimension in training: 15 (stimulus) + 120 (subject-level) = **135 features**.

### Tier 3 — Tabular Baseline

```
features_stimulus_level.csv + features_subject_level.csv (broadcast)
    ↓
GroupKFold(4) by Subject_ID → 4 training folds
    ↓
XGBoost / LightGBM / CatBoost
    ↓
OOF predictions → results/baselines/{model}_oof_subject_preds.csv
```

### Tier 4 — Deep Model

**Option A: BiCA-HS** (Bidirectional Cross-Attention Hybrid Stream):
```
Raw fixation sequence [N, 5]     Expert features [75]
        ↓                               ↓
Transformer Encoder [B,N,128]    BiomarkerStream [B,128]
        └──────────┬─────────────────────┘
                   ↓
         Bidirectional Cross-Attention
         z_fused_1: Expert → Sequence
         z_fused_2: Sequence → Expert
                   ↓
              Classifier → P(SZ)
```

**Option B: GNN-CEFAM** (Fixed — no longer degenerate):
```
Scanpath graph                   Expert features [75]
    ↓                                   ↓
GATConv ×2                       HandcraftedStream [B,128]
    ↓                                   ↓
h_nodes [total_nodes, 256]       z_expert [B, 128]
(pre-pooling, exposed)                  |
    └──────────── CEFAM ────────────────┘
       Direction 1: Q=z_expert, K/V=h_nodes  ← [B,1,N] saliency map
       Direction 2: Q=z_graph_pool, K/V=z_expert (asymmetric, non-sequence)
                    ↓
               z_final [B, 256] → Classifier
```

> **Critical fix**: CEFAM previously used `unsqueeze(1)` on both streams,
> making softmax over `seq_len=1` trivially = 1.0 (no learning signal).
> Fixed: Direction 1 now uses pre-pooling GNN node embeddings `h_nodes [total_nodes, 256]`
> as Keys/Values, producing a genuine `[B, 1, N]` fixation saliency map.

### Tier 5 — Meta-Learner (New)

```
OOF P_tab  [N]  (from Tier 3, same GroupKFold splits)
OOF P_bica [N]  (from Tier 4, same GroupKFold splits)
    ↓
Strategy A: w_tab * P_tab + w_bica * P_bica   (Optuna, 500 trials)
Strategy B: σ(β_tab * P_tab + β_bica * P_bica + b)  (LR, L2, C=1.0)
Best selected by OOF AUC
    ↓
Final P(SZ) + calibration + SHAP attribution
```

**No leakage**: Tier 5 trains only on OOF predictions. Each subject is held out
exactly once during Tier 3/4 training. Leakage is enforced by assertion checks.

---

## 📊 Dataset EMS

| Property | Details |
|---|---|
| **Name** | EMS (Eye Movement for Schizophrenia) |
| **Source** | [GitHub - YingjieSong1/EMS](https://github.com/YingjieSong1/EMS) |
| **Paper** | Song et al., IEEE TNNLS, 2025 |
| **Scale** | 104 SZ + 104 HC = 208 subjects (160 train/valid, 48 test) |
| **Paradigm** | Free-viewing |
| **Stimuli** | 100 images × 4 categories: Social, Natural, Synthetic, Manipulated |
| **Resolution** | 1024 × 768 pixels |
| **Data format** | Raw `.xlsx` (preserves FIX_PUPIL) |
| **Visual features** | `feature_dict_ResNet50.npy` (2048-dim per fixation from ResNet50 backbone) |

### Key data fields

| Field | Description |
|---|---|
| `Subject_ID` | Subject identifier |
| `Stimulus_ID` | Image identifier (1-100) |
| `FIX_X` | Fixation x coordinate (pixels) |
| `FIX_Y` | Fixation y coordinate (pixels) |
| `FIX_DURATION` | Fixation duration (ms) |
| `FIX_PUPIL` | Pupil diameter |
| `FIX_INDEX` | Chronological fixation order |
| `Label` | 0=HC, 1=SZ |

---

## ⚙️ Installation

### Requirements

| Component | Minimum | Recommended |
|---|---|---|
| **Python** | 3.9+ | 3.10+ |
| **GPU** | CUDA 11.7+ | T4/V100 (Colab Pro) |
| **RAM** | 12GB | 16GB+ |

### Setup

```bash
# 1. Create virtual environment
python -m venv venv
.\venv\Scripts\activate   # Windows
source venv/bin/activate  # Linux/Mac

# 2. Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install PyTorch Geometric
pip install torch-geometric

# 4. Install all dependencies
pip install -r requirements.txt

# 5. Verify
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 📁 Directory Structure

```
Eye Movement-Based Schizophrenia Recognition/
│
├── README.md
├── requirements.txt
│
├── configs/
│   ├── cefam_config.yaml        ← Main config (CEFAM hybrid)
│   ├── stgnn_config.yaml        ← ST-GNN standalone ablation
│   └── bica_config.yaml         ← BiCA-HS config
│
├── data/
│   ├── processed/
│   │   ├── clean_fixations.parquet          # Tier 1 output
│   │   ├── features_stimulus_level.csv      # Tier 2A (15 features/trial)
│   │   ├── features_subject_level.csv       # Tier 2B (delta features)
│   │   └── graphs/graphs.pt                 # Tier 4 PyG graph objects
│   ├── external/
│   │   └── feature_dict_ResNet50.npy        # 2048-dim ResNet50 visual features
│   └── metadata/
│       └── stimulus_categories.csv          # Image → category mapping
│
├── src/
│   ├── tier1_preprocessing/
│   ├── tier2_features/
│   ├── tier3_tabular/           ← Now includes delta features + OOF
│   ├── tier4_advanced/          ← CEFAM attention fixed
│   ├── tier5_fusion/            ← [NEW] Meta-learner ensemble
│   │   ├── meta_learner.py
│   │   ├── oof_collector.py
│   │   └── calibration.py
│   ├── explainability/          ← [NEW] SHAP, attention viz, t-SNE
│   │   ├── shap_analysis.py
│   │   ├── attention_viz.py
│   │   └── embedding_viz.py
│   └── verification/            ← [NEW] Leakage guards
│       └── leakage_checks.py
│
├── scripts/
│   ├── train_tier4.py           ← Updated: delta features + OOF
│   └── run_tier5.py             ← [NEW] Meta-learner runner
│
├── Bidirectional Cross-Attention Hybrid Stream/
│   └── scripts/train_bica.py
│
├── experiments/ablation/
│   └── run_ablation_analysis.py
│
└── results/
    ├── baselines/               ← Tier 3 OOF predictions
    ├── cefam/                   ← Tier 4 CEFAM OOF predictions
    ├── bica/                    ← Tier 4 BiCA-HS OOF predictions
    └── tier5/                   ← [NEW] Meta-learner outputs
```

---

## 🚀 Usage

### Step 0: Generate Category Mapping
```bash
python src/utils/generate_category_map.py
```

### Step 1: Tier 1 — Preprocessing
```bash
python src/tier1_preprocessing/preprocess.py --config configs/cefam_config.yaml
```

### Step 2: Tier 2 — Feature Engineering
```bash
# Stimulus-level features (15 per trial)
python src/tier2_features/stimulus_features.py --config configs/cefam_config.yaml

# Subject-level delta features (60 per subject)
python src/tier2_features/subject_aggregator.py --config configs/cefam_config.yaml
```

### Step 3: Tier 3 — Tabular Baseline (with delta features + OOF)
```bash
# Trains on [stimulus + delta] features, saves OOF for Tier 5
python src/tier3_tabular/tabular_models.py \
    --config configs/cefam_config.yaml \
    --model xgboost
```
Output: `results/baselines/xgboost_oof_subject_preds.csv`

### Step 4: Tier 4 — Deep Model (with delta features + OOF)
```bash
# Build graphs
python src/tier4_advanced/graph_builder.py --config configs/cefam_config.yaml

# Train GNN-CEFAM hybrid (saves OOF for Tier 5)
python scripts/train_tier4.py --config configs/cefam_config.yaml --seed 42

# OR: Train BiCA-HS
python "Bidirectional Cross-Attention Hybrid Stream/scripts/train_bica.py" \
    --config configs/bica_config.yaml
```
Output: `results/cefam/cefam_oof_subject_preds.csv`

### Step 5: Tier 5 — Meta-Learner Ensemble (New)
```bash
python scripts/run_tier5.py \
    --tier3-oof results/baselines/xgboost_oof_subject_preds.csv \
    --tier4-oof results/cefam/cefam_oof_subject_preds.csv \
    --output-dir results/tier5/ \
    --plot \
    --calibrate
```
Output: `results/tier5/tier5_results_summary.json`, ROC comparison, reliability diagram.

### Optional: Threshold Calibration (for analysis only — NOT for reported metrics)
```bash
# Use this for analysis ONLY. Do NOT report calibrated-threshold metrics as final results.
python scripts/calibrate_threshold.py \
    --preds results/cefam/cefam_oof_subject_preds.csv
```

---

## 📈 Results & Evaluation

### Evaluation Protocol

| Parameter | Value |
|---|---|
| CV method | GroupKFold(n_splits=4) |
| Group column | Subject_ID |
| Threshold | Fixed at 0.5 (no validation-set tuning) |
| Metrics | AUC, ACC, F1, Precision, Recall, Specificity |
| Subject aggregation | Uniform mean over 4 stimulus categories |

### Leakage Safeguards (New)

Run verification before training:
```python
from src.verification.leakage_checks import run_all_checks
run_all_checks(subject_to_fold, df_train_valid, df_test, df_fixations, n_folds=4)
```

This checks:
1. GroupKFold correctness (every subject in exactly one fold)
2. No subject in both train and val within a fold
3. No threshold optimised on validation set
4. OOF completeness (all training subjects have exactly one OOF prediction)
5. No test subjects in training OOF
6. Pupil normalisation leakage warning

---

## 🔍 Explainability

### SHAP Analysis (Feature Attribution)

```python
from src.explainability.shap_analysis import explain_xgboost, plot_shap_summary

shap_values, X_explain = explain_xgboost(xgb_model, X_val, feature_names)
plot_shap_summary(shap_values, X_explain, feature_names, "results/shap_summary.png")
```

### Attention Visualization (Scanpath Saliency)

```python
from src.explainability.attention_viz import plot_scanpath_attention

plot_scanpath_attention(
    x_coords=x, y_coords=y,
    gnn_attn=gnn_attn_weights,
    cefam_attn=cefam_attn_1.squeeze(),  # [N] per-fixation saliency
    output_path="results/attention_scanpath.png"
)
```

After the CEFAM fix, `cefam_attn_1` has shape `[B, 1, N]` — a genuine distribution over `N` fixation nodes. Previously this was trivially 1.0 (degenerate at `seq_len=1`).

### t-SNE / UMAP Embeddings

```python
from src.explainability.embedding_viz import plot_embeddings
plot_embeddings(z_final, labels, output_path="results/tsne_embeddings.png")
```

### Calibration Curves

```python
from src.tier5_fusion.calibration import plot_reliability_diagram
plot_reliability_diagram(y_true, {"Tier3": p_tab, "Tier4": p_bica, "Tier5": p_final},
                         "results/reliability.png")
```

---

## 🔒 Reproducibility

### Required files (not included in repo)

| File | Required for | Note |
|---|---|---|
| `EMS/Train_Valid.xlsx` | All training | GroupKFold CV splits |
| `EMS/Train_Valid/*.xlsx` | Tier 1 | Raw fixation data |
| `data/external/feature_dict_ResNet50.npy` | Tier 4 GNN | 2048-dim ResNet50 features — generate with `src/utils/extract_resnet_features.py` |
| `data/metadata/stimulus_categories.csv` | Tier 2/3/4 | Generate with `generate_category_map.py` |

> ⚠️ `feature_dict_ResNet50.npy` is required. Generate it by running `python src/utils/extract_resnet_features.py` (requires EMS/Images). Without it, `graph_builder.py` falls back to `np.random.randn(N, 2048)` — random noise that produces meaningless GNN results.

### Seed

```bash
# Set seed consistently across all tiers
export PYTHONHASHSEED=42
python scripts/train_tier4.py --seed 42
```

---

## 📚 References

1. Song, Y. et al. (2025). "EMS: A Large-Scale Eye Movement Dataset, Benchmark, and New Model for Schizophrenia Recognition." *IEEE TNNLS*, 36(5), 9451–9462.
2. Veličković, P. et al. (2018). "Graph Attention Networks." *ICLR 2018*.
3. Lin, T.Y. et al. (2017). "Focal Loss for Dense Object Detection." *ICCV 2017*.
4. Birawo, B. & Kasprowski, P. (2024). "Graph Neural Networks for Scanpath Classification." *gazeRE, ETRA 2024*.
5. Hollmann, N. et al. (2025). "TabPFN: A Transformer That Solves Small Tabular Classification Problems." *Nature*.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
