# 🧠 Eye Movement-Based Schizophrenia Recognition

## Hybrid Context-Aware Eye Movement Recognition Framework

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org)
[![PyG](https://img.shields.io/badge/PyG-2.4%2B-green.svg)](https://pyg.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Nhận dạng Tâm thần phân liệt (Schizophrenia) qua phân tích chuyển động mắt sử dụng framework phân tầng kết hợp Spatiotemporal GNN và Bidirectional Cross-Attention (CEFAM), nhằm vượt qua SOTA hiện tại (MSNet — 81.25% ACC, 88.54% AUC).**

---

## 📋 Mục Lục

- [Tổng Quan](#-tổng-quan)
- [Kiến Trúc 4 Phân Tầng](#-kiến-trúc-4-phân-tầng)
- [Bộ Dữ Liệu EMS](#-bộ-dữ-liệu-ems)
- [Cài Đặt](#-cài-đặt)
- [Cấu Trúc Thư Mục](#-cấu-trúc-thư-mục)
- [Hướng Dẫn Sử Dụng](#-hướng-dẫn-sử-dụng)
- [Kết Quả & Đánh Giá](#-kết-quả--đánh-giá)
- [Kịch Bản Thực Hiện](#-kịch-bản-thực-hiện)
- [Tham Khảo](#-tham-khảo)

---

## 🎯 Tổng Quan

### Framework Nghiên Cứu

Dự án triển khai **Hybrid Context-Aware Eye Movement Recognition Framework** — một hệ thống phân tầng gồm **4 tầng kỹ thuật** độc lập nhưng tương hỗ chặt chẽ, đi từ xử lý tín hiệu thô đến biểu diễn đồ thị không-thời gian nâng cao:

```
╔═══════════════════════════════════════════════════════════════════════╗
║  TIER 1  │ Tiền Xử Lý — Excel → Bộ lọc Không-Thời gian → Parquet  ║
║          │ Bảo toàn FIX_PUPIL (động lực học đồng tử)               ║
╠═══════════════════════════════════════════════════════════════════════╣
║  TIER 2  │ Feature Engineering Ngữ Cảnh Đa Tầng                    ║
║          │ • Stimulus-level: Fixation + Saccade + Scanpath + Pupil  ║
║          │ • Contextual Delta: Δ_Social-Natural, Δ_Manip-Natural    ║
╠═══════════════════════════════════════════════════════════════════════╣
║  TIER 3  │ Tabular Leaderboard + Probability Aggregation            ║
║          │ GroupKFold(4) + Optuna weights [α,β,γ,δ]                 ║
║          │ XGBoost / LightGBM / CatBoost / TabPFN / AutoGluon      ║
╠═══════════════════════════════════════════════════════════════════════╣
║  TIER 4  │ Advanced: Spatiotemporal GNN + CEFAM Fusion              ║
║          │ 2-layer GAT + RINet(1056→64) + Bidirectional Cross-Attn  ║
║          │ Focal Loss + Entropy Sparsity Regularization             ║
╚═══════════════════════════════════════════════════════════════════════╝
```

### So sánh với SOTA

| Phương pháp | ACC | AUC | Đặc điểm |
|---|:---:|:---:|---|
| MSNet (SOTA, 2025) | 81.25% | 88.54% | CNN + Mean-Shift clustering |
| **Tier 3: Tabular Baseline** | **~80-84%** | **~86-90%** | Context-weighted ensemble |
| **Tier 4: GNN + CEFAM** | **>85%** | **>90%** | Graph topology + Cross-attention |

---

## 🏗 Kiến Trúc 4 Phân Tầng

### Phân Tầng 1: Tiền Xử Lý — Bảo Toàn Động Lực Học Đồng Tử

**Vấn đề**: Mã nguồn gốc EMS chuyển Excel → text file **mất thuộc tính `FIX_PUPIL`** (chỉ dấu tải nhận thức).

**Giải pháp**: Load trực tiếp từ `.xlsx` + bộ lọc nghiêm ngặt:

```
Raw Excel (.xlsx)
    ↓
┌─────────────────────────────────────────────────────┐
│  Bộ lọc không gian:                                 │
│    Loại bỏ nếu: FIX_X ≥ 1024 ∨ FIX_Y ≥ 768        │
│                  ∨ FIX_X < 0  ∨ FIX_Y < 0          │
├─────────────────────────────────────────────────────┤
│  Bộ lọc thời gian:                                  │
│    Loại bỏ nếu: FIX_DURATION < 50 ms               │
└─────────────────────────────────────────────────────┘
    ↓
clean_fixations.parquet (nén cột, tối ưu RAM Colab)
```

### Phân Tầng 2: Feature Engineering Ngữ Cảnh Đa Tầng

Vì EMS dùng paradigm **free-viewing**, loại bỏ hoàn toàn SPEM/Antisaccade features.

**Tầng A — Stimulus-Level Features** (per image per subject):

| Nhóm | Features |
|---|---|
| Fixation Statistics | Count, Duration (mean/median/std) |
| Saccade Dynamics | Amplitude $A=\sqrt{(\Delta x)^2+(\Delta y)^2}$, Velocity $V=A/\text{dur}_{i+1}$, Turning angle |
| Scanpath Geometry | Total length, Convex hull area, Center bias, Spatial entropy $H=-\sum p(j)\log_2 p(j)$ |
| Pupil Dynamics | Mean, Std, CV of FIX_PUPIL |

**Tầng B — Contextual Delta Features** (subject-level):

| Delta | Ý nghĩa lâm sàng |
|---|---|
| $\Delta_{\text{Social-Natural}}$ | Đứt gãy hành vi khi đối diện bối cảnh xã hội |
| $\Delta_{\text{Manipulated-Natural}}$ | Phản ứng với ảnh biến đổi nhận thức |
| $\Delta_{\text{Social-Synthetic}}$ | Contrast social vs abstract stimuli |
| $\Delta_{\text{Manipulated-Synthetic}}$ | Contrast manipulation awareness |

### Phân Tầng 3: Tabular Baseline + Probability Aggregation

```
features_stimulus_level.csv (160 × 100 = 16,000 rows)
    ↓
┌───────────────────────────────────┐
│  GroupKFold(n_splits=4)           │  ← Chống rò rỉ: split theo Subject_ID
│  → Tabular Models Training       │
│    • XGBoost • LightGBM          │
│    • CatBoost • TabPFN           │
│    • AutoGluon Tabular           │
└─────────────┬─────────────────────┘
              ↓
┌───────────────────────────────────────────────────────┐
│  P(SZ|Subject) = α·Mean(P_Social) + β·Mean(P_Manip)  │
│                + γ·Mean(P_Natural) + δ·Mean(P_Synth)  │
│  Ràng buộc: α + β + γ + δ = 1                        │
│  Tối ưu: Optuna → maximize AUC                       │
└───────────────────────────────────────────────────────┘
```

### Phân Tầng 4: Spatiotemporal GNN + CEFAM

```
┌──────────────────────┐        ┌────────────────────────┐
│  GNN STREAM          │        │  HANDCRAFTED STREAM    │
│                      │        │                        │
│  Scanpath → Graph    │        │  Tier 2 flat features  │
│  Nodes: [x,y,dur,    │        │       ↓                │
│   pupil,pup_diff,    │        │  FC → z_expert [128]   │
│   RINet_64] = 69-dim │        │                        │
│       ↓              │        │                        │
│  GAT ×2 + BN + ELU   │        │                        │
│       ↓              │        │                        │
│  Global Attn Pool    │        │                        │
│       ↓              │        │                        │
│  z_graph [128]       │        │                        │
└──────────┬───────────┘        └──────────┬─────────────┘
           │                               │
           ▼                               ▼
┌──────────────────────────────────────────────────┐
│  CEFAM: Bidirectional Cross-Attention            │
│                                                  │
│  z_f1 = MHA(Q=z_graph,  K=z_expert, V=z_expert) │
│  z_f2 = MHA(Q=z_expert, K=z_graph,  V=z_graph)  │
│                                                  │
│  z_final = Linear(Concat(z_f1, z_f2))           │
└──────────────────┬───────────────────────────────┘
                   ↓
         FC + Softmax → P(SZ), P(HC)

Loss = Focal Loss + λ · Entropy Sparsity Regularization
```

---

## 📊 Bộ Dữ Liệu EMS

| Thuộc tính | Chi tiết |
|---|---|
| **Tên** | EMS (Eye Movement for Schizophrenia) |
| **Nguồn** | [GitHub - YingjieSong1/EMS](https://github.com/YingjieSong1/EMS) |
| **Bài báo** | Song et al., IEEE TNNLS, 2025 |
| **Quy mô** | 104 SZ + 104 HC = 208 subjects (160 train/valid) |
| **Paradigm** | Free-viewing |
| **Stimuli** | 100 images → 4 categories: **Social, Natural, Synthetic, Manipulated** |
| **Resolution** | 1024 × 768 pixels |
| **Data format** | Raw `.xlsx` (bảo toàn FIX_PUPIL) |
| **Visual features** | `feature_dict_RINet.npy` (1056-dim per fixation) |

### Các trường dữ liệu chính

| Trường | Mô tả |
|---|---|
| `Subject_ID` | Mã định danh đối tượng |
| `Stimulus_ID` | Mã ảnh kích thích (1-100) |
| `FIX_X` | Tọa độ X fixation (pixels) |
| `FIX_Y` | Tọa độ Y fixation (pixels) |
| `FIX_DURATION` | Thời gian fixation (ms) |
| `FIX_PUPIL` | Đường kính đồng tử |
| `FIX_START` | Timestamp bắt đầu fixation |
| `Label` | 0=HC, 1=SZ |

---

## ⚙️ Cài Đặt

### Yêu cầu hệ thống

| Thành phần | Yêu cầu tối thiểu | Khuyến nghị |
|---|---|---|
| **Python** | 3.9+ | 3.10+ |
| **GPU** | Google Colab GPU | Colab Pro (T4/V100) |
| **RAM** | 12GB | 16GB+ |
| **CUDA** | 11.7+ | 12.1+ |

### Cài đặt

```bash
# 1. Tạo virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows

# 2. Cài PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Cài PyTorch Geometric
pip install torch-geometric

# 4. Cài các dependencies
pip install -r requirements.txt

# 5. Kiểm tra
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 📁 Cấu Trúc Thư Mục

```
Eye Movement-Based Schizophrenia Recognition/
│
├── 📄 README.md                          ← Bạn đang ở đây
├── 📄 requirements.txt
├── 📄 .gitignore
│
├── 📂 data/
│   ├── 📂 raw/                           # Raw Excel .xlsx
│   ├── 📂 processed/
│   │   ├── 📄 clean_fixations.parquet    # Tier 1 output
│   │   ├── 📄 features_stimulus_level.csv # Tier 2A
│   │   ├── 📄 features_subject_level.csv  # Tier 2B + deltas
│   │   └── 📂 graphs/                    # Tier 4 PyG objects
│   ├── 📂 stimuli/                       # 100 stimulus images
│   │   ├── 📂 social/ natural/ synthetic/ manipulated/
│   ├── 📂 external/
│   │   └── 📄 feature_dict_RINet.npy     # 1056-dim visual features
│   └── 📂 metadata/
│       └── 📄 stimulus_categories.csv    # Image → category mapping
│
├── 📂 configs/                           # YAML configs per tier
│
├── 📂 src/
│   ├── 📂 tier1_preprocessing/           # Excel→Filter→Parquet
│   ├── 📂 tier2_features/                # Stimulus + Delta features
│   ├── 📂 tier3_tabular/                 # GroupKFold + ML + Optuna
│   ├── 📂 tier4_advanced/                # GNN + CEFAM
│   ├── 📂 training/
│   ├── 📂 evaluation/
│   └── 📂 utils/
│
├── 📂 Spatiotemporal GNN/                # GNN stream workspace
├── 📂 Bidirectional Cross-Attention Hybrid Stream/  # CEFAM workspace
├── 📂 experiments/                       # Baselines + Ablation
├── 📂 notebooks/                         # Shared Jupyter notebooks
├── 📂 docs/
└── 📂 tests/
```

---

## 🚀 Hướng Dẫn Sử Dụng

### Bước 1: Tạo Category Mapping
```bash
# Quét thư mục ảnh và sinh file mapping category kích thích
python src/utils/generate_category_map.py
```

### Tier 1: Tiền xử lý

```bash
# Chạy tiền xử lý dữ liệu (Excel -> Parquet) dựa trên file config
python src/tier1_preprocessing/preprocess.py --config configs/cefam_config.yaml
```

### Tier 2: Feature Engineering

```bash
# Trích xuất đặc trưng trial (stimulus-level)
python src/tier2_features/stimulus_features.py \
    --config configs/cefam_config.yaml

# Trích xuất đặc trưng subject và tính delta ngữ cảnh
python src/tier2_features/subject_aggregator.py \
    --config configs/cefam_config.yaml
```

### Tier 3: Tabular Baseline

```bash
# Huấn luyện baseline XGBoost/LightGBM/CatBoost
python src/tier3_tabular/tabular_models.py \
    --config configs/cefam_config.yaml \
    --model xgboost

# Tối ưu hóa trọng số hội tụ bằng Optuna
python src/tier3_tabular/optuna_weights.py \
    --config configs/cefam_config.yaml \
    --preds results/baselines/xgboost_stimulus_val_preds.csv
```

### Tier 4: GNN + CEFAM

```bash
# Xây dựng đồ thị scanpath PyG
python src/tier4_advanced/graph_builder.py \
    --config configs/cefam_config.yaml

# Huấn luyện mô hình hybrid GNN+CEFAM (tự động chạy GPU nếu có)
python scripts/train_tier4.py \
    --config configs/cefam_config.yaml \
    --seed 42
```

---

## 📈 Kết Quả & Đánh Giá

### Metrics (theo benchmark EMS)

| Metric | Target Tier 3 | Target Tier 4 | MSNet (SOTA) |
|---|:---:|:---:|:---:|
| **Accuracy** | ~80-84% | **>85%** | 81.25% |
| **AUC-ROC** | ~86-90% | **>90%** | 88.54% |
| **F1-Score** | ~0.80 | **>0.84** | ~0.80 |

### Evaluation Protocol

- **CV**: GroupKFold(n_splits=4), group=Subject_ID
- **Anti-leakage**: Toàn bộ 100 trials/subject trong cùng 1 fold
- **Aggregation**: Stimulus → Subject via Optuna-weighted probabilities
- **Statistical test**: Paired t-test / Wilcoxon (p < 0.05)

---

## 📅 Kịch Bản Thực Hiện (12-14 Tuần)

| Phase | Thời gian | Deliverables |
|---|:---:|---|
| **Tier 1**: Preprocessing | Tuần 1-2 | `clean_fixations.parquet` |
| **Tier 2**: Feature Engineering | Tuần 2-4 | `features_stimulus_level.csv`, `features_subject_level.csv` |
| **Tier 3**: Tabular Baseline | Tuần 4-6 | Leaderboard, Optuna weights |
| **Tier 4**: GNN + CEFAM | Tuần 5-10 | Trained models, ablation results |
| **Evaluation** | Tuần 10-12 | Comparative analysis, explainability |
| **Documentation** | Tuần 12-14 | Report, figures, clean code |

---

## 📚 Tham Khảo

1. Song, Y. et al. (2025). "EMS: A Large-Scale Eye Movement Dataset, Benchmark, and New Model for Schizophrenia Recognition." *IEEE TNNLS*, 36(5), 9451–9462.
2. Veličković, P. et al. (2018). "Graph Attention Networks." *ICLR 2018*.
3. Lin, T.Y. et al. (2017). "Focal Loss for Dense Object Detection." *ICCV 2017*.
4. Birawo, B. & Kasprowski, P. (2024). "Graph Neural Networks for Scanpath Classification." *gazeRE, ETRA 2024*.
5. Hollmann, N. et al. (2025). "TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second." *Nature*.
6. CEFAM-DACM (2025/2026). Cross-attention fusion for Alzheimer's Disease recognition.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
