# 🔬 Bidirectional Cross-Attention Fusion (CEFAM) — Tier 4 Fusion Module

> **Module hợp nhất chú ý chéo hai chiều (CEFAM) cho phép hai luồng thông tin GNN và Handcrafted tương tác tương hỗ, kết hợp Focal Loss + Entropy Sparsity Regularization để tối đa hóa khả năng chẩn đoán và giải thích lâm sàng.**

---

## 📋 Mục Lục

- [Tổng Quan](#-tổng-quan)
- [Kiến Trúc CEFAM Chi Tiết](#-kiến-trúc-cefam-chi-tiết)
- [GNN Stream (z_graph)](#-gnn-stream)
- [Handcrafted Stream (z_expert)](#-handcrafted-stream)
- [Bidirectional Cross-Attention](#-bidirectional-cross-attention)
- [Loss Function](#-loss-function-focal-loss--entropy-sparsity-regularization)
- [Bộ Đặc Trưng Tier 2](#-bộ-đặc-trưng-tier-2-cho-handcrafted-stream)
- [Contextual Delta Features](#-contextual-delta-features)
- [Probability Aggregation (Tier 3)](#-probability-aggregation-tier-3)
- [Ablation Studies](#-ablation-studies)
- [Explainability](#-explainability--clinical-interpretation)
- [Quick Start](#-quick-start)

---

## 🎯 Tổng Quan

### Vai trò trong Framework

CEFAM (Cross-attention Enhanced Fusion Attention Module) là **mô-đun hợp nhất** trung tâm của Tier 4, giải quyết sự **phân mảnh** giữa nhóm Y sinh (handcrafted features) và nhóm AI (deep learning):

```
Vấn đề cũ:  GNN features ──── Concatenation ──── Expert features
             (rich nhưng opaque)  (thô sơ, không    (proven nhưng
                                   tương tác)        shallow)

Giải pháp:   GNN features ◄──── CEFAM ────► Expert features
             z_graph         Bidirectional     z_expert
                            Cross-Attention
                            (tương hỗ động)
```

### Ưu thế so với fusion truyền thống

| Fusion method | Tương tác | Interpretable | Expected gain |
|---|:---:|:---:|:---:|
| Concatenation | ❌ Static | ❌ | Baseline |
| Addition/Average | ❌ Static | ❌ | ~0% |
| Unidirectional Attention | ⚠️ One-way | ⚠️ | +3-5% |
| **CEFAM (Bidirectional)** ✅ | ✅ **Mutual** | ✅ | **+8-10%** |

---

## 🏗 Kiến Trúc CEFAM Chi Tiết

### Full Hybrid Model

```
┌──────────────────────────┐        ┌────────────────────────────┐
│     GNN STREAM           │        │   HANDCRAFTED STREAM       │
│                          │        │                            │
│  Graph G = (V, E)        │        │  Tier 2 flat features      │
│  Nodes: 69-dim           │        │  (stimulus-level from      │
│  [x,y,dur,pup,pup_diff,  │        │   features_stimulus_level  │
│   RINet_64]              │        │   or subject-level +       │
│       ↓                  │        │   delta features)          │
│  GAT Layer 1 (4-head)    │        │       ↓                    │
│  + EdgeAttr + BN + ELU   │        │  FC(D_flat → 256)         │
│       ↓                  │        │  + BatchNorm + GELU        │
│  GAT Layer 2 (4-head)    │        │       ↓                    │
│  + Residual + BN + ELU   │        │  FC(256 → 128)            │
│       ↓                  │        │  + BatchNorm + GELU        │
│  Global Attention Pool   │        │       ↓                    │
│       ↓                  │        │  z_expert [128]            │
│  z_graph [128]           │        │                            │
└──────────┬───────────────┘        └──────────┬─────────────────┘
           │                                    │
           ▼                                    ▼
┌──────────────────────────────────────────────────────────────┐
│           CEFAM: Bidirectional Cross-Attention                │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Direction 1: Expert guides Graph                      │  │
│  │  z_fused_1 = MHA(Q=z_graph, K=z_expert, V=z_expert)   │  │
│  │  → "Which graph patterns align with clinical markers?" │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Direction 2: Graph guides Expert                      │  │
│  │  z_fused_2 = MHA(Q=z_expert, K=z_graph, V=z_graph)    │  │
│  │  → "What hidden patterns enhance biomarker meaning?"   │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  z_final = Linear(Concat(z_fused_1, z_fused_2))             │
└──────────────────────┬───────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────────┐
│  Classification Head                                         │
│  FC(256 → 128) → GELU → Dropout → FC(128 → 2) → Softmax    │
│  Output: P(SZ), P(HC)                                       │
└──────────────────────────────────────────────────────────────┘

Loss = Focal Loss(α=0.25, γ=2.0) + λ · Entropy Sparsity Reg.
```

---

## 🔷 GNN Stream

(Chi tiết đầy đủ → xem [Spatiotemporal GNN README](../Spatiotemporal%20GNN/README.md))

| Component | Config | Output |
|---|---|---|
| Node dim | 5 (low-level) + 64 (RINet projected) = **69** | |
| GAT Layer 1 | `GATConv(69→64, heads=4, edge_dim=2)` | `[N × 256]` |
| GAT Layer 2 | `GATConv(256→64, heads=4)` + Residual | `[N × 256]` |
| Global Attention Pooling | `gate_nn: 256→1` | `[batch × 256]` |
| Projection | `Linear(256→128)` | **`z_graph [128]`** |

---

## 🟢 Handcrafted Stream

### Input
Ma trận đặc trưng phẳng tối ưu từ Tier 2 Feature Engineering.

### Architecture

```python
class HandcraftedStream(nn.Module):
    def __init__(self, d_flat, d_out=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_flat, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, d_out),
            nn.BatchNorm1d(d_out),
            nn.GELU(),
        )
    
    def forward(self, features):
        """
        features: [batch × D_flat]  (Tier 2 features)
        Returns:  z_expert [batch × 128]
        """
        return self.encoder(features)
```

---

## 🔀 Bidirectional Cross-Attention

### Cơ chế toán học

**Direction 1** — Expert hướng dẫn Graph:
$$\mathbf{z}_{\text{fused\_1}} = \text{MultiheadAttention}(Q=\mathbf{z}_{\text{graph}}, K=\mathbf{z}_{\text{expert}}, V=\mathbf{z}_{\text{expert}})$$

**Direction 2** — Graph hướng dẫn Expert:
$$\mathbf{z}_{\text{fused\_2}} = \text{MultiheadAttention}(Q=\mathbf{z}_{\text{expert}}, K=\mathbf{z}_{\text{graph}}, V=\mathbf{z}_{\text{graph}})$$

**Fusion**:
$$\mathbf{z}_{\text{final}} = \text{Linear}(\text{Concat}(\mathbf{z}_{\text{fused\_1}}, \mathbf{z}_{\text{fused\_2}}))$$

### Ý nghĩa từng hướng

| Hướng | Q | K, V | Câu hỏi mô hình đặt ra |
|---|---|---|---|
| **Dir 1** | z_graph | z_expert | "Trong đồ thị scanpath, fixation nào **khớp** với bất thường lâm sàng?" |
| **Dir 2** | z_expert | z_graph | "Trong biomarker vector, feature nào được **khuếch đại** bởi graph patterns?" |

### Implementation

```python
class CEFAMFusion(nn.Module):
    """
    Cross-attention Enhanced Fusion Attention Module
    Bidirectional cross-attention between GNN and Expert streams
    """
    def __init__(self, d_model=128, nhead=4, dropout=0.1):
        super().__init__()
        
        # Direction 1: Expert guides Graph
        self.cross_attn_1 = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            dropout=dropout, batch_first=True
        )
        
        # Direction 2: Graph guides Expert
        self.cross_attn_2 = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            dropout=dropout, batch_first=True
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, z_graph, z_expert):
        """
        z_graph:  [batch × 128]
        z_expert: [batch × 128]
        Returns:  z_final [batch × 256], attn_weights
        """
        # Reshape for MHA: [batch × 1 × 128]
        z_g = z_graph.unsqueeze(1)
        z_e = z_expert.unsqueeze(1)
        
        # Direction 1: Q=graph, KV=expert
        z_fused_1, attn_1 = self.cross_attn_1(
            query=z_g, key=z_e, value=z_e
        )
        
        # Direction 2: Q=expert, KV=graph
        z_fused_2, attn_2 = self.cross_attn_2(
            query=z_e, key=z_g, value=z_g
        )
        
        # Concat + Fusion
        z_concat = torch.cat([
            z_fused_1.squeeze(1),   # [batch × 128]
            z_fused_2.squeeze(1)    # [batch × 128]
        ], dim=-1)                  # [batch × 256]
        
        z_final = self.fusion(z_concat)  # [batch × 256]
        
        return z_final, attn_1, attn_2
```

---

## 🔥 Loss Function: Focal Loss + Entropy Sparsity Regularization

### Focal Loss
Tập trung vào hard examples (mẫu khó phân loại), đặc biệt quan trọng cho diagnostic tasks:

$$\mathcal{L}_{\text{focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

| Parameter | Value | Ý nghĩa |
|---|:---:|---|
| $\alpha$ | 0.25 | Balancing factor (SZ vs HC) |
| $\gamma$ | 2.0 | Focusing parameter (giảm loss cho easy examples) |

### Entropy Sparsity Regularization
Thúc đẩy attention weights **tập trung** (sparse) → nâng cao khả năng giải thích lâm sàng:

$$\mathcal{L}_{\text{entropy}} = -\lambda \sum_{i} a_i \log(a_i + \epsilon)$$

Trong đó $a_i$ là cross-attention weights từ CEFAM.

**Ý nghĩa**: Khi attention sparse → mô hình chỉ tập trung vào vài features/patterns quyết định → bác sĩ dễ hiểu hơn (theo chuẩn TIFU - Trust In Faithful Understandability).

### Total Loss

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{focal}} + \lambda_{\text{sparse}} \cdot \mathcal{L}_{\text{entropy}}$$

| Parameter | Default | Range (tuning) |
|---|:---:|---|
| $\lambda_{\text{sparse}}$ | 0.01 | [0.001, 0.1] |
| $\epsilon$ | 1e-8 | Fixed |

### Implementation

```python
class FocalLossWithEntropyReg(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, 
                 entropy_lambda=0.01, eps=1e-8):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.entropy_lambda = entropy_lambda
        self.eps = eps
    
    def forward(self, logits, targets, attention_weights=None):
        # === Focal Loss ===
        probs = F.softmax(logits, dim=-1)
        targets_oh = F.one_hot(targets, 2).float()
        pt = (probs * targets_oh).sum(dim=-1)
        alpha_t = self.alpha * targets.float() + \
                  (1 - self.alpha) * (1 - targets.float())
        focal = -alpha_t * (1 - pt) ** self.gamma * \
                torch.log(pt + self.eps)
        loss = focal.mean()
        
        # === Entropy Sparsity Regularization ===
        if attention_weights is not None:
            # Encourage sparse (low entropy) attention
            ent = -(attention_weights * 
                    torch.log(attention_weights + self.eps)).sum(-1)
            loss += self.entropy_lambda * ent.mean()
        
        return loss
```

---

## 📋 Bộ Đặc Trưng Tier 2 cho Handcrafted Stream

### Stimulus-Level Features (~20-25 per trial)

| # | Nhóm | Features | Tương thích free-viewing? |
|:---:|---|---|:---:|
| 1-4 | **Fixation Stats** | count, dur_mean, dur_median, dur_std | ✅ |
| 5-7 | **Saccade Dynamics** | amplitude_mean, velocity_mean, turning_angle_mean | ✅ |
| 8-11 | **Scanpath Geometry** | total_length, convex_hull_area, center_bias, spatial_entropy | ✅ |
| 12-14 | **Pupil Dynamics** | pupil_mean, pupil_std, pupil_cv | ✅ |
| ~~15~~ | ~~SPEM gain~~ | ~~Smooth pursuit~~ | ❌ **Loại bỏ** |
| ~~16~~ | ~~Antisaccade error~~ | ~~Error rate~~ | ❌ **Loại bỏ** |

---

## 🔄 Contextual Delta Features

### 4 Categories ảnh kích thích

| Category | Ý nghĩa | SZ relevance |
|---|---|---|
| **Social** | Ảnh bối cảnh xã hội (khuôn mặt, tương tác người) | ⭐⭐⭐ SZ bộc lộ đứt gãy rõ nhất |
| **Natural** | Ảnh cảnh tự nhiên | Baseline reference |
| **Synthetic** | Ảnh tổng hợp/trừu tượng | Low-level processing test |
| **Manipulated** | Ảnh bị biến đổi (suy giảm nhận thức) | ⭐⭐ Phản ứng bất thường |

### Delta Computation

$$\Delta_{\text{Social-Natural}}^{(\text{feat})} = \overline{\text{feat}}_{\text{Social}} - \overline{\text{feat}}_{\text{Natural}}$$

**Ví dụ concrete**: Nếu SZ patient có `Δ_Social-Natural_fixation_duration = +200ms`, nghĩa là họ cần **thêm 200ms** để xử lý ảnh xã hội so với ảnh tự nhiên — dấu hiệu suy giảm nhận thức xã hội.

---

## 📊 Probability Aggregation (Tier 3)

### Gom cụm xác suất có trọng số ngữ cảnh

Mô hình dự đoán xác suất ở **stimulus-level**, sau đó gom về **subject-level**:

$$P(\text{SZ} | \text{Subject}) = \alpha \cdot \overline{P}_{\text{Social}} + \beta \cdot \overline{P}_{\text{Manipulated}} + \gamma \cdot \overline{P}_{\text{Natural}} + \delta \cdot \overline{P}_{\text{Synthetic}}$$

**Ràng buộc**: $\alpha + \beta + \gamma + \delta = 1$

**Tối ưu**: Optuna search → maximize AUC

**Kỳ vọng**: $\alpha > \gamma$ (Social quan trọng hơn Natural) và $\beta > \delta$ (Manipulated quan trọng hơn Synthetic)

---

## 🔬 Ablation Studies

| # | Experiment | Modification | Purpose |
|:---:|---|---|---|
| F1 | **Full CEFAM** | Complete model | Reference |
| F2 | GNN only | Remove handcrafted stream | GNN standalone value |
| F3 | Handcrafted only | Remove GNN stream | Expert features value |
| F4 | Concat fusion | Replace CEFAM with concat | Fusion comparison |
| F5 | Add fusion | Replace CEFAM with addition | Fusion comparison |
| F6 | Unidir (Expert→Graph) | Only Direction 1 | Bidirectional value |
| F7 | Unidir (Graph→Expert) | Only Direction 2 | Bidirectional value |
| F8 | CE loss vs Focal Loss | Replace loss function | Focal loss value |
| F9 | With Entropy Reg vs Without | λ=0 | Sparsity value |
| F10 | λ ∈ {0.001, 0.01, 0.05, 0.1} | Vary regularization | Optimal λ |
| F11 | Without delta features | Remove contextual deltas | Delta features value |
| F12 | Without pupil features | Remove pupil group | Pupil value |

---

## 🔍 Explainability & Clinical Interpretation

### 1. CEFAM Attention Weight Analysis

Cross-attention weights từ Direction 1 và 2 cho biết:
- **Dir 1 weights**: Biomarker nào **hướng dẫn** graph representation mạnh nhất
- **Dir 2 weights**: Graph pattern nào **khuếch đại** biomarker signal

### 2. Entropy Sparsity Visualization

```python
# Sparse attention → few dominant features → clinically interpretable
import matplotlib.pyplot as plt

def plot_attention_sparsity(attn_weights, feature_names):
    """Visualize how sparse the cross-attention is"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Attention distribution
    ax1.bar(feature_names, attn_weights.mean(0), color='coral')
    ax1.set_title("Mean Cross-Attention Weights")
    ax1.set_ylabel("Attention Weight")
    
    # Entropy over training
    ax2.plot(training_entropies, label='Attention Entropy')
    ax2.axhline(y=target_entropy, color='r', linestyle='--', 
                label='Target (sparse)')
    ax2.set_title("Attention Entropy over Training")
    ax2.legend()
    
    plt.tight_layout()
```

### 3. Feature Group Importance (via Ablation)

```
Full model                      ████████████████████  88%+
Without scanpath geometry       ████████████████      82%   (-6%)
Without pupil dynamics          ██████████████████    84%   (-4%)
Without saccade dynamics        █████████████████     83%   (-5%)
Without delta features          ██████████████████    84%   (-4%)
Without RINet visual            █████████████████     83%   (-5%)
```

---

## 🚀 Quick Start

```bash
# 1. Train full GNN+CEFAM
python scripts/train_tier4.py \
    --config configs/cefam_config.yaml \
    --gpus 1 --seed 42

# 2. Evaluate
python scripts/evaluate_tier4.py \
    --config configs/cefam_config.yaml \
    --folds 4

# 3. Ablation studies
python experiments/ablation/ablation_fusion.py \
    --experiments all

# 4. Explainability
python src/evaluation/explainability.py \
    --model cefam \
    --checkpoint results/checkpoints/cefam_best.pt \
    --method attention_viz,shap
```

---

## 📂 File Structure

```
Bidirectional Cross-Attention Hybrid Stream/
├── README.md              ← Bạn đang ở đây
├── notebooks/
│   ├── 01_feature_engineering.ipynb
│   ├── 02_stream_analysis.ipynb
│   ├── 03_training_cefam.ipynb
│   └── 04_explainability.ipynb
├── scripts/
│   ├── train_bica.py
│   ├── evaluate_bica.py
│   └── hyperopt_bica.py
└── results/
    ├── checkpoints/
    ├── logs/
    └── figures/
```
