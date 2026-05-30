# 🔬 Spatiotemporal Graph Neural Network — Tier 4 GNN Stream

> **Biểu diễn scanpath dưới dạng đồ thị phi Euclid không-thời gian có hướng và trọng số, xử lý bằng 2-layer GAT kết hợp RINet visual features (1056→64) và Padding Masking, nhằm nhận dạng tâm thần phân liệt.**

---

## 📋 Mục Lục

- [Tổng Quan](#-tổng-quan)
- [Kiến Trúc Chi Tiết](#-kiến-trúc-chi-tiết)
- [Graph Construction](#-graph-construction)
- [Node & Edge Features](#-node--edge-features)
- [Padding Masking Strategy](#-padding-masking-strategy)
- [RINet Visual Feature Projection](#-rinet-visual-feature-projection)
- [GAT Stream & Global Attention Pooling](#-gat-stream--global-attention-pooling)
- [Ablation Studies](#-ablation-studies)
- [Quick Start](#-quick-start)

---

## 🎯 Tổng Quan

### Vai trò trong Framework

GNN Stream là **nhánh đồ thị** trong kiến trúc Tier 4 (Advanced Track), có nhiệm vụ:
1. Chuyển đổi mỗi chuỗi scanpath thành **đồ thị có hướng và trọng số** $G = (V, E)$
2. Học biểu diễn ẩn cấu trúc liên kết không-thời gian bất thường qua GAT
3. Nén thành vector `z_graph` để đưa vào module CEFAM fusion

### Cơ sở khoa học
- GNN trên scanpath tăng **+7.6% ~ +14.3% balanced accuracy** so với CNN/RNN (gazeRE 2024)
- SZ patients thể hiện **restricted scanning**, convex hull area nhỏ, entropy đường quét bất thường → topo graph phát hiện tốt
- GAT attention weights → interpretable (fixation nào quan trọng nhất)

---

## 🏗 Kiến Trúc Chi Tiết

```
Input: Scanpath → Graph G = (V, E)
│
│  Nodes: [x_norm, y_norm, dur_norm, pupil, pupil_diff, RINet_64]
│  dim = 5 + 64 = 69 per node
│  Edges: Sequential (i→i+1) + Spatial k-NN (k=3)
│  Edge attr: [amplitude, angle]
│
▼
┌──────────────────────────────────────────────────────────┐
│  Padding Masking: Pad/Truncate to N ∈ {14, 20, 24, 32}  │
└───────────────────────┬──────────────────────────────────┘
                        ▼
┌──────────────────────────────────────────────────────────┐
│  GAT Layer 1: GATConv(69→64, heads=4) + EdgeAttr        │
│  BatchNorm(256) → ELU → Dropout(0.3)                    │
│  Output: [N × 256]                                      │
└───────────────────────┬──────────────────────────────────┘
                        ▼
┌──────────────────────────────────────────────────────────┐
│  GAT Layer 2: GATConv(256→64, heads=4) + Residual        │
│  BatchNorm(256) → ELU → Dropout(0.3)                    │
│  Output: [N × 256]                                      │
└───────────────────────┬──────────────────────────────────┘
                        ▼
┌──────────────────────────────────────────────────────────┐
│  Global Attention Pooling                                │
│  gate_nn = Linear(256→1)                                 │
│  → z_graph_raw [1 × 256]                                 │
│  → Linear(256→128) → z_graph [1 × 128]                  │
└──────────────────────────────────────────────────────────┘
                        │
                        ▼
              z_graph [128-dim]
              → Feed to CEFAM Fusion
```

---

## 📐 Graph Construction

### Quy trình

```
Scanpath: fix₁(x₁,y₁,t₁,dur₁,pup₁) → fix₂ → ... → fixₙ
    ↓
Graph G = (V, E) có hướng và trọng số

V = {fix₁, fix₂, ..., fixₙ}    (nodes = fixation points)
E = E_temporal ∪ E_spatial       (edges = saccade + proximity)
```

### Edge Types

| Loại cạnh | Hướng | Mô tả | Trọng số |
|---|:---:|---|---|
| **Temporal** (Sequential) | Directed: $i \to i+1$ | Chuỗi thời gian quét mắt | Amplitude $A$, Angle $\theta$ |
| **Spatial** (k-NN, k=3) | Undirected | Cụm chú ý cục bộ, re-fixation | Khoảng cách Euclid |

```python
# Temporal edges
for i in range(n_fixations - 1):
    edges.append((i, i+1))  # directed

# Spatial k-NN edges (k=3)
from scipy.spatial import KDTree
tree = KDTree(coords)
for i, coord in enumerate(coords):
    _, neighbors = tree.query(coord, k=4)  # +1 for self
    for j in neighbors[1:]:
        edges.append((i, j))
        edges.append((j, i))  # undirected
```

---

## 🧬 Node & Edge Features

### Node Feature Vector (69-dimensional)

| # | Feature | Dim | Nguồn | Ý nghĩa |
|:---:|---|:---:|---|---|
| 1 | `x_norm` | 1 | FIX_X / 1024 | Tọa độ X chuẩn hóa |
| 2 | `y_norm` | 1 | FIX_Y / 768 | Tọa độ Y chuẩn hóa |
| 3 | `duration_norm` | 1 | FIX_DURATION (normalized) | Thời gian xử lý nhận thức |
| 4 | `pupil_size` | 1 | FIX_PUPIL (normalized) | Kích thước đồng tử |
| 5 | `pupil_diff` | 1 | $\text{pupil}_i - \text{pupil}_{i-1}$ | **Biến thiên đồng tử** (Δ đồng tử) |
| 6-69 | `rinet_proj` | **64** | `feature_dict_RINet.npy` → MLP(1056→64) | Đặc trưng trực quan cục bộ nén |
| | **Total** | **69** | | |

**`pupil_diff`** — feature mới quan trọng:
- Phản ánh **tốc độ thay đổi** kích thước đồng tử giữa fixation liên tiếp
- Chỉ dấu phản ứng nhận thức real-time (cognitive arousal transitions)
- SZ patients có pupil reactivity bất thường

### Edge Feature Vector (2-dimensional)

| # | Feature | Công thức |
|:---:|---|---|
| 1 | Saccade amplitude | $A = \sqrt{(x_{i+1} - x_i)^2 + (y_{i+1} - y_i)^2}$ |
| 2 | Saccade angle | $\theta = \arctan2(y_{i+1} - y_i, x_{i+1} - x_i)$ |

---

## 🎭 Padding Masking Strategy

### Vấn đề
Mỗi trial có số fixation khác nhau. Cần chuẩn hóa kích thước đồ thị cho batching.

### Giải pháp: Padding + Masking

```python
class PaddingMasker:
    def __init__(self, max_len: int = 24):
        """
        max_len: Ngưỡng giới hạn chuỗi
        Ablation thử N ∈ {14, 20, 24, 32}
        """
        self.max_len = max_len
    
    def __call__(self, fixations):
        n = len(fixations)
        if n >= self.max_len:
            # Truncate: giữ max_len fixation đầu tiên
            return fixations[:self.max_len], mask=all_true
        else:
            # Pad with zeros + mask
            padded = zero_pad(fixations, self.max_len)
            mask = [True]*n + [False]*(self.max_len - n)
            return padded, mask
```

### Ablation: Tối ưu N

| N | Ưu điểm | Nhược điểm |
|:---:|---|---|
| 14 | Nhanh, ít padding | Mất fixation cuối → mất hành vi vĩ mô |
| 20 | Cân bằng | |
| **24** ✅ | **Giữ nguyên ~95% hành vi quét mắt** | Moderate computation |
| 32 | Đầy đủ nhất | Nhiều padding → noise |

---

## 🔬 RINet Visual Feature Projection

### Mục đích
File `feature_dict_RINet.npy` chứa **1056-dim** visual features trích xuất từ mô hình RINet cho mỗi fixation point → quá lớn cho GPU memory.

### Giải pháp: MLP Projection 1056 → 64

```python
class RINetProjector(nn.Module):
    """
    Giảm chiều 1056-dim RINet → 64-dim
    Lý do: 
    - Giảm tải bộ nhớ đồ họa (~16× compression)
    - Tránh overfitting trên dataset nhỏ (208 subjects)
    - 64-dim đủ để encode visual semantics
    """
    def __init__(self, input_dim=1056, output_dim=64):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(1056, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.projector(x)  # [num_nodes × 64]
```

### Tích hợp với Node Features

```python
# Final node feature = concat(low_level, rinet_projected)
low_level = [x_norm, y_norm, dur_norm, pupil, pupil_diff]  # 5-dim
rinet_proj = rinet_projector(rinet_raw)                      # 64-dim
node_features = torch.cat([low_level, rinet_proj], dim=-1)   # 69-dim
```

---

## 🔷 GAT Stream & Global Attention Pooling

### 2-Layer GAT

```python
class GATStream(nn.Module):
    def __init__(self, in_dim=69, hidden_dim=64, heads=4, 
                 out_dim=128, dropout=0.3):
        super().__init__()
        # Layer 1
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, 
                            edge_dim=2, concat=True)
        self.bn1 = BatchNorm(hidden_dim * heads)
        
        # Layer 2 + Residual
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, 
                            heads=heads, edge_dim=2, concat=True)
        self.bn2 = BatchNorm(hidden_dim * heads)
        self.residual = nn.Linear(in_dim, hidden_dim * heads)
        
        # Global Attention Pooling
        gate_nn = nn.Sequential(
            nn.Linear(hidden_dim * heads, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.pool = GlobalAttention(gate_nn)
        
        # Final projection
        self.proj = nn.Linear(hidden_dim * heads, out_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, edge_attr, batch, mask=None):
        # Layer 1
        h = self.gat1(x, edge_index, edge_attr)
        h = self.bn1(h)
        h = F.elu(h)
        h = self.dropout(h)
        
        # Layer 2 + Residual
        h2 = self.gat2(h, edge_index, edge_attr)
        h2 = self.bn2(h2)
        h2 = h2 + self.residual(x)  # skip connection
        h2 = F.elu(h2)
        h2 = self.dropout(h2)
        
        # Graph-level readout
        z_graph = self.pool(h2, batch)    # [batch_size × 256]
        z_graph = self.proj(z_graph)      # [batch_size × 128]
        
        return z_graph
```

**Tại sao 2 layer (không phải 3)?**
- Đồ thị scanpath nhỏ (14-32 nodes) → 2 hops đã cover toàn bộ
- Tránh over-smoothing trên small graphs
- Đủ capacity cho pattern recognition

---

## 🔬 Ablation Studies

### Planned Experiments

| # | Experiment | Modification | Purpose |
|:---:|---|---|---|
| T4.1 | **Sequence length** | N ∈ {14, 20, **24**, 32} | Optimal truncation |
| T4.2 | GNN only (no CEFAM) | Remove fusion | GNN stream standalone |
| T4.3 | With RINet vs Without | Node: 69-dim vs 5-dim | Visual features value |
| T4.4 | With pupil_diff vs Without | Node: 69 vs 68-dim | Pupil dynamics value |
| T4.5 | Sequential edges only | No spatial k-NN | Spatial topology value |
| T4.6 | Spatial edges only | No temporal | Temporal ordering value |
| T4.7 | k ∈ {2, 3, 5, 7} | Vary k-NN | Graph density |
| T4.8 | 1-layer vs 2-layer vs 3-layer | GAT depth | Model depth |
| T4.9 | GCN backbone | Replace GAT | Attention value |
| T4.10 | Sum pooling | Replace Global Attn Pool | Readout strategy |

---

## 🚀 Quick Start

```bash
# 1. Build graphs from preprocessed data
python src/tier4_advanced/graph_builder.py \
    --input data/processed/clean_fixations.parquet \
    --rinet data/external/feature_dict_RINet.npy \
    --output data/processed/graphs/ \
    --max-len 24 --k-neighbors 3

# 2. Train GNN+CEFAM (sanity check)
python scripts/train_tier4.py \
    --config configs/cefam_config.yaml \
    --overfit-batches 1 --max-epochs 50

# 3. Full training
python scripts/train_tier4.py \
    --config configs/cefam_config.yaml \
    --gpus 1 --seed 42

# 4. Ablation: sequence length
for N in 14 20 24 32; do
    python scripts/train_tier4.py \
        --config configs/cefam_config.yaml \
        --max-seq-len $N --seed 42
done
```

---

## 📂 File Structure

```
Spatiotemporal GNN/
├── README.md              ← Bạn đang ở đây
├── notebooks/
│   ├── 01_eda_scanpath.ipynb
│   ├── 02_graph_construction.ipynb
│   ├── 03_training_gnn.ipynb
│   └── 04_analysis_results.ipynb
├── scripts/
│   ├── train_stgnn.py
│   ├── evaluate_stgnn.py
│   └── hyperopt_stgnn.py
└── results/
    ├── checkpoints/
    ├── logs/
    └── figures/
```
