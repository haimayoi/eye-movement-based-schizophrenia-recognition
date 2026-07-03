import json
from pathlib import Path

path = Path("diagnostic_visualization.ipynb")
if not path.exists():
    print(f"Error: {path} not found.")
    exit(1)

with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Define the new cells to append
new_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### ── Additional Visualizations (UMAP, LDA, Normalization comparison) ──"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. UMAP Projection of ResNet50 Patch Features\n",
            "try:\n",
            "    from umap import UMAP\n",
            "    import matplotlib.pyplot as plt\n",
            "    import numpy as np\n",
            "\n",
            "    print(\"Running UMAP feature projection...\")\n",
            "    if 'feat_dict' in locals() and feat_dict:\n",
            "        feats = np.array(list(feat_dict.values()))\n",
            "        if len(feats.shape) > 2:\n",
            "            feats = feats.reshape(feats.shape[0], -1)\n",
            "        \n",
            "        reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)\n",
            "        embedding = reducer.fit_transform(feats)\n",
            "        \n",
            "        plt.figure(figsize=(8, 6))\n",
            "        plt.scatter(embedding[:, 0], embedding[:, 1], c='purple', alpha=0.6, edgecolors='w', s=30)\n",
            "        plt.title('UMAP Projection of ResNet50 Patch Features', fontsize=12, fontweight='bold')\n",
            "        plt.xlabel('UMAP Dimension 1')\n",
            "        plt.ylabel('UMAP Dimension 2')\n",
            "        plt.grid(True, alpha=0.3)\n",
            "        plt.savefig(PROJECT_ROOT / 'results' / 'tier4_resnet_umap.png', bbox_inches='tight')\n",
            "        plt.show()\n",
            "        print(\"Saved: results/tier4_resnet_umap.png\")\n",
            "    else:\n",
            "        print(\"⚠️ ResNet50 features (feat_dict) not found in workspace. Skipping UMAP.\")\n",
            "except ImportError:\n",
            "    print(\"⚠️ umap-learn library is not installed. Run 'pip install umap-learn' to execute this cell.\")\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 2. LDA / GBDT Decision Separation (XGBoost OOF Probabilities)\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "print(\"Plotting decision space separation...\")\n",
            "feature_path = PROJECT_ROOT / 'results' / 'baselines_s42' / 'xgboost_oof_subject_preds.csv'\n",
            "if feature_path.exists():\n",
            "    df_preds = pd.read_csv(feature_path)\n",
            "    plt.figure(figsize=(8, 5))\n",
            "    sns.histplot(data=df_preds, x='XGBoost', hue='Label', kde=True, bins=25, alpha=0.6, palette={0: '#1f77b4', 1: '#d62728'})\n",
            "    plt.title('Decision Space Separation (XGBoost OOF Probabilities)', fontsize=12, fontweight='bold')\n",
            "    plt.xlabel('OOF Predicted Probability P(SZ)')\n",
            "    plt.ylabel('Subject Count')\n",
            "    plt.legend(title='Group', labels=['SZ', 'HC'])\n",
            "    plt.grid(True, alpha=0.3)\n",
            "    plt.savefig(PROJECT_ROOT / 'results' / 'lda_class_separation.png', bbox_inches='tight')\n",
            "    plt.show()\n",
            "    print(\"Saved: results/lda_class_separation.png\")\n",
            "else:\n",
            "    print(\"⚠️ OOF predictions not found. Run XGBoost baseline first to generate lda_class_separation.png.\")\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 3. Feature Distribution Before/After Normalization\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "if 'df_clean_train' in locals() and 'df_raw' in locals():\n",
            "    fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "    \n",
            "    # Raw pupil diameter distribution\n",
            "    ax = axes[0]\n",
            "    sns.histplot(df_raw['FIX_PUPIL'].dropna(), ax=ax, color='gray', kde=True, bins=50)\n",
            "    ax.set_title('Raw Pupil Diameter Distribution', fontweight='bold')\n",
            "    ax.set_xlabel('Raw Pupil size (mm)')\n",
            "    ax.set_ylabel('Fixation Count')\n",
            "    \n",
            "    # Normalized pupil diameter distribution\n",
            "    ax = axes[1]\n",
            "    sns.histplot(df_clean_train['FIX_PUPIL'].dropna(), ax=ax, color='green', kde=True, bins=50)\n",
            "    ax.set_title('Fold-Wise Normalized Pupil Diameter', fontweight='bold')\n",
            "    ax.set_xlabel('Z-scored Pupil size')\n",
            "    ax.set_ylabel('Fixation Count')\n",
            "    \n",
            "    plt.tight_layout()\n",
            "    plt.savefig(PROJECT_ROOT / 'results' / 'pupil_normalization_comparison.png', bbox_inches='tight')\n",
            "    plt.show()\n",
            "    print(\"Saved: results/pupil_normalization_comparison.png\")\n",
            "else:\n",
            "    print(\"⚠️ Dataframes (df_raw / df_clean_train) not loaded in workspace. Run Section 1 first.\")\n"
        ]
    }
]

nb["cells"].extend(new_cells)

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Successfully appended {len(new_cells)} cells to {path}!")
