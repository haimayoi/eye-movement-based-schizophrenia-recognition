import json
from pathlib import Path

path = Path("diagnostic_visualization.ipynb")
if not path.exists():
    print(f"Error: {path} not found.")
    exit(1)

with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find the UMAP code cell and update its source
found = False
for idx, cell in enumerate(nb.get("cells", [])):
    if cell.get("cell_type") == "code":
        src = "".join(cell.get("source", []))
        if "UMAP Projection of ResNet50 Patch Features" in src or "1. UMAP Projection" in src:
            print(f"Found UMAP cell at index {idx}! Updating source...")
            cell["source"] = [
                "# 1. UMAP Projection of ResNet50 Subject-Level Mean Embeddings\n",
                "try:\n",
                "    from umap import UMAP\n",
                "    import matplotlib.pyplot as plt\n",
                "    import numpy as np\n",
                "    from collections import defaultdict\n",
                "\n",
                "    print(\"Running UMAP feature projection...\")\n",
                "    if 'feat_dict' in locals() and feat_dict:\n",
                "        # 1. Aggregate features to subject-level (homogenize shapes)\n",
                "        print(\"Aggregating trial-level features to subject-level mean embeddings...\")\n",
                "        sid_to_label = {}\n",
                "        if 'df_clean_train' in locals():\n",
                "            sid_to_label = dict(zip(df_clean_train['Subject_ID'], df_clean_train['Label']))\n",
                "        else:\n",
                "            # Fallback label lookup convention: 0-103 = SZ, 104-207 = HC\n",
                "            for (sub_id, _) in feat_dict.keys():\n",
                "                sid_to_label[sub_id] = 1 if sub_id <= 103 else 0\n",
                "        \n",
                "        subj_vecs = defaultdict(list)\n",
                "        for (sub_id, img_name), arr in feat_dict.items():\n",
                "            if arr is not None and len(arr.shape) >= 1:\n",
                "                # arr shape: [N_fixations, 2048]\n",
                "                if len(arr.shape) == 1:\n",
                "                    subj_vecs[sub_id].append(arr)\n",
                "                else:\n",
                "                    subj_vecs[sub_id].append(arr.mean(axis=0))\n",
                "        \n",
                "        X_list = []\n",
                "        labels = []\n",
                "        for sid, vecs in subj_vecs.items():\n",
                "            lbl = sid_to_label.get(sid, -1)\n",
                "            if lbl in [0, 1] and len(vecs) > 0:\n",
                "                X_list.append(np.mean(vecs, axis=0))\n",
                "                labels.append(lbl)\n",
                "        \n",
                "        X = np.array(X_list)\n",
                "        y = np.array(labels)\n",
                "        print(f\"Projecting {X.shape[0]} subjects with dimension {X.shape[1]}...\")\n",
                "        \n",
                "        # 2. Run UMAP\n",
                "        reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.15)\n",
                "        embedding = reducer.fit_transform(X)\n",
                "        \n",
                "        # 3. Plot with colors matching label (HC vs SZ)\n",
                "        plt.figure(figsize=(8, 6))\n",
                "        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap='coolwarm', alpha=0.8, edgecolors='w', s=50)\n",
                "        plt.title('UMAP Projection of Subject-Level ResNet50 Mean Embeddings', fontsize=12, fontweight='bold')\n",
                "        plt.xlabel('UMAP Dimension 1')\n",
                "        plt.ylabel('UMAP Dimension 2')\n",
                "        plt.legend(handles=scatter.legend_elements()[0], labels=['HC (Healthy)', 'SZ (Schizophrenia)'], loc='upper right')\n",
                "        plt.grid(True, alpha=0.3)\n",
                "        plt.savefig(PROJECT_ROOT / 'results' / 'tier4_resnet_umap.png', bbox_inches='tight')\n",
                "        plt.show()\n",
                "        print(\"Saved: results/tier4_resnet_umap.png\")\n",
                "    else:\n",
                "        print(\"⚠️ ResNet50 features (feat_dict) not found in workspace. Run previous cells first.\")\n",
                "except ImportError:\n",
                "    print(\"⚠️ umap-learn library is not installed. Run 'pip install umap-learn' to execute this cell.\")\n"
            ]
            found = True
            break

if found:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("Successfully updated the UMAP cell in the notebook!")
else:
    print("Error: Could not find the UMAP cell in the notebook.")
