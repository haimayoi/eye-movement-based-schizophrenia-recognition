"""
Fix script for diagnostic_visualization.ipynb
Applies all identified bug fixes directly to the notebook JSON.
"""
import json
from pathlib import Path

NB_PATH = Path('diagnostic_visualization.ipynb')
BACKUP_PATH = Path('diagnostic_visualization_fixed_backup.ipynb')

with open(NB_PATH, encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
print(f'Total cells: {len(cells)}')

fixes_applied = []

for i, cell in enumerate(cells):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])

    # ===== FIX 1: Remove duplicate drive.mount in Cell 1 =====
    if src.strip() == "from google.colab import drive\ndrive.mount('/content/drive')" or \
       src.strip() == "from google.colab import drive\r\ndrive.mount('/content/drive')":
        # Replace with a comment-only cell marking it as removed
        new_src = "# Cell 1 removed - drive mount is handled in Cell 2 (Global Setup)\n# Run Cell 2 directly instead."
        cell['source'] = [new_src]
        fixes_applied.append(f'Cell {i}: Removed duplicate drive.mount')

    # ===== FIX 2: Fix drive.mount in Global Setup (Cell 2) =====
    if 'drive.mount' in src and 'PROJECT_ROOT' in src and 'force_remount=True' in src:
        new_src = src.replace(
            "drive.mount('/content/drive', force_remount=True)",
            "if not os.path.exists('/content/drive/MyDrive'):\n        drive.mount('/content/drive')\n    else:\n        print('Drive already mounted.')"
        )
        if new_src != src:
            cell['source'] = [new_src]
            fixes_applied.append(f'Cell {i}: Fixed force_remount=True -> conditional mount check')

    # ===== FIX 3: Fix bondage=None invalid kwarg in scatter (Cell 39) =====
    if 'bondage=None' in src:
        new_src = src.replace(', bondage=None', '').replace('bondage=None, ', '').replace('bondage=None', '')
        cell['source'] = [new_src]
        fixes_applied.append(f'Cell {i}: Removed invalid `bondage=None` kwarg from scatter()')

    # ===== FIX 4: Guard for ResNet50 npy load (Cell 35) =====
    if 'feature_dict_ResNet50.npy' in src and 'allow_pickle=True' in src and 'if feat_path.exists()' not in src:
        new_src = (
            "import numpy as np\n"
            "from collections import defaultdict\n\n"
            "feat_path = PROJECT_ROOT / 'data' / 'external' / 'feature_dict_ResNet50.npy'\n"
            "if feat_path.exists():\n"
            "    size_mb = feat_path.stat().st_size / 1e6\n"
            "    print(f'Loading ResNet50 features ({size_mb:.1f} MB)...')\n"
            "    feat_dict = np.load(feat_path, allow_pickle=True).item()\n"
            "    print(f'Loaded {len(feat_dict)} entries.')\n"
            "else:\n"
            "    print('⚠️ feature_dict_ResNet50.npy not found. Tier 4C-E will be skipped.')\n"
            "    feat_dict = {}\n"
        )
        cell['source'] = [new_src]
        fixes_applied.append(f'Cell {i}: Guarded ResNet50 npy load with file existence check')

    # ===== FIX 5: Guard for Cell 36 that uses feat_dict =====
    if 'for (sub_id, img_name), arr in feat_dict.items()' in src and 'if not feat_dict' not in src:
        guard = "if not feat_dict:\n    print('⚠️ feat_dict is empty. Skipping Tier 4D.')\nelse:\n"
        # Indent the original cell content
        indented = '\n'.join('    ' + line if line.strip() else line for line in src.splitlines())
        new_src = guard + indented + '\n'
        cell['source'] = [new_src]
        fixes_applied.append(f'Cell {i}: Guarded feat_dict usage with emptiness check')

print('\nFixes applied:')
for fix in fixes_applied:
    print(' -', fix)

# Write fixed notebook
with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'\n✅ Fixed notebook saved to {NB_PATH}')
print(f'   Original backup at: diagnostic_visualization_backup.ipynb (already exists)')
