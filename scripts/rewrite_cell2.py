import json

with open('diagnostic_visualization.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# Completely rewrite Cell 2 with correct indentation
new_cell2_source = """\
# ── Global Setup ──────────────────────────────────────────────────────────────
import os, sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi': 120, 'font.size': 11,
                     'axes.titlesize': 13, 'axes.labelsize': 11,
                     'axes.spines.top': False, 'axes.spines.right': False})
sns.set_palette('Set2')

# Xac dinh PROJECT_ROOT tu dong dua tren moi truong chay
if 'google.colab' in sys.modules or os.path.exists('/content/drive'):
    try:
        from google.colab import drive
        if not os.path.exists('/content/drive/MyDrive'):
            drive.mount('/content/drive')
        else:
            print('Drive already mounted.')
    except Exception as e:
        print('Drive mount warning:', e)
    PROJECT_ROOT = Path('/content/drive/MyDrive/Eye Movement-Based Schizophrenia Recognition')
else:
    # Neu chay local
    PROJECT_ROOT = Path(os.getcwd())

sys.path.insert(0, str(PROJECT_ROOT))

SEED = 42
np.random.seed(SEED)

COLOR_HC   = '#2196F3'
COLOR_SZ   = '#F44336'
COLOR_T3   = '#4CAF50'
COLOR_T4   = '#FF9800'
COLOR_T5   = '#9C27B0'
COLOR_SOTA = '#607D8B'

print('Setup hoan tat - PROJECT_ROOT:', PROJECT_ROOT)
"""

nb['cells'][2]['source'] = [new_cell2_source]

with open('diagnostic_visualization.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Cell 2 rewritten and saved.')

# Verify
src = ''.join(nb['cells'][2]['source'])
lines = src.splitlines()
for i, line in enumerate(lines[17:26], start=18):
    print(f'{i:3d}: {repr(line)}')
