import json
from pathlib import Path

notebooks = [
    "diagnostic_visualization.ipynb",
    "diagnostic_visualization_backup.ipynb",
    "dav276.ipynb"
]

for nb_name in notebooks:
    path = Path(nb_name)
    if not path.exists():
        continue
    print(f"\nScanning {nb_name}...")
    try:
        with open(path, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except Exception as e:
        print(f"Error loading {nb_name}: {e}")
        continue

    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            if "Real Attention" in src or "real_attention" in src or "real_cefam_attn" in src:
                print(f"Found in cell {idx}!")
                for line_idx, line in enumerate(cell["source"]):
                    print(f"  Line {line_idx}: {repr(line)}")
