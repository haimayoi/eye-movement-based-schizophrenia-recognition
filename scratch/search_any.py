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
    print(f"Scanning {nb_name}...")
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    for idx, cell in enumerate(nb.get("cells", [])):
        src = "".join(cell.get("source", []))
        if "real_cefam" in src or "real_attention" in src or "Kích thước" in src or "attention thực tế" in src:
            print(f"  -> Found in cell {idx}!")
            # print first 100 chars
            print(f"     Source snippet: {src[:150]}")
