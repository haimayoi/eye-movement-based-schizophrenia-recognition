import json
from pathlib import Path

path = Path("diagnostic_visualization.ipynb")
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx, cell in enumerate(nb.get("cells", [])):
    if cell.get("cell_type") == "code":
        src = "".join(cell.get("source", []))
        if "df_hc" in src:
            print(f"Found df_hc in cell {idx}!")
            # print the first 250 characters of this cell
            print("--- CELL SOURCE ---")
            print(src[:600])
            print("-------------------")
