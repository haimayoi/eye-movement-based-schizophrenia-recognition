import json
from pathlib import Path

path = Path("diagnostic_visualization.ipynb")
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

found = False
for idx, cell in enumerate(nb.get("cells", [])):
    if cell.get("cell_type") == "code":
        src = "".join(cell.get("source", []))
        if "Kích thước nút" in src:
            print(f"Found Vietnamese string in cell {idx}!")
            # Let's inspect the code lines
            for line_idx, line in enumerate(cell["source"]):
                if "Kích thước nút" in line:
                    print(f"Line {line_idx}: {line}")
                    # Replace with English
                    cell["source"][line_idx] = line.replace(
                        "Kích thước nút đại diện cho độ lớn attention thực tế trích xuất từ GNN và CEFAM",
                        "Node size represents the actual attention magnitude extracted from GNN and CEFAM"
                    )
                    found = True

if found:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("Successfully replaced and saved notebook!")
else:
    print("Could not find the target Vietnamese string in notebook cells.")
