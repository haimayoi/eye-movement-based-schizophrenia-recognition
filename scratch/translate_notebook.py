import json
from pathlib import Path

path = Path("diagnostic_visualization.ipynb")
if not path.exists():
    print(f"Error: {path} not found.")
    exit(1)

with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

replacements = {
    "Kích thước nút đại diện cho độ lớn attention thực tế trích xuất từ GNN và CEFAM": 
        "Node size represents the actual attention weight magnitude from GNN and CEFAM",
    
    "Kích thước nút ∝ duration, màu sắc ∝ pupil size, mũi tên = thời gian":
        "Node size ∝ duration, color ∝ pupil size, arrows = temporal sequence",
        
    "Đường kính đồng tử (normalized pupil)":
        "Normalized pupil diameter",
        
    "Kích thước điểm ∝ xác suất dự đoán SZ của CEFAM; màu = thể loại ảnh":
        "Node size ∝ CEFAM SZ probability; color = stimulus category",
        
    "Trước khi lọc": "Before filtering",
    "Sau khi lọc": "After filtering",
    "Bị loại": "Excluded",
    "Biên lọc": "Filter boundary",
    "Số lượng fixation": "Fixation count"
}

modified_cells = 0
for idx, cell in enumerate(nb.get("cells", [])):
    if cell.get("cell_type") == "code":
        source = cell.get("source", [])
        cell_modified = False
        for line_idx, line in enumerate(source):
            for vi, en in replacements.items():
                if vi in line:
                    source[line_idx] = line.replace(vi, en)
                    cell_modified = True
        if cell_modified:
            modified_cells += 1
            print(f"Translated cell {idx}!")

if modified_cells > 0:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"Successfully translated {modified_cells} cells in {path}!")
else:
    print("No Vietnamese strings found or already translated.")
