import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

with open('diagnostic_visualization.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# Print Cell 2 source exactly
src = ''.join(nb['cells'][2]['source'])
print('=== CELL 2 SOURCE ===')
print(src)
