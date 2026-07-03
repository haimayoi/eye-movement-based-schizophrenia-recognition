import json, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

with open('diagnostic_visualization.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
print('Total cells:', len(cells))

# Print full code cells
for i, c in enumerate(cells):
    src = ''.join(c.get('source', []))
    ctype = c['cell_type']
    if ctype == 'code':
        print('=== Cell', i, '(code) ===')
        print(src)
        print()
