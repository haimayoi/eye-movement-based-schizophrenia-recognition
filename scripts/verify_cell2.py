import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

with open('diagnostic_visualization.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

src = ''.join(nb['cells'][2]['source'])
# Show only the drive.mount block
lines = src.splitlines()
for i, line in enumerate(lines):
    if 'google.colab' in line or 'drive' in line or 'PROJECT_ROOT' in line.split('#')[0]:
        print(f'{i+1:3d}: {line}')
