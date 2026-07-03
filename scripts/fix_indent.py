import json

with open('diagnostic_visualization.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# Fix Cell 2: replace the broken indentation block
old_block = (
    "    try:\n"
    "        from google.colab import drive\n"
    "        if not os.path.exists('/content/drive/MyDrive'):\n"
    "        drive.mount('/content/drive')\n"
    "    else:\n"
    "        print('Drive already mounted.')\n"
    "    except:\n"
    "        pass\n"
)

new_block = (
    "    try:\n"
    "        from google.colab import drive\n"
    "        if not os.path.exists('/content/drive/MyDrive'):\n"
    "            drive.mount('/content/drive')\n"
    "        else:\n"
    "            print('Drive already mounted.')\n"
    "    except:\n"
    "        pass\n"
)

src = ''.join(nb['cells'][2]['source'])
if old_block in src:
    fixed_src = src.replace(old_block, new_block)
    nb['cells'][2]['source'] = [fixed_src]
    print('Fix applied successfully.')
    print('\n=== FIXED CELL 2 ===')
    print(fixed_src)
else:
    print('Pattern not found. Current source:')
    print(repr(src))

with open('diagnostic_visualization.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('\nNotebook saved.')
