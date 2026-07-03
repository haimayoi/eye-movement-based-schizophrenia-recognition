notebooks = [
    "diagnostic_visualization.ipynb",
    "diagnostic_visualization_backup.ipynb",
    "dav276.ipynb"
]

for nb_name in notebooks:
    print(f"\nScanning {nb_name}...")
    try:
        with open(nb_name, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if "tier4_real_attention" in line or "tier4_graph_topology" in line or "case_study_sz_vs_hc" in line:
                    print(f"  Line {idx + 1}: {line.strip()[:150]}")
    except Exception as e:
        print(f"  Error: {e}")
