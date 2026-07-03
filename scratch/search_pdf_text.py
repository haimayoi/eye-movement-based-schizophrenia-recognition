import sys
try:
    import pypdf
except ImportError:
    import subprocess
    print("Installing pypdf...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pypdf"])
    import pypdf

reader = pypdf.PdfReader("Eye_Movement_Based_Schizophrenia_Recognition.pdf")
print(f"Total pages: {len(reader.pages)}")

found_any = False
for idx, page in enumerate(reader.pages):
    text = page.extract_text()
    if "Topology" in text or "topology" in text or "Graph" in text:
        print(f"\n--- Page {idx + 1} contains 'Graph' or 'Topology' ---")
        found_any = True
        # print matching lines
        for line in text.split("\n"):
            if any(w in line for w in ["Topology", "topology", "Graph", "Fig.", "Figure", "fig"]):
                print(f"  {line.strip()[:120]}")

if not found_any:
    print("Could not find any occurrences of 'Graph' or 'Topology' in the PDF text.")
