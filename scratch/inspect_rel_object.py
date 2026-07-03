import sys
import io
from pptx import Presentation

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

prs = Presentation("DAV_slides_v2_pruned.pptx")
slide = prs.slides[0]

for rel_id, rel in slide.part.rels.items():
    if "slideLayout" in rel.reltype:
        print(f"Relationship ID: {rel_id}")
        print(f"Type of rel: {type(rel)}")
        print(f"Attributes: {dir(rel)}")
        print(f"rel.target_ref: {rel.target_ref}")
        print(f"rel._target: {rel._target}")
        if hasattr(rel, '_target_ref'):
            print(f"rel._target_ref: {rel._target_ref}")
