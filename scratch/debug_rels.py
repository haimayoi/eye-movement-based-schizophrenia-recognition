from pptx import Presentation

prs = Presentation("DAV_slides_v2_pruned.pptx")
slide = prs.slides[0]

# Print initial rels XML
print("--- BEFORE ---")
print(slide.part.rels.xml)

# Target layout
target_layout = prs.slide_layouts[5]

# Change layout rel target
for rel_id, rel in slide.part.rels.items():
    if "slideLayout" in rel.reltype:
        print(f"\nModifying rel {rel_id}: target={rel.target_ref}")
        rel._target = target_layout.part

# Print rels XML after modification
print("\n--- AFTER ---")
print(slide.part.rels.xml)
