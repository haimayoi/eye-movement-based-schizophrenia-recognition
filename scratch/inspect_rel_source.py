import sys
import io
import inspect
from pptx import Presentation

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

prs = Presentation("DAV_slides_v2_pruned.pptx")
slide = prs.slides[0]

for rel_id, rel in slide.part.rels.items():
    if "slideLayout" in rel.reltype:
        print("=== Class _Relationship Source ===")
        print(inspect.getsource(rel.__class__))
        
        print("\n=== Property target_ref Source ===")
        print(inspect.getsource(rel.__class__.target_ref.fget))
        
        print("\n=== Property target_partname Source ===")
        if hasattr(rel.__class__, 'target_partname'):
            print(inspect.getsource(rel.__class__.target_partname.fget))
        break
