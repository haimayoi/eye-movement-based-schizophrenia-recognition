import json
import sys
import os

def extract_notebook(ipynb_path, output_txt_path):
    if not os.path.exists(ipynb_path):
        print(f"File not found: {ipynb_path}")
        return
    
    with open(ipynb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    cells = nb.get('cells', [])
    out_lines = []
    
    for idx, cell in enumerate(cells):
        cell_type = cell.get('cell_type')
        source = cell.get('source', [])
        source_str = "".join(source)
        
        out_lines.append(f"\n{'='*40}\nCELL {idx} ({cell_type})\n{'='*40}\n")
        out_lines.append(source_str)
        out_lines.append("\n")
        
        # If it's a code cell and has text/stream or tabular output, extract a summary of it
        outputs = cell.get('outputs', [])
        if outputs:
            out_lines.append(f"\n--- Cell Output ({len(outputs)} outputs) ---")
            for out_idx, out in enumerate(outputs):
                out_type = out.get('output_type')
                out_lines.append(f"\n[Output {out_idx} - {out_type}]:")
                
                # Check for stream text
                if out_type == 'stream':
                    text = "".join(out.get('text', []))
                    # Limiting text output to avoid giant files
                    if len(text) > 3000:
                        text = text[:1500] + "\n... [TRUNCATED] ...\n" + text[-1500:]
                    out_lines.append(text)
                # Check for execute_result or display_data
                elif out_type in ['execute_result', 'display_data']:
                    data = out.get('data', {})
                    if 'text/plain' in data:
                        text_plain = "".join(data['text/plain'])
                        if len(text_plain) > 2000:
                            text_plain = text_plain[:1000] + "\n... [TRUNCATED] ...\n" + text_plain[-1000:]
                        out_lines.append(text_plain)
                    if 'image/png' in data:
                        out_lines.append(" <PNG IMAGE DATA PRESENT>")
                elif out_type == 'error':
                    ename = out.get('ename', '')
                    evalue = out.get('evalue', '')
                    out_lines.append(f"Error: {ename} - {evalue}")
                    traceback = "".join(out.get('traceback', []))
                    out_lines.append(traceback)
            out_lines.append("-" * 30 + "\n")
            
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.writelines(out_lines)
    print(f"Extracted {ipynb_path} to {output_txt_path}")

if __name__ == "__main__":
    extract_notebook(
        r"D:\DAV\Eye Movement-Based Schizophrenia Recognition\run_pipeline_result_with_test.ipynb",
        r"D:\DAV\Eye Movement-Based Schizophrenia Recognition\scratch\run_pipeline_extracted.txt"
    )
    extract_notebook(
        r"D:\DAV\Eye Movement-Based Schizophrenia Recognition\diagnostic_visualization.ipynb",
        r"D:\DAV\Eye Movement-Based Schizophrenia Recognition\scratch\diagnostic_visualization_extracted.txt"
    )
