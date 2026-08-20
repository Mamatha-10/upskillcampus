from pathlib import Path
import json

path = Path('traffic_project/notebooks/Traffic_Forecasting_standard_executed.ipynb')
nb = json.loads(path.read_text(encoding='utf-8', errors='ignore'))
for i, cell in enumerate(nb.get('cells', [])):
    if i >= 60:
        break
    cell_type = cell.get('cell_type')
    if cell_type == 'markdown':
        text = ''.join(cell.get('source', [])).strip()
        if text:
            print(f'MARKDOWN[{i}]: {text[:300].replace("\n", " ")}')
            print('---')
    elif cell_type == 'code':
        src = ''.join(cell.get('source', [])).strip()
        if src:
            print(f'CODE[{i}]: {src[:300].replace("\n", " ")}')
            outputs = cell.get('outputs', [])
            for out in outputs[:2]:
                if 'text' in out:
                    print(' OUTPUT:', ''.join(out['text'])[:300].replace("\n", " "))
                elif out.get('output_type') == 'display_data' and out.get('data', {}).get('text/plain'):
                    print(' OUTPUT:', ''.join(out['data']['text/plain'])[:300].replace("\n", " "))
            print('---')
