import json
nb=json.load(open('generatepdf.ipynb','r',encoding='utf-8'))
for i,cell in enumerate(nb['cells']):
    if cell.get('cell_type')=='code' and len(''.join(cell.get('source',[])))>100:
        src=''.join(line+('\n' if not line.endswith('\n') else '') for line in cell['source'])
        lines=src.splitlines()
        for idx in range(730, 762):
            if idx-1 < len(lines):
                print(f'{idx:4}: {lines[idx-1]!r}')
        break
