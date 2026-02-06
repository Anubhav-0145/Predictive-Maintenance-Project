import json,traceback
nb=json.load(open('generatepdf.ipynb','r',encoding='utf-8'))
# find first large code cell containing our PDF code
for i,cell in enumerate(nb['cells']):
    if cell.get('cell_type')=='code' and len(''.join(cell.get('source',[])))>100:
        src=''.join(line+('\n' if not line.endswith('\n') else '') for line in cell['source'])
        print(f'--- Cell {i+1} (len {len(src)}) ---')
        print('--- START OF CELL ---')
        print(src)
        print('--- END OF CELL ---')
        try:
            compile(src,'<cell>','exec')
            print('-> compile OK')
        except Exception:
            print('-> compile ERROR:')
            traceback.print_exc()
        break
