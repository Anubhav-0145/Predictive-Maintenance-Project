import json
nb=json.load(open('generatepdf.ipynb','r',encoding='utf-8'))
src=''.join(sum([cell.get('source',[]) for cell in nb['cells'] if cell.get('cell_type')=='code'],[]))
print('total triple-quote occurrences:', src.count('"""'))
# print fragments around each occurrence
idx=0
for i in range(len(src)):
    if src.startswith('"""', i):
        print('occurrence at pos', i)
        print(src[max(0,i-50):i+50])
        idx+=1
        if idx>10: break
