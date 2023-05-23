import pandas as pd
import numpy as np
import pickle
import pdb

def export_excel(res):
    pf = pd.DataFrame(list([res]))
    order = list(res.keys())
    pf = pf[order]
    file_path = pd.ExcelWriter('./raidar_noe.xlsx')
    pf.to_excel(file_path,encoding = 'utf-8',index = False)
    file_path.save()
    
f_read = open('all_vc.pkl', 'rb')
all_vc = pickle.load(f_read)
f_read.close()
f_read = open('noe.pkl', 'rb')
res = pickle.load(f_read)
f_read.close()
#for i in range(len(res)):
#    res[i] = dict(res[i])
#    sub = all_vc.keys() - res[i].keys()
#    for j in sub:
#        res[i][j] = 0
#    assert len(res[i].keys()) == len(all_vc.keys())
sub = all_vc.keys() - res.keys()
for j in sub:
    res[j] = 0
assert len(res.keys()) == len(all_vc.keys())

#pdb.set_trace()
export_excel(res)
