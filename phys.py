import numpy as np
m = np.array([0,0,1,1,2])
tmp = np.zeros(5)
l = [1,1,1,1,1]
n = []
title = ""
def ap(sub_n,index):
    for i in range(index,len(sub_n)):
        sub_n[i] = 1
        tmp_loc = sub_n.copy()
        tmp_loc = np.append(tmp_loc,np.dot(sub_n,m))
        n.append(tmp_loc)
        sub_n[i] = 0

for i in range(len(l)):
    if l[i] == 0:   continue
    l[i] = 0
    tmp[i] = 1
    ap(tmp,i+1)
    tmp[i] = 0

for i in range(5):
    title = title + f"n{i+1}" + " "
title = title + " E"
print(title)
for ele in n:
    print(ele)
