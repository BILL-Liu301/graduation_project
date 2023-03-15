import numpy as np
import pandas as pd
import csv

file_dis = "230314/dis.csv"
file_v = "230314/v_npc.csv"

dis = pd.read_csv(file_dis)
v = pd.read_csv(file_v)

# stamp x y Vx Vy
data = np.zeros([dis.shape[1], 5])

for i in range(data.shape[0]):
    data[i, 0] = i + 1
    data[i, 1] = 0.0
    data[i, 2] = float(dis.columns[i])
    data[i, 3] = 0.0
    data[i, 4] = float(v.columns[i])

print(data)
