import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import csv_trans, csv_reshape

# 前提参数
batch_size = 20

# 读取数据
path = './1/loss.csv'
csv_data = pd.read_csv(path)
csv_Time, csv_Step, csv_Value = csv_trans(csv_data)

# 重组数据，分析loss
csv_Value_analyse = csv_reshape(csv_Value, batch_size)
csv_Step_analyse = csv_reshape(csv_Value, batch_size)
plt.figure()
for i in range(csv_Step_analyse.shape[0]):
    ab = np.polyfit(csv_Step_analyse[i, :], csv_Value_analyse[i, :], 1)
    plt.plot(csv_Step_analyse[i, :], np.poly1d(ab)(csv_Step_analyse[i, :]))
# plt.plot(csv_Step, csv_Value)
plt.show()

# 分析lr和loss之间的线性关系

