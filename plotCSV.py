import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plotColumn = 4
dropFirstNValues = 0

data = pd.read_csv("task_student/bbdc_2023_AWI_data_develop_student.csv", sep=";", na_values=["NA", "NaN", None, np.nan])
data.fillna(np.nan, inplace=True)
data = data.drop([i for i in range(0, dropFirstNValues)], axis=0)
data.iloc[:, plotColumn] = data.iloc[:, plotColumn].replace(0, np.nan)

dataLength = data.shape[0]
xx = np.arange(0, dataLength, 1)

plt.plot(xx, data.iloc[:, plotColumn], linewidth=1, alpha=1)
plt.ylabel(data.columns[plotColumn])
plt.legend(["Original"])
plt.show()

