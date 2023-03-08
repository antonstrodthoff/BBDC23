import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

csv = pd.read_csv("task_student/bbdc_2023_AWI_data_develop_student.csv", sep=";")
#csv.fillna(0, inplace=True)
datalength = csv.shape[0]

xx = np.arange(0, datalength - 1, 1)
yy = list(np.array(csv.iloc[1:, 3]))

nans, x = nan_helper(yy)
yy[nans] = np.interp(x(nans), x(~nans), y[~nans])

print(xx)
print(xx.shape)
print(yy)
print(yy.shape)

plt.plot(xx, yy)
plt.show()

#merge conflict comment
