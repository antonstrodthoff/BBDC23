import sys
sys.path.insert(0, './task_student/')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from upload_scoring import score_all

def indexToDate(index):
    return pd.to_datetime(index, unit="D", origin=pd.Timestamp("1962.01.01"))

def writeData(data, dataLength):
    results = pd.read_csv("task_student/bbdc_2023_AWI_data_evaluate_skeleton_student.csv", sep=";")
    resultLength = results.shape[0]

    for i in range(1,resultLength):
        datestring = results.iloc[i, 0]
        for j in range((dataLength-364), dataLength):
            if(indexToDate(j).strftime("%d.%m.") == datestring[:-4]):
                results.iloc[i, 2:8] = data.iloc[j-1, 2:8]
                break

    results.to_csv("task_student/bbdc_2023_AWI_data_evaluate_skeleton_student_out.csv", sep=";", index=False)

plotColumn = 2

data = pd.read_csv("task_student/bbdc_2023_AWI_data_develop_student.csv", sep=";", na_values=["NA", "NaN", None, np.nan])
data.fillna(np.nan, inplace=True)
data = data.drop(0)
data[["SECCI", "Temperatur", "Salinität", "NO2", "NO3", "NOx"]] = data[["SECCI", "Temperatur", "Salinität", "NO2", "NO3", "NOx"]].astype(float)

interpolatedData = data.copy()
interpolatedData = interpolatedData.interpolate(method="linear", limit_direction="both")
    
rollingMeanData = interpolatedData.copy()
rollingMeanData.iloc[:, 2:7] = rollingMeanData.iloc[:, 2:7].rolling(window=50, center=True, min_periods=25).mean()

dataLength = data.shape[0]
xx = np.arange(0, dataLength, 1)

plt.plot(xx, data.iloc[:, plotColumn], linewidth=1, alpha=1)
plt.plot(xx, interpolatedData.iloc[:, plotColumn], linewidth=1, alpha=0.3)
plt.plot(xx, rollingMeanData.iloc[:, plotColumn], linewidth=1, alpha=0.3)
plt.ylabel(data.columns[plotColumn])
plt.legend(["Original", "Interpolated", "Averaged"])
#plt.show()

#allowed yearToCompare values are 1962 to 2008
yearToCompare = 2008
startIndex = (yearToCompare - 1962 + 1) * 365
print(f"The score of the year %i compared to the year %i is:" % (yearToCompare, yearToCompare + 1))
print(score_all(interpolatedData.iloc[startIndex:startIndex+365, 2:7], interpolatedData.iloc[startIndex-365:startIndex, 2:7]))
print(score_all(rollingMeanData.iloc[startIndex:startIndex+365, 2:7], rollingMeanData.iloc[startIndex-365:startIndex, 2:7]))

#writeData(data, dataLength)

