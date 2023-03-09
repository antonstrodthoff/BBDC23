import sys
sys.path.insert(0, './task_student/')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from upload_scoring import score_all

def indexToDate(index):
    return pd.to_datetime(index, unit="D", origin=pd.Timestamp("1962.01.01"))

def avg(dataSeries, N):
    meanSeries = dataSeries.rolling(window=N, center=True).mean()
    return meanSeries

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
rollingMeanData = data.copy()

dataLength = data.shape[0]

for i in range(2,7):
    #writes interpolated values into the dataframe
    interpolatedData.iloc[:, i] = interpolatedData.iloc[:, i].interpolate(method="linear", limit_direction="both")
    #writes averaged values into the dataframe
    rollingMeanData.iloc[:, i] = avg(rollingMeanData.iloc[:, i], 50)

xx = np.arange(0, dataLength, 1)

plt.plot(xx, data.iloc[:, plotColumn], linewidth=1, alpha=1)
plt.plot(xx, interpolatedData.iloc[:, plotColumn], linewidth=1, alpha=0.3)
plt.plot(xx, rollingMeanData.iloc[:, plotColumn], linewidth=1, alpha=0.3)
plt.ylabel(data.columns[plotColumn])
plt.legend(["Original", "Interpolated", "Averaged"])
plt.show()

data.fillna(0, inplace=True)
print("The score of the year 2009 compared to the year 2010 is:")
print(score_all(data.iloc[17166:17531, 2:7], data.iloc[16801:17166, 2:7]))
print("(All np.nan values are replaced with 0.0)")

#writeData(data, dataLength)

