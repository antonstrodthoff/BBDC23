import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def indexToDate(index):
    return pd.to_datetime(index, unit="D", origin=pd.Timestamp("1962.01.01"))

#print(indexToDate(10).strftime("%d.%m.%Y"))

data = pd.read_csv("task_student/bbdc_2023_AWI_data_develop_student.csv", sep=";", na_values=["NA", "NaN", None, np.nan])
data.fillna(np.nan, inplace=True)
data = data.drop(0)
data[["SECCI", "Temperatur", "Salinität", "NO2", "NO3", "NOx"]] = data[["SECCI", "Temperatur", "Salinität", "NO2", "NO3", "NOx"]].astype(float)

dataLength = data.shape[0]

def avg(dataSeries, N):
    meanSeries = dataSeries.rolling(window=N, center=True).mean()
    return meanSeries

for i in range(2,8):
    data.iloc[1:, i] = data.iloc[1:, i].interpolate(method="linear", limit_direction="both")        #writes interpolated values into the dataframe
    data.iloc[1:, i] = avg(data.iloc[1:, i], 50)                                                    #writes averaged values into the dataframe

#averageData = avg(data.iloc[1:, 3], 50)
    
xx = np.arange(0, dataLength-1, 1)
plt.plot(xx, data.iloc[1:, 8])
plt.ylabel(data.columns[8])
plt.show()



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

#writeData(data, dataLength)
