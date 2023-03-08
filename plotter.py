import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def indexToDate(index):
    return pd.to_datetime(index, unit="D", origin=pd.Timestamp("1962.01.01"))

print(indexToDate(1).strftime("%Y.%m.%d"))

csv = pd.read_csv("task_student/bbdc_2023_AWI_data_develop_student.csv", sep=";", na_values=["NA", "NaN", None, np.nan])
csv.fillna(np.nan, inplace=True)
csv = csv.drop(0)
csv[["SECCI", "Temperatur", "Salinität", "NO2", "NO3", "NOx"]] = csv[["SECCI", "Temperatur", "Salinität", "NO2", "NO3", "NOx"]].astype(float)

datalength = csv.shape[0]

xx = np.arange(0, datalength - 1, 1)
yy = csv.iloc[1:, 3].interpolate(method="linear", limit_direction="both")

#plt.plot(xx, yy)
#plt.show()

