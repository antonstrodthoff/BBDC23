import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

column = "SECCI"

original_data = pd.read_csv("./task_student/bbdc_2023_AWI_data_develop_student.csv", sep=";", na_values=["NaN", "nan", "NA", np.nan, None])[["Datum", column]]
original_data.drop(axis=0, index=0, inplace=True)
original_data["Datum"] = pd.to_datetime(pd.to_datetime(original_data["Datum"], format="%d.%m.%Y").dt.strftime("%d.%m.%Y"))
original_data.dropna(how="any", axis=0, inplace=True)

sylt_data = pd.read_csv("./research_data/List_Reede.csv", sep=",", na_values=["NaN", "nan", "NA", np.nan, None])
sylt_data["Datum"] = pd.to_datetime(pd.to_datetime(sylt_data["Date/Time"], format="%Y-%m-%dT%H:%M").dt.strftime("%d.%m.%Y"))
sylt_data.drop(axis=1, columns=["Date/Time"], inplace=True)
sylt_data.fillna(sylt_data.mean(), inplace=True)

sylt_data.plot(x="Datum")

all_data = pd.merge(original_data, sylt_data, on="Datum", how="outer")
all_data = all_data.sort_values(by="Datum").iloc[3218:9705, :]
all_data.iloc[:, 1:] = all_data.iloc[:, 1:].astype(float).interpolate(method="linear", axis=0)

print(original_data.info())
print(original_data.head(10))

print(sylt_data.info())
print(sylt_data.head(10))

print(all_data.info())
print(all_data.head(100))

all_data.to_csv("~/Desktop/test.csv", sep=",", index=False)

plt.show()

