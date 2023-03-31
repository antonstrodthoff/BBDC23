import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

from removeOutliers import *

sylt_data = pd.read_csv("./research_data/List_Reede.csv", sep=",", na_values=["NaN", "nan", "NA", np.nan, None])
sylt_data["Datum"] = pd.to_datetime(pd.to_datetime(sylt_data["Date/Time"], format="%Y-%m-%dT%H:%M").dt.date)
sylt_data.drop(axis=1, columns=["Date/Time"], inplace=True)
sylt_data.fillna(sylt_data.mean(), inplace=True)


for i in sylt_data.columns:
    if(i != "Datum"):
        sylt_data = removeOutliers(sylt_data, column=i, window=8, threshold=0.5)

new_index = pd.date_range(start="01.01.1984", end="01.02.2014", freq='D')

new_index = pd.DataFrame(new_index, columns=["Datum"])

print(sylt_data.info())

sylt_data = pd.merge(new_index, sylt_data, on="Datum", how="left")

print(sylt_data.info())


daten = sylt_data.pop("Datum")
sylt_data = sylt_data.interpolate(method="linear", axis=0)
sylt_data.to_csv("./test2.csv", sep=",", index=True)

sylt_data["[NO3]- [µmol/l]"].plot()
plt.show()

sylt_data["Datum"] = daten
sylt_data.to_csv("./test2.csv", sep=",", index=False)
