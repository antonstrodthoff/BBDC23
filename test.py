import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

from removeOutliers import *

sylt_data = pd.read_csv("./research_data/List_Reede.csv", sep=",", na_values=["NaN", "nan", "NA", np.nan, None])
sylt_data["Datum"] = pd.to_datetime(pd.to_datetime(sylt_data["Date/Time"], format="%Y-%m-%dT%H:%M").dt.strftime("%d.%m.%Y"))
sylt_data.drop(axis=1, columns=["Date/Time"], inplace=True)
sylt_data.fillna(sylt_data.mean(), inplace=True)
sylt_data.set_index("Datum", inplace=True)

sylt_data = sylt_data[~sylt_data.index.duplicated()]

for i in sylt_data.columns:
    if(i != "Datum"):
        sylt_data = removeOutliers(sylt_data, column=i, window=8, threshold=0.5)

new_index = pd.date_range(start=sylt_data.index.min(), end=sylt_data.index.max(), freq='D')

sylt_data = sylt_data.reindex(new_index)

#sylt_data = sylt_data.interpolate(method="linear", axis=0)
#sylt_data.to_csv("./test2.csv", sep=",", index=True)
plt.show()