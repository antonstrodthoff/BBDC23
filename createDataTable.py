import pandas as pd
import numpy as np
import datetime as dt

sylt_data = pd.read_csv("./research_data/List_Reede.csv", sep=",", na_values=["NaN", "nan", "NA", np.nan, None])
sylt_data["Datum"] = pd.to_datetime(pd.to_datetime(sylt_data["Date/Time"], format="%Y-%m-%dT%H:%M").dt.date)
sylt_data.drop(axis=1, columns=["Date/Time"], inplace=True)
sylt_data.fillna(sylt_data.mean(), inplace=True)

date_Column = pd.DataFrame(pd.date_range(start="2003-01-01", end="2014-12-31", freq="D"))
date_Column.columns = ["Datum"]

all_data = pd.merge(date_Column, sylt_data, on="Datum", how="outer")

print(all_data.head())
print(all_data.info())

