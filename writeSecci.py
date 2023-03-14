import pandas as pd
import numpy as np
from interpolater import writeColumnToResultFile

data = pd.read_csv("./task_student/bbdc_2023_AWI_data_develop_student.csv", sep=";", na_values=["NA", "NaN", None, np.nan, ""])
data.drop(0, axis=0, inplace=True)
data = data.loc[15706:16070, ["Datum", "NOx"]]
#data["SECCI"] = data["SECCI"].astype(float).interpolate(method="linear", limit_direction="both")
#data["Datum"] = pd.to_datetime(data["Datum"], format="%d.%m.%Y").dt.strftime("%d.%m.2013")
data2004 = pd.DataFrame({"Datum": pd.to_datetime(data["Datum"], format="%d.%m.%Y").dt.strftime("%d.%m.2004"), "NOx": data["NOx"].astype(float).interpolate(method="linear", limit_direction="both")})
data2011 = pd.DataFrame({"Datum": pd.to_datetime(data["Datum"], format="%d.%m.%Y").dt.strftime("%d.%m.2011"), "NOx": data["NOx"].astype(float).interpolate(method="linear", limit_direction="both")})
data2012 = pd.DataFrame({"Datum": pd.to_datetime(data["Datum"], format="%d.%m.%Y").dt.strftime("%d.%m.2012"), "NOx": data["NOx"].astype(float).interpolate(method="linear", limit_direction="both")})
data2013 = pd.DataFrame({"Datum": pd.to_datetime(data["Datum"], format="%d.%m.%Y").dt.strftime("%d.%m.2013"), "NOx": data["NOx"].astype(float).interpolate(method="linear", limit_direction="both")})

data = pd.concat([data2004, data2011, data2012, data2013], ignore_index=True)

print(data.info())
print(data.head(10))
print(data.tail(10))

#writeColumnToResultFile(data, "NOx", 1)

