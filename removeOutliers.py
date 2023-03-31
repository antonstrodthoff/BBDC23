import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# column = "[NO3]- [µmol/l]"
# data = pd.read_csv("./research_data/List_Reede.csv", sep=",", na_values=["NaN", "nan", "NA", np.nan, None])[["Date/Time", column]]
# data["Datum"] = pd.to_datetime(pd.to_datetime(data["Date/Time"], format="%Y-%m-%dT%H:%M").dt.strftime("%d.%m.%Y"))
# data.drop("Date/Time", axis=1, inplace=True)
# data.dropna(how="any", axis=0, inplace=True)
# data = data.reindex(columns=["Datum", column])



# data2 = pd.read_csv("./task_student/bbdc_2023_AWI_data_develop_student.csv", sep=";", na_values=["NaN", "nan", "NA", np.nan, None])
# data2.drop(0, axis=0, inplace=True)
# data2.iloc[:,2:] = data2.iloc[:,2:].astype(float)
# data2["Datum"] = pd.to_datetime(pd.to_datetime(data2["Datum"], format="%d.%m.%Y").dt.strftime("%d.%m.%Y"))
# data2.dropna(how="any", axis=0, inplace=True)
# data2 = data2.reindex()

def removeOutliers(data, column="SECCI", window=8, threshold=0.2, plot=False):
    if(plot):
        data[column].plot(linewidth=1)

    #remove ouliers
    local_mean = data[column].rolling(window=window, center=True).mean()
    local_std = data[column].rolling(window=window, center=True).std()
    upper = local_mean + (local_std * threshold)
    lower = local_mean - (local_std * threshold)
    data[column] = np.where((data[column] > upper) | (data[column] < lower), local_mean, data[column])

    if(plot):
        data[column].plot(linewidth=1)
        plt.show()

    return data

