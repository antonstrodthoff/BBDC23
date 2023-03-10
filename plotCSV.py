import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import interpolater as ip

def plotColumn(fileName="", plotColumn=2, separator=";", dropFirstNValues=1, dateFormat="%d.%m.%Y", label="", window=0):
    data = pd.read_csv(fileName, sep=separator, na_values=["NA", "NaN", None, np.nan, ""], on_bad_lines="skip")
    data.fillna(np.nan, inplace=True)
    data = data.drop([i for i in range(0, dropFirstNValues)], axis=0)
    if window == 0:
        data.iloc[:, plotColumn] = data.iloc[:, plotColumn].replace(0, np.nan).astype(float)
    else:
        data.iloc[:, plotColumn] = data.iloc[:, plotColumn].replace(0, np.nan).astype(float).rolling(window=window, min_periods=window//2).mean()

    print(data.head(10))
    ip.writeColumnToResultFile(data, plotColumn, data.shape[0])

    dataLength = data.shape[0]
    xx = [dt.datetime.strptime(d, dateFormat).date() for d in data.iloc[:, 0]]

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m.%Y"))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=365*5))
    plt.plot(xx, data.iloc[:, plotColumn], linewidth=1, alpha=1, label=label)
    plt.gcf().autofmt_xdate()

#plot1
plotColumn(fileName="task_student/bbdc_2023_AWI_data_develop_student.csv", plotColumn=2, separator=";", dropFirstNValues=1, label="training")
#plot2
#plotColumn(fileName="research_data/List_Reede.csv", plotColumn=3, separator=",", dropFirstNValues=1, dateFormat="%Y-%m-%dT%H:%M", label="reasearched", window=10)

#plt.legend()
#plt.show()

#test = pd.DataFrame(np.array([["01.01.2000", "02.01.2000", "03.01.2000"], [1,2,3]]).T)
#test = test.where(test.iloc[:, 0].astype(str) == "01.01.2000", np.nan, inplace=False).dropna(how="all")
#print(test)

#data = pd.read_csv("task_student/bbdc_2023_AWI_data_develop_student.csv", sep=";", na_values=["NA", "NaN", None, np.nan, ""])
#print(pd.to_datetime(data.iloc[:, 0]).info())
#data = data.where(data.iloc[:, 0].astype(str) == "01.01.2000", np.nan, inplace=False).dropna(how="all")
#print(data)

