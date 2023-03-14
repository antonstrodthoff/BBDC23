import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
#import interpolater as ip

def plotColumn(fileName="", plotColumn=2, separator=";", dropFirstNValues=1, dateFormat="%d.%m.%Y", label="", window=0, alpha=1):
    data = pd.read_csv(fileName, sep=separator, na_values=["NA", "NaN", None, np.nan, ""], on_bad_lines="skip")
    data.fillna(np.nan, inplace=True)
    data = data.drop([i for i in range(0, dropFirstNValues)], axis=0)
    if window == 0:
        data.iloc[:, plotColumn] = data.iloc[:, plotColumn].replace(0, np.nan).astype(float)
    else:
        data.iloc[:, plotColumn] = data.iloc[:, plotColumn].replace(0, np.nan).astype(float).interpolate(limit_direction="both").rolling(window=window, min_periods=window//2).mean()

    #ip.writeColumnToResultFile(data, plotColumn, data.shape[0])

    dataLength = data.shape[0]
    xx = [dt.datetime.strptime(d, dateFormat).date() for d in data.iloc[:, 0]]

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m.%Y"))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=365*5))

    plt.plot(xx, data.iloc[:, plotColumn], linewidth=0.5, alpha=alpha, label=label)
    plt.gcf().autofmt_xdate()

def plotSecciOverTime():
    data = pd.read_csv("task_student/bbdc_2023_AWI_data_develop_student.csv", sep=";", na_values=["NA", "NaN", None, np.nan, ""], on_bad_lines="skip").drop(0).dropna(how="any", subset=["Uhrzeit", "SECCI"]).iloc[:, [1,2]]
    data.iloc[:,0] = pd.to_datetime(data.iloc[:,0], format="%H:%M").dt.hour * 60 + pd.to_datetime(data.iloc[:,0], format="%H:%M").dt.minute

    plt.scatter(data.iloc[:, 1], data.iloc[:, 0], linewidth=10, alpha=0.01, marker="o")
    plt.xticks(np.arange(0, 10, 1) * 10)


#plot training columns
#plotColumn(fileName="task_student/bbdc_2023_AWI_data_develop_student.csv", plotColumn=2, separator=";", dropFirstNValues=1, label="t_secci", window=0)
#plotColumn(fileName="task_student/bbdc_2023_AWI_data_develop_student.csv", plotColumn=3, separator=";", dropFirstNValues=1, label="t_temp", window=0)
#plotColumn(fileName="task_student/bbdc_2023_AWI_data_develop_student.csv", plotColumn=4, separator=";", dropFirstNValues=1, label="t_salinity", window=0)
#plotColumn(fileName="task_student/bbdc_2023_AWI_data_develop_student.csv", plotColumn=5, separator=";", dropFirstNValues=1, label="t_no2", window=0)
plotColumn(fileName="task_student/bbdc_2023_AWI_data_develop_student.csv", plotColumn=6, separator=";", dropFirstNValues=1, label="t_no3", window=50)
#plotColumn(fileName="task_student/bbdc_2023_AWI_data_develop_student.csv", plotColumn=7, separator=";", dropFirstNValues=1, label="t_nox", window=0)

#plot researched columns
#plotColumn(fileName="research_data/List_Reede.csv", plotColumn=3, separator=",", dropFirstNValues=1, dateFormat="%Y-%m-%dT%H:%M", label="r_salinity", window=10, alpha=1)
plotColumn(fileName="research_data/List_Reede.csv", plotColumn=11, separator=",", dropFirstNValues=1, dateFormat="%Y-%m-%dT%H:%M", label="r_no3", window=50, alpha=1)

#plotSecciOverTime()

plt.legend()
plt.show()

