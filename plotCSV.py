import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plotColumn(fileName="", plotColumn=2, separator=";", dropFirstNValues=1, label=""):
    data = pd.read_csv(fileName, sep=separator, na_values=["NA", "NaN", None, np.nan])
    data.fillna(np.nan, inplace=True)
    data = data.drop([i for i in range(0, dropFirstNValues)], axis=0)
    data.iloc[:, plotColumn] = data.iloc[:, plotColumn].replace(0, np.nan).astype(float)

    dataLength = data.shape[0]
    xx = np.arange(0, dataLength, 1)

    plt.plot(xx, data.iloc[:, plotColumn], linewidth=1, alpha=1, label=label)

#plot1
plotColumn(fileName="task_student/bbdc_2023_AWI_data_develop_student.csv", plotColumn=2, separator=";", dropFirstNValues=1, label="training")
#plot2
plotColumn(fileName="task_student/bbdc_2023_AWI_data_develop_student.csv", plotColumn=2, separator=";", dropFirstNValues=1, label="training2")

plt.legend()
plt.show()

