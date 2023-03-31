import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("./final.csv", sep=";", na_values=["NaN", "nan", "NA", np.nan, None])
data.drop(0, axis=0, inplace=True)
data.drop("Uhrzeit", axis=1, inplace=True)
data.iloc[:, 1:] = data.iloc[:, 1:].astype(float).rolling(window=50, min_periods=25, center=True).mean()
data.drop("Datum", axis=1, inplace=True)

data2 = pd.read_csv("./final.csv", sep=";", na_values=["NaN", "nan", "NA", np.nan, None])
data2.iloc[1:, 2:] = data
data2.to_csv("./final_2.csv", sep=";", index=False)
print(data2.head())
print(data2.info())

print(data.head())
print(data.info())

data.plot()
plt.show()
