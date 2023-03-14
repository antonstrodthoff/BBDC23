import pandas as pd
import datetime
import numpy as np

data = pd.DataFrame({"Datum":pd.date_range(datetime.datetime.strptime("5.1.1984", "%d.%m.%Y"), periods=10954), "Uhrzeit":[np.nan for _ in range(10954)], "SECCI":[np.nan for _ in range(10954)], "Temperatur":[np.nan for _ in range(10954)], "Salinität":[np.nan for _ in range(10954)], "NO2":[np.nan for _ in range(10954)], "NO3":[np.nan for _ in range(10954)], "NOx":[np.nan for _ in range(10954)]})
data.iloc[:, 0] = data.iloc[:, 0].dt.strftime("%d.%m.%Y")

#print(data)

data.to_csv("research_data/List_Reede_interpolated.csv", sep=";", index=False, lineterminator="\n")