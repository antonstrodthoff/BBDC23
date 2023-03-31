import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense

from tensorflow import keras
from tensorflow.keras import layers
from removeOutliers import *


column = "NO3"
timeshift = -10


prediction_dates = pd.read_csv("./task_student/bbdc_2023_AWI_data_evaluate_skeleton_student.csv", sep=";")[["Datum"]]
prediction_dates.drop(axis=0, index=0, inplace=True)
prediction_dates["Datum"] = pd.to_datetime(prediction_dates["Datum"], format="%d.%m.%Y")

original_data = pd.read_csv("./task_student/bbdc_2023_AWI_data_develop_student.csv", sep=";", na_values=["NaN", "nan", "NA", np.nan, None])[["Datum", column]]
original_data.drop(axis=0, index=0, inplace=True)
original_data["Datum"] = pd.to_datetime(pd.to_datetime(original_data["Datum"], format="%d.%m.%Y").dt.strftime("%d.%m.%Y"))
original_data[column] = original_data[column].astype(float)
original_data.dropna(how="any", axis=0, inplace=True)
original_data = removeOutliers(original_data, column=column, window=8, threshold=0.2)

original_data[column] = original_data[column].shift(periods=timeshift, axis=0)


sylt_data = pd.read_csv("./research_data/List_Reede.csv", sep=",", na_values=["NaN", "nan", "NA", np.nan, None])
sylt_data["Datum"] = pd.to_datetime(pd.to_datetime(sylt_data["Date/Time"], format="%Y-%m-%dT%H:%M").dt.strftime("%d.%m.%Y"))
sylt_data.drop(axis=1, columns=["Date/Time"], inplace=True)
sylt_data.fillna(sylt_data.mean(), inplace=True)
for i in sylt_data.columns:
    if(i != "Datum"):
        sylt_data = removeOutliers(sylt_data, column=i, window=8, threshold=0.5)

all_data = pd.merge(original_data, sylt_data, on="Datum", how="outer")
all_data.iloc[:, 1:] = all_data.iloc[:, 1:].astype(float).interpolate(method="linear", axis=0)


all_data.drop("Datum", axis=1, inplace=True)
all_data = all_data.iloc[:-1000,:]


all_data.dropna(how="any", axis=0, inplace=True)

all_data[column].plot()
all_data["[NO3]- [µmol/l]"].plot()
plt.show()
exit()

all_data.to_csv("./test.csv", sep=",", index=True)

train_data = all_data.sample(frac=0.8, random_state=1)
test_data = all_data.drop(train_data.index)

train_features = train_data.copy()
test_features = test_data.copy()

train_labels = train_features.pop(column)
test_labels = test_features.pop(column)


normalizer = tf.keras.layers.Normalization(axis=-1)

normalizer.adapt(np.array(train_features))

regression_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(1)
])

regression_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.0001),
    loss='mean_squared_error')

history = regression_model.fit(
   train_features, train_labels,
   epochs=600,
   verbose=1,
   validation_split = 0.2)

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 5])
  plt.xlabel('Epoch')
  plt.ylabel('Error [{}]'.format(column))
  plt.legend()
  plt.grid(True)

test_predictions = regression_model.predict(test_features).flatten()



def plot_predicions_over_label(test_labels, test_predictions):
    a = plt.axes(aspect='equal')
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [{}]'.format(column))
    plt.ylabel('Predictions [{}]'.format(column))
    # lims = [0, 100]
    # plt.xlim(lims)
    # plt.ylim(lims)
    # _ = plt.plot(lims, lims)

plot_predicions_over_label(test_labels, test_predictions)

#plot_loss(history)



plt.show()


