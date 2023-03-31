import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense

from tensorflow import keras
from tensorflow.keras import layers
from removeOutliers import *


column = "Temperatur"

original_data = pd.read_csv("./task_student/bbdc_2023_AWI_data_develop_student.csv", sep=";", na_values=["NaN", "nan", "NA", np.nan, None])[["Datum", column]]
original_data.drop(axis=0, index=0, inplace=True)
original_data["Datum"] = pd.to_datetime(pd.to_datetime(original_data["Datum"], format="%d.%m.%Y").dt.strftime("%d.%m.%Y"))
original_data.dropna(how="any", axis=0, inplace=True)

sylt_data = pd.read_csv("./research_data/List_Reede.csv", sep=",", na_values=["NaN", "nan", "NA", np.nan, None])
sylt_data["Datum"] = pd.to_datetime(pd.to_datetime(sylt_data["Date/Time"], format="%Y-%m-%dT%H:%M").dt.strftime("%d.%m.%Y"))
sylt_data.drop(axis=1, columns=["Date/Time"], inplace=True)
sylt_data.fillna(sylt_data.mean(), inplace=True)

#sylt_data.plot(x="Datum")

all_data = pd.merge(original_data, sylt_data, on="Datum", how="outer")
all_data = all_data.sort_values(by="Datum").iloc[5145:11870, :]
all_data.iloc[:, 1:] = all_data.iloc[:, 1:].astype(float).interpolate(method="linear", axis=0)
all_data.drop("Datum", axis=1, inplace=True)


all_data=removeOutliers(data=all_data, column="Temperatur", window=8, threshold=0.2)

# all_data["Temperatur"].plot(linewidth=1)
# plt.show()



#print(original_data.info())
#print(original_data.head(10))

#print(sylt_data.info())
#print(sylt_data.head(10))

#print(all_data.info())
#print(all_data.head(100))

all_data.to_csv("./test.csv", sep=",", index=False)

#plt.show()

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
    # layers.Dense(16, activation='relu'),
    # layers.Dense(16, activation='relu'),
    layers.Dense(1)
])

regression_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.01),
    loss='mean_squared_error')

# generator = keras.preprocessing.sequence.TimeseriesGenerator(train_features, train_labels, length=16, batch_size=1)
# print(generator[0])
# history = regression_model.fit_generator(generator, steps_per_epoch=len(generator), epochs=20)

#history = regression_model.fit(
#    train_features, train_labels,
#    epochs=100,
#    verbose=0,
#    validation_split = 0.2)

#generator = keras.preprocessing.sequence.TimeseriesGenerator(train_features, train_labels, length=60, batch_size=64)

#history = regression_model.fit_generator(generator, steps_per_epoch=len(generator), epochs=20)

# hist = pd.DataFrame(history.history)
# hist['epoch'] = history.epoch
# print(hist.tail())

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 20])
  plt.xlabel('Epoch')
  plt.ylabel('Error [{}]'.format(column))
  plt.legend()
  plt.grid(True)

#test_predictions = regression_model.predict(test_features).flatten()

# a = plt.axes(aspect='equal')
# plt.scatter(test_labels, test_predictions)
# plt.xlabel('True Values [{}]'.format(column))
# plt.ylabel('Predictions [{}]'.format(column))
# lims = [0, 15]
# plt.xlim(lims)
# plt.ylim(lims)
# _ = plt.plot(lims, lims)

#plot test_predictions vs number of test data
#plt.plot(test_predictions, label="Predictions")

#plot_loss(history)

#plt.show()


