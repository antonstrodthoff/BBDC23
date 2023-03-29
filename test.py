import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

xx = np.array([[1,2,3,4,5,6,7,8,9,10], [2,4,6,8,10,12,14,16,18,20]]).T
yy = np.array([1.5,3,4.5,6,7.5,9,10.5,12,13.5,15])

data_gen = TimeseriesGenerator(xx, yy, length=2, batch_size=1)

model = Sequential()
model.add(Dense(32, input_shape=(2,2,), activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

history = model.fit_generator(data_gen, epochs=100)
#history = model.fit(xx, yy, epochs=100)

print(data_gen[0])
print(history.history)

test = model.predict(np.array([[145, 146], [290, 292]]).T)
print(test)

