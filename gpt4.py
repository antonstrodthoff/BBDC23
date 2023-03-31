#Here's a complete Python script that demonstrates how to read and prepare data from the "student_develop.csv" file using pandas and then train a regression neural network using Keras to predict the missing values. Note that you may need to adjust the model architecture and parameters to improve the prediction results.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping

# Read data from CSV file
data = pd.read_csv("./task_student/bbdc_2023_AWI_data_develop_student.csv", parse_dates=["Datum"], index_col="Datum", dayfirst=True)
print(data.head())
print(data.info())

# Fill missing values using interpolation
data.interpolate(method='time', inplace=True)

# Create a function to generate features and labels for the model
def create_dataset(data, look_back=30):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data.iloc[i:i + look_back].values)
        Y.append(data.iloc[i + look_back].values)
    return np.array(X), np.array(Y)

# Set a look-back period
look_back = 30

# Scale the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Create the dataset
X, Y = create_dataset(pd.DataFrame(data_scaled), look_back)

# Split the dataset into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create the regression neural network
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(look_back, X_train.shape[2])))
model.add(Dense(Y_train.shape[1]))
model.compile(optimizer='adam', loss='mse')

# Set early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train, Y_train, epochs=100, validation_split=0.1, batch_size=32, callbacks=[early_stopping])

# Function to predict a range of dates
def predict_range(start_date, end_date, look_back, data, model, scaler):
    start_index = data.index.get_loc(start_date) - look_back
    end_index = data.index.get_loc(end_date)
    input_data = data.iloc[start_index:end_index].values
    input_data = scaler.transform(input_data)
    input_data = input_data.reshape(1, input_data.shape[0], input_data.shape[1])
    predictions = model.predict(input_data)
    return scaler.inverse_transform(predictions)

# Predict values for the given date ranges
start_dates = ["2004-01-01", "2011-01-01"]
end_dates = ["2004-12-31", "2013-12-31"]

for start_date, end_date in zip(start_dates, end_dates):
    preds = predict_range(start_date, end_date, look_back, data, model, scaler)
    print(f"Predictions for {start_date} to {end_date}:")
    print(pd.DataFrame(preds, columns=data.columns))

#Make sure you have the required libraries installed. You can install them using pip:

#pip install pandas numpy scikit-learn tensorflow
#This script reads the "student_develop.csv" file, fills missing values using time-based interpolation, and prepares the data for the LSTM regression model. It then trains the model and predicts the values for the specified date ranges.
