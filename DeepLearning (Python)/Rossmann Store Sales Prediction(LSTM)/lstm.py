from sklearn import preprocessing
import keras
from keras.layers import LSTM, Dense, Activation, Dropout
from keras.models import Sequential
import pandas as pd
import numpy as np
from keras import backend as K
from keras.callbacks import History

history = History()
train = pd.read_csv(
    "/data_processing/data/TrainData_Kaggle.csv")
stores = train['Store'].unique()
columns = ['Store', 'DayOfWeek', 'Sales', 'Customers', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday', 'StoreType',
           'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2',
           'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval', 'day', 'month', 'year']


def root_mean_squared_percentage_error(y_true, y_pred):
    if y_true != 0:
        return K.sqrt(K.mean(K.square(1.0 - float(y_pred) / float(y_true))))
    else:
        return 0.0


big_y = train.Sales
train = train.drop(train.columns[0], axis=1)
train = train.drop(['Sales', 'Customers'], axis=1)
big_x = np.array(train)
big_x = np.reshape(big_x, (big_x.shape[0], 1, 20))

model = Sequential()
model.add(LSTM(units=10, return_sequences=True, input_shape=(big_x.shape[1], 20)))
model.add(Dropout(0.2))

# Adding a second LSTM layer and Dropout layer
model.add(LSTM(units=10, return_sequences=True))
model.add(Dropout(0.2))

# Adding a third LSTM layer and Dropout layer
model.add(LSTM(units=10, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=10, return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(units=1))
callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)
model.compile(loss='mean_squared_error', optimizer='adam', metrics='accuracy')

history = model.fit(big_x, big_y, epochs=1000, batch_size=50, verbose=2, validation_split=0.2, callbacks=[callback])
print(history.history)
model.save(
    "/models/kaggle_lstm_model_3")
