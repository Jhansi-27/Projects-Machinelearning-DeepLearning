from tensorflow import keras
import pandas as pd
import numpy as np

test_data = pd.read_csv(
    "/data_processing/data/TestData_Kaggle.csv",
    low_memory=False)
test_data_without_id = test_data.drop('Id', axis=1)
store_id = test_data.Id
test_data_without_id = test_data_without_id.drop(test_data_without_id.columns[0], axis=1)
test_data_without_id = np.array(test_data_without_id)
test_data_without_id = np.reshape(test_data_without_id, (test_data_without_id.shape[0], 1, 20))
model = keras.models.load_model('/models/kaggle_lstm_model_3')
prediction = model.predict(test_data_without_id)

np.savetxt("/predictions/kaggle_result_2.csv", prediction, delimiter=",")
