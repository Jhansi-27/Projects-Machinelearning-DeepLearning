<h2>Design a DeepNeuralNetwork to forecast future sales of Rossmann superstore using its historical sales data</h2>

Rossmann operates over 3,000 drug stores in 7 European countries. Currently, Rossmann store managers are tasked with predicting their daily sales for up to six weeks in advance. Store sales are influenced by many factors, including promotions, competition, school and state holidays, seasonality, and locality. With thousands of individual managers predicting sales based on their unique circumstances, the accuracy of results can be quite varied.

<ul>
<li>Historical sales data for 1,115 Rossmann stores is available on kaggle. The task is to forecast the "Sales" column for the test set.</li>
<li>The dataset can be downloaded using the following link : (https://www.kaggle.com/c/rossmann-store-sales).</li>
</ul>

#### Files
```
train.csv - historical data including Sales
test.csv - historical data excluding Sales
sample_submission.csv - a sample submission file in the correct format
store.csv - supplemental information about the stores
```
#### Data fields
```
Id - an Id that represents a (Store, Date) duple within the test set
Store - a unique Id for each store
Sales - the turnover for any given day (this is what you are predicting)
Customers - the number of customers on a given day
Open - an indicator for whether the store was open: 0 = closed, 1 = open
StateHoliday - indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None
SchoolHoliday - indicates if the (Store, Date) was affected by the closure of public schools
StoreType - differentiates between 4 different store models: a, b, c, d
Assortment - describes an assortment level: a = basic, b = extra, c = extended
CompetitionDistance - distance in meters to the nearest competitor store
CompetitionOpenSince[Month/Year] - gives the approximate year and month of the time the nearest competitor was opened
Promo - indicates whether a store is running a promo on that day
Promo2 - Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating
Promo2Since[Year/Week] - describes the year and calendar week when the store started participating in Promo2
PromoInterval - describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store
```
#### This repository contains 3 pythonfiles.

<ol>
  
<li>RossmannStore_EDA_DataTransformation.ipynb, in which which EDA, data preprocessing and data transfromation has been performed on train and test datasets.</li>
  
<li>[lstm.py](https://github.com/Jhansi-27/University-Projects-Machinelearning-DeepLearning/blob/main/DeepLearning%20(Python)/Rossmann%20Store%20Sales%20Prediction(LSTM)/lstm.py
) This program takes processed data and trains LSTM model, to predict future sales 6 weeks in advance and saves the model</li>
<li>lstm_predict.py(https://github.com/Jhansi-27/University-Projects-Machinelearning-DeepLearning/blob/main/DeepLearning%20(Python)/Rossmann%20Store%20Sales%20Prediction(LSTM)/lstm_predict.py) This programs reads the saved model and makes predictions on test data for submission on kaggle</li>
</ol>
