from django.shortcuts import render
from django.http import HttpResponse
import requests
import json 

import math
import numpy as np 
import pandas as pd
import yfinance as yf
import seaborn as sns
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle


def index(request):

    back_period = 30 

    stock = yf.Ticker("AAPL")
    hist = stock.history(period='max', interval='1d')
    data = hist.filter(items=['Close'])
    dataset1=data.values

    stock = yf.Ticker("CIPLA.NS")
    hist = stock.history(period='max', interval='1d')
    data = hist.filter(items=['Close'])
    dataset2=data.values

    dataset=np.concatenate((dataset1,dataset2))

    scaler=MinMaxScaler()
    scaled_dataset=scaler.fit_transform(dataset)
    scaled_dataset1= scaled_dataset[0:dataset1.shape[0]]
    scaled_dataset2= scaled_dataset[dataset1.shape[0]:]


    data_x=[]
    data_y=[]

    for i in range(back_period,scaled_dataset1.shape[0]):
        data_x.append(scaled_dataset1[i-back_period:i,:])
        data_y.append(scaled_dataset1[i,0])

    for i in range(back_period,scaled_dataset2.shape[0]):
        data_x.append(scaled_dataset2[i-back_period:i,:])
        data_y.append(scaled_dataset2[i,0])


    data_x=np.array(data_x)
    data_y=np.array(data_y)

    data_y =data_y.reshape(-1,1)

    training_size = math.ceil(data_x.shape[0]*0.7)

    data_x,data_y= shuffle(data_x,data_y,random_state=1)

    train_x= data_x[0:training_size,:]
    train_y= data_y[0:training_size,:]

    model = Sequential()
    model.add(LSTM(40, input_shape=(train_x.shape[1],1),return_sequences=True))
    model.add(LSTM(25))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(train_x, train_y, epochs=3, batch_size=1)

    test_x=data_x[training_size:,:]
    test_y=data_y[training_size:,:]
    test_predict= model.predict(test_x)
    test_predict= scaler.inverse_transform(test_predict)
    test_y= scaler.inverse_transform(test_y)

    error= np.sqrt(np.mean(((test_predict- test_y)**2)))


    stock = yf.Ticker("GOOGL")
    hist = stock.history(period='max', interval='1d')

    vol_data= hist['Volume']
    data = hist.filter(items=['Close'])
    dataset=data.values

    scaled_dataset= scaler.transform(dataset)

    training_size = math.ceil(dataset.shape[0]*0.7)
    train_data= data[0:training_size+back_period]
    test_data= data[training_size+back_period:]

    test_scaled_data = scaled_dataset[training_size:]
    test_x=[]

    for i in range(back_period,test_scaled_data.shape[0]):
        test_x.append(test_scaled_data[i-back_period:i,:])

    test_x=np.array(test_x)
    test_predict= model.predict(test_x)
    # test_predict = np.reshape(test_predict,(test_predict.shape[0],test_predict.shape[1]))

    test_predict= scaler.inverse_transform(test_predict)


    test_data['Predictions'] = test_predict

    fig = plt.figure(figsize=(15,12))
    plt.plot(train_data['Close'])
    plt.plot(test_data['Predictions'])
    plt.plot(test_data['Close'])

    context = {'response' : plt}
    return render(request , 'index.html' , context)