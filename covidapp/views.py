from django.shortcuts import render
from django.http import HttpResponse
import requests
import json 

from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.http import FileResponse
import sweetviz
from pandas_profiling import ProfileReport
import plotly
import plotly.express as px

import math
import numpy as np 
import pandas as pd
import yfinance as yf
from plotly.offline import plot 
import seaborn as sns
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import matplotlib  
from PIL import Image
from io import BytesIO 
matplotlib.use('Agg')

import io
import urllib, base64

@api_view()
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
    model.fit(train_x, train_y, epochs=0, batch_size=1)

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

    figu = plt.figure(figsize=(15,12))
    plt.plot(train_data['Close'])
    plt.plot(test_data['Predictions'])
    plt.plot(test_data['Close'])

    fig = plt.gcf()
    #convert graph into dtring buffer and then we convert 64 bit code into image
    buf = io.BytesIO()
    fig.savefig(buf,format='jpg')
    buf.seek(0)
    stri = base64.b64encode(buf.read())
    return render(request , 'index.html' , {'graph' : stri})


@api_view()
def index2(request):
    data_stocks = yf.Tickers('MSFT AAPL GOOG TSLA FB AMZN')
    data_fetch = data_stocks.history(period='1d', interval='1m')
    raw_data = pd.DataFrame(data=data_fetch)
    raw_data = raw_data.drop(['Dividends','Stock Splits'], axis=1)
    raw_data.to_csv(r'D__\Internship - LSCG\Stocks_Data\All Stocks.csv')
    data = raw_data.drop(['High','Low','Open','Volume'], axis=1)
    data.to_csv(r'D__\Internship - LSCG\Stocks_Data\Closing Stocks.csv',header=False)

    data = pd.read_csv(r'D__\Internship - LSCG\Stocks_Data\Closing Stocks.csv')
    data.columns = ['Datetime','AAPL', 'AMZN','FB', 'GOOG', 'MSFT', 'TSLA']
   
    data.to_csv(r'D__\Internship - LSCG\Stocks_Data\Closing Stocks.csv',index=None)
    data_plot = pd.read_csv(r'D__\Internship - LSCG\Stocks_Data\Closing Stocks.csv')
    data_plot.plot(x='Datetime',y=['AAPL','AMZN','FB','GOOG','MSFT','TSLA'],style='-')
    data_plot.head()
    plt.title('Relative price change')
    plt.legend(loc='upper left', fontsize=12)
    plt.tight_layout()
    plt.grid(True)
    fig = px.line(data_plot,x=data_plot['Datetime'],y=['AMZN','AAPL','MSFT','TSLA','FB','GOOG'],title='Interactive Stocks')
    fig.update_layout(template='plotly_dark')
    plt_div = plot(fig, output_type='div')
    return render(request , 'index.html' , {'graph' : plt_div})
    






    


