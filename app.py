from flask import Flask, render_template, current_app, request, redirect, url_for, g, render_template_string
from markupsafe import Markup
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from pandas import DatetimeIndex
import json
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn import preprocessing
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sklearn.metrics as metrics
import math
from time import sleep
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import datetime
import warnings
from sklearn.model_selection import train_test_split as split
from sklearn.preprocessing import MinMaxScaler
import warnings
import itertools
warnings.filterwarnings("ignore")
from IPython import display
import os
import re
import seaborn as sns
import plotly.express as px
import warnings
import yfinance as yf
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
from math import floor
import threading
import ta
from queue import Queue
import hashlib
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
import sqlite3
from sqlite3 import Error
import time
from time import ctime, sleep
import datetime
import csv
import time
from yahoo_fin import stock_info as si
import matplotlib.pyplot as plt
import mpld3
import pandas_ta as pta
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_host=1)

class InvalidTickerError(Exception):
    pass
def get_data(ticker, period):
    try:
        df = yf.download(ticker, period=period)
        if df.empty:
            raise InvalidTickerError(f"Invalid ticker: {ticker}")
        return df
    except Exception as e:
        raise InvalidTickerError(f"Invalid ticker: {ticker}") from e

def Data_fetch_transform(data):
    # data = yf.download(ticker)
    data['Date'] = pd.to_datetime(data.index, infer_datetime_format=True)
    data_feature_selected = data.drop(axis=1, labels=["Open", "High", "Low", "Volume"])
    data_feature_selected['differenced_trasnformation_demand'] = data_feature_selected['Adj Close'].diff().values
    data_feature_selected['differenced_demand_filled'] = np.where(pd.isnull(data_feature_selected['differenced_trasnformation_demand']), data_feature_selected['Adj Close'], data_feature_selected['differenced_trasnformation_demand'])
    data_feature_selected['differenced_inv_transformation_demand'] = data_feature_selected['differenced_demand_filled'].cumsum()
    np.testing.assert_array_equal(data_feature_selected['Adj Close'].values, data_feature_selected['differenced_inv_transformation_demand'].values)
    current_datetime = datetime.datetime.now()
    # Extract the date portion
    current_date = current_datetime.date()
    # Convert the date to a string
    current_date_string = current_date.strftime('%Y-%m-%d')
    df1 = data_feature_selected.copy()
    # mask = (df1['Date'] > '2010-01-01') & (df1['Date'] <= current_date_string)
    y = df1['Adj Close']
    scaler=MinMaxScaler(feature_range=(0,1))
    y=scaler.fit_transform(np.array(y).reshape(-1,1))
    ##splitting dataset into train and test split
    training_size=int(len(y)*0.65)
    test_size=len(y)-training_size
    train_data,test_data=y[0:training_size,:],y[training_size:len(y),:1]
    def create_dataset(dataset, time_step=1):
	    dataX, dataY = [], []
	    for i in range(len(dataset)-time_step-1):
		    a = dataset[i:(i+time_step), 0]    
		    dataX.append(a)
		    dataY.append(dataset[i + time_step, 0])
	    return np.array(dataX), np.array(dataY)
    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
    return X_train,X_test,y_train,ytest,scaler

def biLSTM(ticker, result_queue):
    bilstm_model = load_model("bilstm_1000_epochs.h5")
    X_train,X_test,y_train,ytest,scaler = Data_fetch_transform(ticker)
    train_predict=bilstm_model.predict(X_train)
    test_predict=bilstm_model.predict(X_test)
    train_predict=scaler.inverse_transform(train_predict)
    test_predict=scaler.inverse_transform(test_predict)
    predictions = bilstm_model.predict(X_test)
    def evaluate_predictions(predictions, ytest, outliers):
        ratio = []
        differences = []
        for pred in range(len(ytest)):
            ratio.append((ytest[pred]/predictions[pred])-1)
            differences.append(abs(ytest[pred]- predictions[pred]))
            
            
        n_outliers = int(len(differences) * outliers)
        outliers = pd.Series(differences).astype(float).nlargest(n_outliers)
            
        return ratio, differences, outliers    
    ratio, differences, outliers = evaluate_predictions(predictions, ytest, 0.01)
    for index in outliers.index: 
        outliers[index] = predictions[index]

    def predict_next_day_closing_price(model, X_test, scaler):

        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)
        return predictions[-1][0]
    
    next_day = predict_next_day_closing_price(bilstm_model, X_test, scaler)

    # return next_day, predictions, ytest
    result_queue.put((next_day, predictions, ytest))

def format_market_cap(market_cap):
    if market_cap is None:
        return 'N/A'

    suffixes = ['', 'K', 'M', 'B', 'T']
    suffix_index = 0

    while market_cap >= 1000 and suffix_index < len(suffixes) - 1:
        market_cap /= 1000
        suffix_index += 1

    return f'{market_cap:.2f} {suffixes[suffix_index]}'


def create_candlestick_chart(data):
    chart = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        increasing_line_color='green',
        decreasing_line_color='red',
        name='Candlestick'
    )])

    chart.update_layout(
        # title='Stock Chart',
        xaxis=dict(
            rangeslider=dict(visible=False),
            type='date',
            showticklabels=False,
            gridcolor='gray'
        ),
        xaxis_title='Date',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(
            title='Price',
            color='white',
            gridcolor='gray' 
        ),
        font=dict(color='white'), 
        hovermode='x unified', 
        hoverdistance=100, 
        spikedistance=1000, 
        xaxis_showspikes=True,  
        yaxis_showspikes=True, 
        xaxis_spikemode='across',  
        yaxis_spikemode='across',  
        xaxis_spikecolor='white', 
        yaxis_spikecolor='white',
        height=580  
    )

    graph = chart.to_html(full_html=False)

    return graph

def load_csv(file_path):
    items = []
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        for row in reader:
            items.append(row)
    return items

# Index the items based on the search keyword
def index_items(items, keyword):
    indexed_items = []
    for item in items:
        if item['company_name'].lower().startswith(keyword.lower()):
            indexed_items.append(item['company_name'])
            if len(indexed_items) >= 6:
                break
    return indexed_items

# Get the ticker for a given company name
def get_ticker(items, company_name):
    for item in items:
        if item['company_name'].lower() == company_name.lower():
            return item['ticker']
    return None

@app.route('/search', methods=['GET'])
def search():
    keyword = request.args.get('keyword', '')
    # Load CSV file
    items = load_csv('valid_tickers.csv')
    # Index the items based on the search keyword
    indexed_items = index_items(items, keyword)
    return jsonify(indexed_items)

@app.route('/select', methods=['GET'])
def select():
    company_name = request.args.get('company_name', '')
    # Load CSV file
    items = load_csv('valid_tickers.csv')
    # Get the ticker for the selected company name
    ticker = get_ticker(items, company_name)
    return jsonify({'ticker': ticker})

def get_stock_data(ticker):
    df = si.get_data(ticker)
    df['date'] = df.index
    return df

def format_data(df):
    DATA_LEN = 300
    dates = df['date'][len(df)-DATA_LEN:len(df)].to_list()
    
    # Fill NaN values in 'close' column with the mean
    df['close'].fillna(df['close'].mean(), inplace=True)
    
    close_prices = df['close'][len(df)-DATA_LEN:len(df)].to_list()
    open_prices = df['open'][len(df)-DATA_LEN:len(df)].to_list()
    volumes = df['volume'][len(df)-DATA_LEN:len(df)].to_list()
    high_prices = df['high'][len(df)-DATA_LEN:len(df)].to_list()
    low_prices = df['low'][len(df)-DATA_LEN:len(df)].to_list()
    close_for_calc = df['close'][len(df)-DATA_LEN:len(df)]
    
    return dates, close_prices, open_prices, volumes, high_prices, low_prices, close_for_calc


def linear_regression_prediction(close_prices):
    dataset = np.array(close_prices)
    training = len(dataset)
    dataset = np.reshape(dataset, (dataset.shape[0], 1))

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0:int(training), :]

    x_train = []
    y_train = []
    prediction_days = 60

    for i in range(prediction_days, len(train_data)):
        x_train.append(train_data[i-prediction_days:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    reg = LinearRegression().fit(x_train, y_train)

    x_tomm = close_prices[len(close_prices) - prediction_days:]
    x_tomm = np.array(x_tomm)
    x_tomm = scaler.transform(x_tomm.reshape(-1, 1))

    prediction = reg.predict(x_tomm.reshape(1, -1))
    prediction = scaler.inverse_transform(prediction.reshape(-1, 1))

    return round(prediction[0][0], 2)
def future_predictions(close_prices, prediction_days, future_days):
    dataset = np.array(close_prices)
    training = len(dataset)
    dataset = np.reshape(dataset, (dataset.shape[0], 1))

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0:int(training), :]

    x_train = []
    y_train = []

    for i in range(prediction_days, len(train_data)):
        x_train.append(train_data[i-prediction_days:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    reg = LinearRegression().fit(x_train, y_train)

    predicted_prices = []
    tot_prices = list(close_prices)

    for i in range(future_days):
        x_prices = tot_prices[len(tot_prices) - prediction_days:]
        x_prices = np.array(x_prices, dtype=object)  # Specify dtype=object to handle ragged nested sequences
        x_prices = scaler.transform(x_prices.reshape(-1, 1))

        prediction = reg.predict(x_prices.reshape(1, -1))
        prediction = scaler.inverse_transform(prediction.reshape(-1, 1))

        tot_prices.append(prediction)
        predicted_prices.append(prediction)

    tot_prices = np.array(tot_prices, dtype=object)  # Specify dtype=object for the final array
    predicted_prices = np.array(predicted_prices, dtype=object)  # Specify dtype=object for the predicted prices array

    tot_prices = np.reshape(tot_prices, (tot_prices.shape[0]))
    predicted_prices = np.reshape(predicted_prices, (predicted_prices.shape[0]))

    return tot_prices, predicted_prices

@app.route('/', methods=['GET', 'POST'])
def index():
    ip_add = request.remote_addr
    timestamp = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5, minutes=30))).strftime("%a %b %d %H:%M:%S %Y")
    text_to_add = f"\n{timestamp}:  {ip_add}\n"
    file_path = 'templates/ip_logs.txt'
    try:
        with open(file_path, "a") as file:
            file.write(text_to_add)
    except FileNotFoundError:
        with open(file_path, "w") as file:
            file.write(text_to_add)   

    if request.method == 'POST':
        ticker = request.form.get('ticker')
    else:
        ticker = 'TSLA'

        if ticker.isspace():
            render_template('errorpage.html')
            exit()
    
    try:
        period = '10y'
        df = get_data(ticker, period)
        company = yf.Ticker(ticker)
        info = "NONE"
        chart = create_candlestick_chart(df)
        try:
            company_name = info['longName']
            market_cap = info['marketCap']
            market_cap_formatted = format_market_cap(market_cap)
            short_description = info['longBusinessSummary']
        except:
            company_name = "No data found"
            market_cap = 0
            market_cap_formatted = 0
            short_description = "No data found"
                   # === GRU ===
	    
        df_GRU = get_stock_data(ticker)
        dates, close_prices, open_prices, volumes, high_prices, low_prices, close_for_calc = format_data(df_GRU)
        prediction_GRU = linear_regression_prediction(close_prices)
        future_days = 10
        tot_prices, predicted_prices = future_predictions(close_prices, 60, future_days)
        prediction_list = []
        for i in range(future_days):
            prediction_list.append(predicted_prices[i])
	    
	           # === END ===

        closing_prices = df['Close']

        high_value = get_today_high(ticker)
        increase_status_high, percentage_change_high = get_percentage_change_high(ticker)
        close_value = get_today_close(ticker)
        increase_status_Close, percentage_change_Close = get_percentage_change_Close(ticker)
        open_value = get_today_open(ticker)
        increase_status_Open, percentage_change_Open = get_percentage_change_Open(ticker)

        chart_data = [{'x': str(date), 'y': price} for date, price in closing_prices.items()]
        ma100 = closing_prices.rolling(window=100).mean()
        ma100 = [{'x': str(date), 'y': price} for date, price in ma100.items() if not pd.isna(price)]
        ma200 = closing_prices.rolling(window=200).mean()
        ma200 = [{'x': str(date), 'y': price} for date, price in ma200.items() if not pd.isna(price)]

        data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
        data_testing = pd.DataFrame(df['Close'][int(len(df)*0.7):int(len(df))])

        scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_array = scaler.fit_transform(data_training)

        x_train = []
        y_train = []

        for i in range(100, data_training_array.shape[0]):
            x_train.append(data_training_array[i-100: i])
            y_train.append(data_training_array[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)

        # Load model
        model = load_model('lstm_1000_epochs.h5')

        result_queue = Queue()

        # Creating thread for model prediction
        bilstm_thread = threading.Thread(target=biLSTM, args=(df, result_queue))
        bilstm_thread.start()

        

        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

        input_data = scaler.fit_transform(final_df)

        x_test = []
        y_test = []

        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i-100:i])
            y_test.append(input_data[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)

        y_predict = model.predict(x_test)

        scaler = scaler.scale_

        scale_factor = 1/scaler[0]
        y_predict = y_predict * scale_factor
        y_test = y_test * scale_factor

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df.index[int(len(df)*0.70):], y=y_test, name='Original Price'))
        fig2.add_trace(go.Scatter(x=df.index[int(len(df)*0.70):], y=y_predict[:, 0], name='Predict'))
        fig2.update_layout(
                        xaxis_title='Date',
                        yaxis_title="Price (standardized)",
                        height=500 ,
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        title=dict(font=dict(color='white')),
                        xaxis=dict(showticklabels=False,gridcolor='gray'),
                        yaxis=dict(gridcolor='gray')
                        )
        graph_html = fig2.to_html(full_html=False)

        last_100_days = data_testing[-100:].values
        scaler = MinMaxScaler()
        last_100_days_scaled = scaler.fit_transform(last_100_days)

        predicted_prices = []

        # Wait for the thread to finish
        bilstm_thread.join()
        
        biLSTM_predicted_price, predictions_biLSTM, biLSTM_ytest = result_queue.get()

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df.index[int(len(df)*0.70):], y=biLSTM_ytest, name='Original Price'))
        fig3.add_trace(go.Scatter(x=df.index[int(len(df)*0.70):], y=predictions_biLSTM[:, 0], name='Predict'))
        fig3.update_layout(
                        xaxis_title='Date',
                        yaxis_title="Price (standardized)",
                        height=500 ,
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        title=dict(font=dict(color='white')),
                        xaxis=dict(showticklabels=False,gridcolor='gray'),
                        yaxis=dict(gridcolor='gray')
                        )
        bilstm_graph_html = fig3.to_html(full_html=False)


        for i in range(1):
            X_test = np.array([last_100_days_scaled])
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            predicted_price = model.predict(X_test)
            predicted_prices.append(predicted_price)
            last_100_days_scaled = np.append(last_100_days_scaled, predicted_price)
            last_100_days_scaled = np.delete(last_100_days_scaled, 0)

        predicted_prices = np.array(predicted_prices)
        predicted_prices = predicted_prices.reshape(predicted_prices.shape[0], predicted_prices.shape[2])
        predicted_prices = scaler.inverse_transform(predicted_prices)
        predicted_price = predicted_prices[0][0]
        
        # if(biLSTM_predicted_price > predicted_price):
        #     uprange = floor(biLSTM_predicted_price)
        #     downrange = floor(predicted_price)
        # else:
        #     uprange = floor(predicted_price)
        #     downrange = floor(biLSTM_predicted_price)
        if floor(prediction_GRU) > 1:    
            uprange=floor(prediction_GRU)+1
            downrange=floor(prediction_GRU)-1
        else:
            gr = str(prediction_GRU)
            dk  = len(gr[2:])
            lk = "."            
            print(dk)
            for i in range(dk):
                if (i != (dk-1)): 
                    lk += '0'
                else:
                    lk +='1'
            buffet = float(lk)
            uprange= prediction_GRU + buffet
            downrange= prediction_GRU - buffet
         
        return render_template('index.html', ticker=ticker, chart_data=chart_data, predicted_price=round(predicted_price, 2), biLSTM_predicted_price=round(biLSTM_predicted_price, 2), uprange = uprange, downrange = downrange, bilstm_graph_html = bilstm_graph_html, ma100=ma100,ma200=ma200, graph_html=graph_html,high_value=high_value,close_value=close_value,open_value=open_value,high_status=increase_status_high,high_percent=percentage_change_high,Close_status=increase_status_Close,Close_percent=percentage_change_Close,Open_status=increase_status_Open,Open_percent=percentage_change_Open,company_name=company_name,market_cap=market_cap_formatted,short_description=short_description,chart=chart,prediction_GRU=prediction_GRU,prediction_list=prediction_list)
    except InvalidTickerError as e:
        return render_template('errorpage.html')
        if request.method == 'POST':
            ticker = request.form['ticker']
            index()


@app.route('/track')
def track():
    with open('templates/ip_logs.txt', 'r') as file:
        text_content = file.read()
    rendered_content = Markup(text_content)
    return render_template('rendered_text.html', content=rendered_content)


# Function to get today's high value of a stock
def get_today_high(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period='1d')
    if not data.empty:
        return round(data['High'].iloc[-1],3)
    return None

# Function to get today's close value of a stock
def get_today_close(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period='1d')
    if not data.empty:
        return round(data['Close'].iloc[-1], 3)
    return None

# Function to get today's open value of a stock
def get_today_open(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period='1d')
    if not data.empty:
        return round(data['Open'].iloc[-1],3)
    return None

def get_percentage_change_high(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period='2d')
    if len(data) >= 2:
        yesterday_high = data['High'].iloc[-2]
        today_high = data['High'].iloc[-1]
        percentage_change = ((today_high - yesterday_high) / yesterday_high) * 100
        percentage_change = round(percentage_change,4)
        if percentage_change > 0:
            increase_status = 'Increased'
        elif percentage_change < 0:
            increase_status = 'Decreased'
        else:
            increase_status = 'No change'
        return increase_status, percentage_change
    return None, None

def get_percentage_change_Close(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period='2d')
    if len(data) >= 2:
        yesterday_high = data['Close'].iloc[-2]
        today_high = data['Close'].iloc[-1]
        percentage_change = ((today_high - yesterday_high) / yesterday_high) * 100
        percentage_change = round(percentage_change,4)
        if percentage_change > 0:
            increase_status = 'Increased'
        elif percentage_change < 0:
            increase_status = 'Decreased'
        else:
            increase_status = 'No change'
        return increase_status, percentage_change
    return None, None

def get_percentage_change_Open(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period='2d')
    if len(data) >= 2:
        yesterday_high = data['Open'].iloc[-2]
        today_high = data['Open'].iloc[-1]
        percentage_change = ((today_high - yesterday_high) / yesterday_high) * 100
        percentage_change = round(percentage_change,4)
        if percentage_change > 0:
            increase_status = 'Increased'
        elif percentage_change < 0:
            increase_status = 'Decreased'
        else:
            increase_status = 'No change'
        return increase_status, percentage_change
    return None, None

@app.route('/faq')
def faq():
    return render_template('pages-faq.html')

@app.route('/contact')
def contact():
    return render_template('pages-contact.html')

@app.route('/about')
def about():
    return render_template('pages-about.html')

@app.route('/overview')
def overview():
    return render_template('pages-overview.html')

@app.route('/register')
def register():
    return render_template('pages-register.html')

@app.route('/news')
def news():
    return render_template('news.html')

@app.route('/gchat')
def gchat():
    return render_template('gchat.html')

@app.route('/login')
def login():
    return render_template('pages-login.html')

def fetch_stock_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="5y")

    stock_data = data.reset_index()  # Reset index to convert Date into a column
    stock_data.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)

    return stock_data

def create_graph(x, y, indicator, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, name='Close Price'))
    if indicator is not None:
        fig.add_trace(go.Scatter(x=x, y=indicator))
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      font=dict(color='white'),
                      title=dict(font=dict(color='white')),
                      xaxis=dict(showticklabels=False,gridcolor='gray'),
                      yaxis=dict(gridcolor='gray'))
    return fig

@app.route('/indicators',methods=['GET', 'POST'])
def indicators():
    if request.method == 'POST':
        ticker = request.form['ticker']
    else:
        ticker = 'TSLA'

        if ticker.isspace():
            render_template('errorpage.html')
            exit()
    
    try:
        stock_data = fetch_stock_data(ticker)
        # Calculate indicators
        sma_indicator = ta.trend.sma_indicator(stock_data['close'], window=20)
        ema_indicator = ta.trend.ema_indicator(stock_data['close'], window=20)
        rsi_indicator = ta.momentum.rsi(stock_data['close'], window=14)
        wma_indicator = ta.trend.WMAIndicator(stock_data['close'], window=20)
        vwap_indicator = ta.volume.VolumeWeightedAveragePrice(stock_data['high'], stock_data['low'], stock_data['close'], stock_data['volume'])
        stochastic_indicator = ta.momentum.StochasticOscillator(stock_data['high'], stock_data['low'], stock_data['close'], window=14, smooth_window=3)
        atr_indicator = ta.volatility.AverageTrueRange(stock_data['high'], stock_data['low'], stock_data['close'], window=14)
        cmf_indicator = ta.volume.ChaikinMoneyFlowIndicator(stock_data['high'], stock_data['low'], stock_data['close'], stock_data['volume'], window=20)


        # Calculate Bollinger Bands
        bb_indicator = ta.volatility.BollingerBands(close=stock_data['close'], window=20, window_dev=2)
        bb_upper = bb_indicator.bollinger_hband()  # Upper Bollinger Band
        bb_middle = bb_indicator.bollinger_mavg()  # Middle Bollinger Band
        bb_lower = bb_indicator.bollinger_lband()  # Lower Bollinger Band

        # Create separate graphs for each indicator
        sma_graph = go.Figure()
        sma_graph.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['close'], name='Close Price'))
        sma_graph.add_trace(go.Scatter(x=stock_data['date'], y=sma_indicator, name='SMA (20)'))
        sma_graph.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      font=dict(color='white'),
                      title=dict(font=dict(color='white')),
                      xaxis=dict(showticklabels=False,gridcolor='gray'),
                      yaxis=dict(gridcolor='gray'))

        ema_graph = go.Figure()
        ema_graph.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['close'], name='Close Price'))
        ema_graph.add_trace(go.Scatter(x=stock_data['date'], y=ema_indicator, name='EMA (20)'))
        ema_graph.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      font=dict(color='white'),
                      title=dict(font=dict(color='white')),
                      xaxis=dict(showticklabels=False,gridcolor='gray'),
                      yaxis=dict(gridcolor='gray'))

        rsi_graph = go.Figure()
        rsi_graph.add_trace(go.Scatter(x=stock_data['date'], y=rsi_indicator, name='RSI (14)'))
        rsi_graph.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      font=dict(color='white'),
                      title=dict(font=dict(color='white')),
                      xaxis=dict(showticklabels=False,gridcolor='gray'),
                      yaxis=dict(gridcolor='gray'))

        bb_graph = go.Figure()
        bb_graph.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['close'], name='Close Price'))
        bb_graph.add_trace(go.Scatter(x=stock_data['date'], y=bb_upper, name='BB Upper'))
        bb_graph.add_trace(go.Scatter(x=stock_data['date'], y=bb_lower, name='BB Lower'))
        bb_graph.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      font=dict(color='white'),
                      title=dict(font=dict(color='white')),
                      xaxis=dict(showticklabels=False,gridcolor='gray'),
                      yaxis=dict(gridcolor='gray'))


        roc_graph = create_graph(stock_data['date'], stock_data['close'], None, 'ROC Chart')

        wma_graph = create_graph(stock_data['date'], stock_data['close'], wma_indicator.wma(), 'WMA (20) Chart')
   
        vwap_graph = create_graph(stock_data['date'], stock_data['close'], vwap_indicator.vwap, 'VWAP Chart')
    
        stochastic_graph = create_graph(stock_data['date'], stock_data['close'], stochastic_indicator.stoch(), 'Stochastic (14, 3) Chart')
    
        atr_graph = create_graph(stock_data['date'], stock_data['close'], atr_indicator.average_true_range(), 'ATR (14) Chart')
    
        cmf_graph = create_graph(stock_data['date'], stock_data['close'], cmf_indicator.chaikin_money_flow(), 'CMF (20) Chart')

        # Convert the Plotly graphs to HTML
        sma_graph_html = sma_graph.to_html(full_html=False)
        ema_graph_html = ema_graph.to_html(full_html=False)
        rsi_graph_html = rsi_graph.to_html(full_html=False)
        bb_graph_html = bb_graph.to_html(full_html=False)
        roc_graph_html = roc_graph.to_html(full_html=False)
        wma_graph_html = wma_graph.to_html(full_html=False)
        vwap_graph_html = vwap_graph.to_html(full_html=False)
        stochastic_graph_html = stochastic_graph.to_html(full_html=False)
        atr_graph_html = atr_graph.to_html(full_html=False)
        cmf_graph_html = cmf_graph.to_html(full_html=False)


        return render_template('indicators.html',ticker=ticker, sma_graph_html=sma_graph_html, ema_graph_html=ema_graph_html, rsi_graph_html=rsi_graph_html, bb_graph_html=bb_graph_html, roc_graph_html=roc_graph_html, wma_graph_html=wma_graph_html, vwap_graph_html=vwap_graph_html, stochastic_graph_html=stochastic_graph_html, atr_graph_html=atr_graph_html, cmf_graph_html=cmf_graph_html)
    
    except InvalidTickerError as e:
        return render_template('errorpage.html')
    
# Define the buy_stock function for turtle  
def buy_stock_turtle(real_movement,signal,initial_money,max_buy,max_sell,df):
            starting_money = initial_money
            states_sell = []
            states_buy = []
            current_inventory = 0
            def buy(i, initial_money, current_inventory):
                shares = initial_money // real_movement[i]
                if shares < 1:
                    print('day %d: total balances %f, not enough money to buy a unit price %f'% (i, initial_money, real_movement[i]))
                else:
                    if shares > max_buy:
                        buy_units = max_buy
                    else:
                        buy_units = shares
                    initial_money -= buy_units * real_movement[i]
                    current_inventory += buy_units
                    print('day %d: buy %d units at price %f, total balance %f'% (i, buy_units, buy_units * real_movement[i], initial_money))
                    states_buy.append(0)
                return initial_money, current_inventory
            for i in range(real_movement.shape[0] - int(0.025 * len(df))):
                state = signal[i]
                if state == 1:
                    initial_money, current_inventory = buy(i, initial_money, current_inventory)
                    states_buy.append(i)
                elif state == -1:
                    if current_inventory == 0:
                        print('day %d: cannot sell anything, inventory 0' % (i))
                    else:
                        if current_inventory > max_sell:
                            sell_units = max_sell
                        else:
                            sell_units = current_inventory
                        current_inventory -= sell_units
                        total_sell = sell_units * real_movement[i]
                        initial_money += total_sell
                        try:
                            invest = ((real_movement[i] - real_movement[states_buy[-1]])/ real_movement[states_buy[-1]]) * 100
                        except:
                            invest = 0
                        print('day %d, sell %d units at price %f, investment %f %%, total balance %f,'% (i, sell_units, total_sell, invest, initial_money))
                    states_sell.append(i)
            
            invest = ((initial_money - starting_money) / starting_money) * 100
            total_gains = initial_money - starting_money
            return states_buy, states_sell, total_gains, invest
    
def plot_stock_data(close, states_buy, states_sell, total_gains, invest):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=close.index, y=close, mode='lines', name='Close', line=dict(color='red', width=2)))
    fig.add_trace(go.Scatter(x=close.index[states_buy], y=close[states_buy], mode='markers',
                            marker=dict(symbol='triangle-up', size=10, color='magenta'), name='Buying Signal'))
    fig.add_trace(go.Scatter(x=close.index[states_sell], y=close[states_sell], mode='markers',
                            marker=dict(symbol='triangle-down', size=10, color='black'), name='Selling Signal'))
    fig.update_layout(title='Total Gains: {:.2f}, Total Investment: {:.2f}%'.format(total_gains, invest))
    fig.update_layout(showlegend=True,plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        title=dict(font=dict(color='white')),
                        xaxis=dict(showticklabels=False,gridcolor='gray'),
                        yaxis=dict(gridcolor='gray'))

    return fig

@app.route('/trade',methods=['GET', 'POST'])
def trade():
    if request.method == 'POST':
        ticker = request.form['ticker']
        years = int(request.form['years'])
        initial_money = float(request.form['initial_money'])
        max_buy = int(request.form['max_buy'])
        max_sell = int(request.form['max_sell'])
    else:
        ticker = 'AAPL'
        years = 4
        initial_money=100000
        max_buy=100
        max_sell=100

        if ticker.isspace():
            render_template('errorpage.html')
            exit()
    
    try:
        start_date = pd.Timestamp.now().date() - pd.DateOffset(years=years)
        end_date = pd.Timestamp.now().date()
        df = yf.download(ticker, start=start_date, end=end_date)

        # trading statergy using turtle
        count = int(np.ceil(len(df) * 0.1))
        signals_turtle = pd.DataFrame(index=df.index)
        signals_turtle['signal'] = 0.0
        signals_turtle['trend'] = df['Close']
        signals_turtle['RollingMax'] = (signals_turtle.trend.shift(1).rolling(count).max())
        signals_turtle['RollingMin'] = (signals_turtle.trend.shift(1).rolling(count).min())
        signals_turtle.loc[signals_turtle['RollingMax'] < signals_turtle.trend, 'signal'] = -1
        signals_turtle.loc[signals_turtle['RollingMin'] > signals_turtle.trend, 'signal'] = 1

        states_buy_turtle, states_sell_turtle, total_gains_turtle, invest_turtle = buy_stock_turtle(df.Close, signals_turtle['signal'],initial_money,max_buy,max_sell,df)
        close = df['Close']
        fig_turtle = plot_stock_data(close, states_buy_turtle, states_sell_turtle, total_gains_turtle, invest_turtle)
        graph_json_turtle = fig_turtle.to_json()

        return render_template('trade.html',graph_json_turtle=graph_json_turtle,ticker=ticker, years=years,initial_money=initial_money, max_buy=max_buy, max_sell=max_sell)



    except InvalidTickerError as e:
        return render_template('errorpage.html')

################Blockchain##################
def create_connection():
    conn = None
    try:
        if not os.path.exists("logs.db"):
            conn = sqlite3.connect('logs.db')
        else:
            conn = sqlite3.connect('logs.db')
        return conn
    except Error as e:
        print(e)
    return conn


def generate_key_pair():
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    public_key = private_key.public_key()
    return private_key, public_key


def save_private_key(private_key):
    pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    with open("private_key.pem", "wb") as f:
        f.write(pem)


def save_public_key(public_key):
    pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    with open("public_key.pem", "wb") as f:
        f.write(pem)


def load_private_key():
    if os.path.isfile("private_key.pem"):
        with open("private_key.pem", "rb") as f:
            pem = f.read()
            return serialization.load_pem_private_key(pem, password=None, backend=default_backend())
    else:
        return None


def load_public_key():
    if os.path.isfile("public_key.pem"):
        with open("public_key.pem", "rb") as f:
            pem = f.read()
            return serialization.load_pem_public_key(pem, backend=default_backend())
    else:
        return None


private_key = load_private_key()
public_key = load_public_key()
if private_key is None or public_key is None:
    private_key, public_key = generate_key_pair()
    save_private_key(private_key)
    save_public_key(public_key)


@app.before_request
def before_request():
    g.private_key = private_key
    g.public_key = public_key


class Block:
    def __init__(self, timestamp, data, previous_hash):
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.current_hash = self.hash_block()
        self.encrypted_data = None
        self.decrypted_data = None

    def encrypt_data(self, public_key):
        data_bytes = self.data.encode()
        encrypted_data = public_key.encrypt(
            data_bytes,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        self.encrypted_data = encrypted_data.hex()

    def decrypt_data(self, private_key):
        try:
            encrypted_data = bytes.fromhex(self.encrypted_data)
            decrypted_data = private_key.decrypt(
                encrypted_data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            self.decrypted_data = decrypted_data.decode()
        except Exception as e:
            print(e)

    def hash_block(self):
        input_string = f"{self.timestamp}{self.data}{self.previous_hash}"
        input_bytes = input_string.encode()
        hash_bytes = hashlib.sha256(input_bytes)
        hash_hex = hash_bytes.hexdigest()
        return hash_hex


class Blockchain:
    def __init__(self):
        self.chain = self.load_blocks_from_db()

    def load_blocks_from_db(self):
        conn = create_connection()
        with conn:
            c = conn.cursor()
            try:
                c.execute("SELECT * FROM blocks")
                rows = c.fetchall()

                blocks = []
                for row in rows:
                    timestamp, encrypted_data, previous_hash, current_hash = row
                    new_block = Block(timestamp, "", previous_hash)
                    new_block.encrypted_data = encrypted_data
                    new_block.current_hash = current_hash
                    blocks.append(new_block)

                return blocks
            except sqlite3.OperationalError:
                return []

    def add_block(self, data, public_key):
        timestamp = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5, minutes=30))).strftime("%a %b %d %H:%M:%S %Y")
        previous_hash = self.chain[-1].current_hash if self.chain else ""
        new_block = Block(timestamp, data, previous_hash)
        new_block.encrypt_data(public_key)
        self.chain.append(new_block)
        self.save_to_db(new_block)

    def save_to_db(self, block):
        conn = create_connection()
        with conn:
            create_table(conn)
            c = conn.cursor()
            c.execute("INSERT INTO blocks (timestamp, encrypted_data, previous_hash, current_hash) VALUES (?, ?, ?, ?)",
                      (block.timestamp, block.encrypted_data, block.previous_hash, block.current_hash))
            conn.commit()


def create_table(conn):
    try:
        c = conn.cursor()
        create_table_sql = '''
            CREATE TABLE IF NOT EXISTS blocks
            ("timestamp" TEXT PRIMARY KEY, encrypted_data TEXT, previous_hash TEXT, current_hash TEXT)
        '''
        c.execute(create_table_sql)
    except Error as e:
        print(e)


blockchain = Blockchain()


##############################MODELFUNCTION FOR LOGS#################################################################

def models(ticker):

    period = '10y'
    df = get_data(ticker, period)

    def biLSTM(data_frame):
        bilstm_model = load_model("bilstm_1000_epochs.h5")
        X_train,X_test,y_train,ytest,scaler = Data_fetch_transform(data_frame)
        train_predict=bilstm_model.predict(X_train)
        test_predict=bilstm_model.predict(X_test)
        train_predict=scaler.inverse_transform(train_predict)
        test_predict=scaler.inverse_transform(test_predict)
        predictions = bilstm_model.predict(X_test)
        def evaluate_predictions(predictions, ytest, outliers):
            ratio = []
            differences = []
            for pred in range(len(ytest)):
                ratio.append((ytest[pred]/predictions[pred])-1)
                differences.append(abs(ytest[pred]- predictions[pred]))
                
                
            n_outliers = int(len(differences) * outliers)
            outliers = pd.Series(differences).astype(float).nlargest(n_outliers)
                
            return ratio, differences, outliers    
        ratio, differences, outliers = evaluate_predictions(predictions, ytest, 0.01)
        for index in outliers.index: 
            outliers[index] = predictions[index]

        def predict_next_day_closing_price(model, X_test, scaler):

            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            predictions = model.predict(X_test)
            predictions = scaler.inverse_transform(predictions)
            return predictions[-1][0]
        
        next_day = predict_next_day_closing_price(bilstm_model, X_test, scaler)

        return round(next_day, 3)

    
    def LSTM(df):
        closing_prices = df['Close']
        data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.7)])
        data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.7):int(len(df))])

        scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_array = scaler.fit_transform(data_training)

        x_train = []
        y_train = []

        for i in range(100, data_training_array.shape[0]):
            x_train.append(data_training_array[i - 100: i])
            y_train.append(data_training_array[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)

        # Load model
        model = load_model('lstm_1000_epochs.h5')

        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

        input_data = scaler.fit_transform(final_df)

        x_test = []
        y_test = []

        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i - 100:i])
            y_test.append(input_data[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)

        y_predict = model.predict(x_test)

        scaler = scaler.scale_

        scale_factor = 1 / scaler[0]
        y_predict = y_predict * scale_factor
        y_test = y_test * scale_factor

        last_100_days = data_testing[-100:].values
        scaler = MinMaxScaler()
        last_100_days_scaled = scaler.fit_transform(last_100_days)

        predicted_prices = []

        for i in range(1):
            X_test = np.array([last_100_days_scaled])
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            predicted_price = model.predict(X_test)
            predicted_prices.append(predicted_price)
            last_100_days_scaled = np.append(last_100_days_scaled, predicted_price)
            last_100_days_scaled = np.delete(last_100_days_scaled, 0)

        predicted_prices = np.array(predicted_prices)
        predicted_prices = predicted_prices.reshape(predicted_prices.shape[0], predicted_prices.shape[2])
        predicted_prices = scaler.inverse_transform(predicted_prices)
        predicted_price = predicted_prices[0][0]

        return round(predicted_price, 3)

    lstm_price = LSTM(df)
    bilstm_price = biLSTM(df)

    # if (bilstm_price > lstm_price):
    #     uprange = floor(bilstm_price
    #     downrange = floor(lstm_price)
    # else:
    #     uprange = floor(lstm_price)
    #     downrange = floor(bilstm_price)    
    #############gru##################
    df_GRU = get_stock_data(ticker)
    dates, close_prices, open_prices, volumes, high_prices, low_prices, close_for_calc = format_data(df_GRU)
    prediction_GRU = linear_regression_prediction(close_prices)

    uprange = floor(prediction_GRU)+1
    downrange = floor(prediction_GRU)-1

    return lstm_price, bilstm_price, uprange, downrange, prediction_GRU


#######################################################################################################################

def round_off(value):
    formatted_value = "{:.2f}".format(value)
    return float(formatted_value)

################################Only for TSLAe####################################

def generate_block_every_second():
    scheduled_time_pre = "01:20 AM"
    scheduled_time_close = "01:35 AM"
    while True:
        public_key = load_public_key()
        if public_key is not None:
            current_time = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5, minutes=30))).strftime("%I:%M %p")
            if current_time == scheduled_time_close:
                symbol = "TSLA"
                data = f"Open: {round(get_today_open(symbol),3)}, High: {round(get_today_high(symbol),3)}, Close: {round(get_today_close(symbol),3)}"
                blockchain.add_block(data, public_key)
            if current_time == scheduled_time_pre:
                symbol = "TSLA"
                lstm, bilstm, uprange, downrange, gru = models(symbol)
                lstm_r = round_off(lstm)
                bilstm_r = round_off(bilstm)
                gru_r = round_off(gru)
                data = f"LSTM: {lstm_r}, BiLSTM: {bilstm_r}, GRU: {gru_r}, Range: {uprange} - {downrange}"
                blockchain.add_block(data, public_key)
        sleep(60)

block_add = threading.Thread(target=generate_block_every_second)
block_add.start()
###########################################Only for TSLAe############################



@app.route("/logs")
def logs():
    private_key = g.private_key

    blocks = []
    for block in blockchain.chain:
        if block.timestamp == 0:
            continue
        if block.decrypted_data is None:
            block.decrypt_data(private_key)

        block_dict = {
            "timestamp": block.timestamp,
            "data": block.decrypted_data,
            "previous_hash": block.previous_hash,
            "current_hash": block.current_hash
        }
        blocks.append(block_dict)

    return render_template("logs.html", blocks=blocks)



if __name__ == '__main__':
    app.run(debug=False,threaded=True,use_reloader=False)
