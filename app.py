from flask import Flask, render_template, request 
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
from matplotlib.pylab import rcParams
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
from sklearn.model_selection import train_test_split as split
from sklearn.preprocessing import MinMaxScaler
import warnings
import itertools
warnings.filterwarnings("ignore")
from IPython import display
from matplotlib import pyplot
import os
import re
import seaborn as sns
import plotly.express as px
import warnings
from matplotlib.patches import Patch
import yfinance as yf
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
from math import floor
import threading
import ta
from queue import Queue

app = Flask(__name__)

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
    current_datetime = datetime.now()
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


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker']
    else:
        ticker = 'GOOGL'

        if ticker.isspace():
            render_template('errorpage.html')
            exit()
    
    try:
        period = '10y'
        df = get_data(ticker, period)
        company = yf.Ticker(ticker)
        info = company.info
        chart = create_candlestick_chart(df)
        company_name = info['longName']
        market_cap = info['marketCap']
        market_cap_formatted = format_market_cap(market_cap)
        short_description = info['longBusinessSummary']



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
        
        if(biLSTM_predicted_price > predicted_price):
            uprange = floor(biLSTM_predicted_price)
            downrange = floor(predicted_price)
        else:
            uprange = floor(predicted_price)
            downrange = floor(biLSTM_predicted_price)

         
        return render_template('index.html', ticker=ticker, chart_data=chart_data, predicted_price=round(predicted_price, 2), biLSTM_predicted_price=round(biLSTM_predicted_price, 2), uprange = uprange, downrange = downrange, bilstm_graph_html = bilstm_graph_html, ma100=ma100,ma200=ma200, graph_html=graph_html,high_value=high_value,close_value=close_value,open_value=open_value,high_status=increase_status_high,high_percent=percentage_change_high,Close_status=increase_status_Close,Close_percent=percentage_change_Close,Open_status=increase_status_Open,Open_percent=percentage_change_Open,company_name=company_name,market_cap=market_cap_formatted,short_description=short_description,chart=chart)
    except InvalidTickerError as e:
        return render_template('errorpage.html')
        if request.method == 'POST':
            ticker = request.form['ticker']
            index()



# Function to get today's high value of a stock
def get_today_high(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period='1d')
    if not data.empty:
        return data['High'].iloc[-1]
    return None

# Function to get today's close value of a stock
def get_today_close(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period='1d')
    if not data.empty:
        return data['Close'].iloc[-1]
    return None

# Function to get today's open value of a stock
def get_today_open(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period='1d')
    if not data.empty:
        return data['Open'].iloc[-1]
    return None

def get_percentage_change_high(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period='2d')
    if len(data) >= 2:
        yesterday_high = data['High'].iloc[-2]
        today_high = data['High'].iloc[-1]
        percentage_change = ((today_high - yesterday_high) / yesterday_high) * 100
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
        ticker = 'GOOGL'

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


if __name__ == '__main__':
    app.run(debug=False,threaded=True,use_reloader=False)
