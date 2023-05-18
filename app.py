from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from pandas import DatetimeIndex
import json
import plotly.graph_objects as go

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker']
    else:
        ticker = 'AAPL' 

    period = '10y'
    df = yf.download(ticker, period=period)

    closing_prices = df['Close']
    high_value = get_today_high(ticker)
    close_value = get_today_close(ticker)
    open_value = get_today_open(ticker)
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
    model = load_model('keras_model.h5')

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
                    yaxis_title='Price',
                    # width=1000,
                    height=500, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    graph_html = fig2.to_html(full_html=False)


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

    return render_template('index.html', ticker=ticker, chart_data=chart_data, predicted_price=predicted_price, ma100=ma100,ma200=ma200, graph_html=graph_html,high_value=high_value,close_value=close_value,open_value=open_value)

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

# render my other html pages
# @app.route('/pages-blank')
# def pages_blank():
#     return render_template('pages-blank.html')

# @app.route('/pages-contact')
# def pages_contact():
#     return render_template('pages-contact.html')

# @app.route('/pages-faq')
# def pages_faq():
#     return render_template('pages-faq.html')

# @app.route('/pages-login')
# def pages_login():
#     return render_template('pages-login.html')

# @app.route('/pages-register')
# def pages_register():
#     return render_template('pages-register.html')

# @app.route('/user-profile')
# def user_profile():
#     return render_template('user-profile.html')

if __name__ == '__main__':
    app.run(debug=True)
