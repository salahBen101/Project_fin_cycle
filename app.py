from flask import Flask, render_template, request
import requests
import json
import numpy as np
import matplotlib.pyplot as plt


app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello():
    return render_template('index.html')



@app.route('/', methods=['POST','GET'])
def index():
    if request.method == 'POST':
        symbol = request.form['symbol']

    def retrieve_stock_data(symbol, api_key):

        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={api_key}'
        response = requests.get(url)
        data = json.loads(response.text)['Time Series (Daily)']
        prices = []
        dates = []
        for date, values in data.items():
            prices.append(float(values['4. close']))
            dates.append(date)
        prices.reverse()
        dates.reverse()
        return prices, dates


    def prepare_data(prices):

        n = len(prices) - 1
        X = np.ones((n, 2))
        X[:, 1] = np.arange(n)
        y = np.array(prices[:-1]).reshape(-1, 1)
        X_norm = np.zeros(X.shape)
        X_norm[:, 0] = 1
        X_norm[:, 1] = (X[:, 1] - np.mean(X[:, 1])) / np.std(X[:, 1])
        return X_norm, y


    def train_model(X_norm, y, alpha, iterations):

        n = len(y)
        theta = np.random.randn(2, 1)
        for i in range(iterations):
            h = np.dot(X_norm, theta)
            error = h - y
            gradient = np.dot(X_norm.T, error) / n
            theta -= alpha * gradient
        return theta


    def predict_price(theta, X_norm):
        X_test = np.array([[1, X_norm[-1, 1] + 1]])
        X_test_norm = np.zeros(X_test.shape)
        X_test_norm[:, 0] = 1
        X_test_norm[:, 1] = (X_test[:, 1] - np.mean(X_norm[:, 1])) / np.std(X_norm[:, 1])
        y_pred = np.dot(X_test_norm, theta)
        return y_pred[0][0]



    def calculate_rmse(predictions, targets):
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        return rmse

    def plot_data(dates, prices, y_pred):
        plt.plot(dates[:-1], prices[:-1], label='Actual')
        plt.plot(dates[-1], y_pred, 'ro', label='Predicted')
        plt.xlabel('Date')
        plt.ylabel('Closing Price')
        plt.title('Stock Price Prediction')
        plt.legend()
        plt.show()

    api_key = '2FEAGIKJ516YM7G9'
    alpha = 0.003
    iterations = 4000

    prices, dates = retrieve_stock_data(symbol, api_key)
    X_norm, y = prepare_data(prices)
    theta = train_model(X_norm, y, alpha, iterations)
    y_pred = predict_price(theta, X_norm)
    y_for_if = y_pred
    y_actual = np.array(prices[1:])  
    rmse = calculate_rmse(y_pred, y_actual)
    #plot_data(dates, prices, y_pred)
    '''print(f"The predicted price for {dates[-1]} is {y_pred:.2f}")
    plot_data(dates, prices, y_pred)'''
    return render_template('result.html', y_for_if = y_for_if, dates=dates, prices=prices, y_pred=y_pred, rmse= rmse)


if __name__ =="__main__":
    app.run(debug=True, port=8000)
  




