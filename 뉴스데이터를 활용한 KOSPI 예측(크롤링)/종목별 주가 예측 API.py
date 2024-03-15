'''종목별 주가 예측 API'''
'''
야후 파이낸스의 get_data_yahoo 함수를 이용하여 종목별 다음날 종가를 예측하는 API 제작
 - 종목코드, 시작날짜를 입력하면 해당 종목의 다음 날 종가 예측
'''
# 1. 클래스 생성
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

class StockPrediction:
    def StockData(self, code, date):
        yf.pdr_override()
        df = pdr.get_data_yahoo(f'{code}.KS', start=f'{date}')
        df = df[['Open', 'High', 'Low', 'Volume', 'Close']]
        return df

    def MinMaxScaler(self, data):
        numerator = data - np.min(data, 0)
        dominator = np.max(data, 0) - np.min(data, 0)
        return numerator / (dominator + 1e-7)

    def DataSet(self, code, date):
        df = self.StockData(code, date)
        df_x = self.MinMaxScaler(df)
        df_y = df_x[['Close']]

        x = df_x.values.tolist()
        y = df_y.values.tolist()

        data_x = []
        data_y = []
        window_size = 10

        for i in range(len(y) - window_size):
            x2 = x[i:i + window_size]
            y2 = y[i + window_size]
            data_x.append(x2)
            data_y.append(y2)

        # 훈련용 데이터셋 70%
        train_size = int(len(data_x) * 0.7)
        train_x = np.array(data_x[:train_size])
        train_y = np.array(data_y[:train_size])

        # 테스트용 데이터셋 30%
        test_x = np.array(data_x[train_size:])
        test_y = np.array(data_y[train_size:])
        return df, df_y, train_x, train_y, test_x, test_y

    def LSTMModel(self, code, date):
        df, df_y, train_x, train_y, test_x, test_y = self.DataSet(code, date)

        model=  Sequential()
        model.add(LSTM(units=10, activation='relu', return_sequences=True, input_shape=(10, 5)))
        model.add(Dropout(0, 1))
        model.add(LSTM(units=10, activation='relu'))
        model.add(Dropout(0, 1))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(train_x, train_y, epochs=100, batch_size=30)
        pred_y = model.predict(test_x)

        return df, df_y, test_y, pred_y

    def PredictionResult(self, code, date):
        df, df_y, test_y, pred_y = self.LSTMModel(code, date)

        plt.figure()
        plt.plot(test_y, color='red', label='real stock price')
        plt.plot(pred_y, color='blue', label='predicted stock price')
        plt.title('Real Stock Price vs Predicted Stock Price')
        plt.xlabel('time')
        plt.ylabel('stock price')
        plt.legend(loc='best')
        plt.show()

        print("다음 날 예측 종가 : ", df.Close[-1]*pred_y[-1]/df_y.Close[-1])
