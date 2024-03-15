import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

'''1. 야후 파이낸스로 주가 구하기'''
# 데이터를 받아오는 함수
def StockData(code, date):
    yf.pdr_override()
    df = pdr.get_data_yahoo(f'{code}.KS', start=f'{date}')
    df = df[['Open', 'High', 'Low', 'Volume', 'Close']]
    return df

df = StockData('005930', '2019-01-01') # 삼성전자 주가 받아오기
print(df.head())

'''2. MinMaxScaler 함수를 적용'''
# 계산 시간 단축을 위해 OHLVC 데이터를 0과 1 사이의 값으로 변환
def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7) # 0으로 나누기 에러 방지


df_x = MinMaxScaler(df)
df_y = df_x[['Close']]
x = df_x.values.tolist()
y = df_y.values.tolist()

print(df_x.info())
print("-----------------------")
print(df_y.info())
print("-----------------------")
print(x[-1])
print("-----------------------")
print(y[-1])

'''데이터셋 준비하기'''
data_x = []
data_y = []
# 이전 10일 동안 OHLVC 데이터로 다음 날 종가 예측
window_size = 10
for i in range(len(y) - window_size):
    x2 = x[i:i + window_size]
    y2 = y[i + window_size]
    data_x.append(x2)
    data_y.append(y2)
print("이전 10일 동안 OHLVC : ", x2, "\n 다음날 종가 : ", y2)

# 훈련용 데이터셋 70%
train_size = int(len(data_x) * 0.7)
train_x = np.array(data_x[0:train_size])
train_y = np.array(data_y[0:train_size])

# 테스트용 데이터셋 30%
test_size = len(data_x) - train_size
test_x = np.array(data_x[train_size:len(data_x)])
test_y = np.array(data_y[train_size:len(data_y)])

'''모델 생성하기'''
model = Sequential()
model.add(LSTM(units=10, activation='relu', return_sequences=True, input_shape=(window_size, 5))) # (10, 5) 입력 형태를 가지는 LSTM층
model.add(Dropout(0, 1)) # 입력값의 10%를 0으로 치환하여 과적합 방지
model.add(LSTM(units=10, activation='relu'))
model.add(Dropout(0, 1))
model.add(Dense(units=1))
print(model.summary())

model.compile(optimizer='adam', loss='mean_squared_error') # 최적화 도구: adam, 손실함수: MSE
model.fit(train_x, train_y, epochs=100, batch_size=30) # epochs: 전체 데이터셋 학습 횟수, batch_size: 한 번에 제공되는 훈련 데이터
pred_y = model.predict(test_x)

plt.figure()
plt.plot(test_y, color='red', label='real stock price')
plt.plot(pred_y, color='blue', label='predicted stock price')
plt.title('Real Stock Price vs Predicted Stock Price')
plt.xlabel('time')
plt.ylabel('stock price')
plt.legend(loc='best')
plt.show()
