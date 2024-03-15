from pandas_datareader import data as pdr

import yfinance as yf
yf.pdr_override()
import pandas as pd
import datetime
import matplotlib.pyplot as plt

pd.set_option('display.precision', 4)

# 코스피지수 크롤링
start_date = datetime.datetime(2023, 8, 1)
end_date= datetime.datetime(2024, 2, 28)
# ^KS11: 코스피
df_null = yf.download('^KS11', start_date, end_date)
# 결측치 제거
df = df_null.dropna()

# 새로운 칼럼 생성
# (Price : 당일 대비 다음날 주가가 상승했으면 1, 하락했으면 0으로 표시)
df["Price"] = 0
for i in range(0, 120):
    if df['Close'][i] < df['Close'][i + 1]:
        df['Price'][i] = 1
    else:
        df['Price'][i] = 0
print(df)

# kospi_주가데이터.xlsx 이름으로 파일 저장
df.to_csv('kospi_주가데이터.csv')
