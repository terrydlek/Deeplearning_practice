"""
수집날짜 리스트 생성
다음날 주가가 하락한 경우 : date_0
다음날 주가가 상승한 경우 : date_1
"""
import yfinance as yf
yf.pdr_override()
import pandas as pd
import datetime
import matplotlib.pyplot as plt

price_date = pd.read_csv('kospi_주가데이터.csv')
df_0 = price_date[price_date['Price']==0]['Date']

date_0 = []
for i in range(0, len(df_0)):
    date_0.append(str(df_0.tolist()[i])[:10].replace('-', ''))
# print(date_0)

df_1 = price_date[price_date['Price']==1]['Date']

date_1 = []
for i in range(0, len(df_1)):
    date_1.append(str(df_1.tolist()[i])[:10].replace('-', ''))
# print(date_1)


# 팟스넷 뉴스 타이틀 크롤링
# 수집 대상: 팍스넷 '많이 본 뉴스' -> '증권' 분야 뉴스타이틀
# 수집 날짜: 23.8.1 ~ 24.2.28
import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib import parse

result_list = []
error_cnt = 0

def paxnet_news_title(dates):
    base_url = 'http://www.paxnet.co.kr/news/much?newsSetId=4667&currentPageNo={}&genDate={}&objId=N4667'
    headers = {
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)'
    }

    for date in dates:
        for page in range(1, 3):
            url = base_url.format(page, date)
            res = requests.get(url, headers=headers)
            if res.status_code == 200:
                soup = BeautifulSoup(res.text, features='lxml')
                title_list = soup.select('ul.thumb-list li')
                print(title_list)
                for title in title_list:
                    try:
                        news_title = title.select_one('dl.text > dt').text.strip()
                        result_list.append([news_title])
                    except:
                        error_cnt += 1


paxnet_news_title(date_0)

title_df_0 = pd.DataFrame(result_list, columns=['뉴스제목'])
title_df_0['주가변동'] = 0
title_df_0.head()

result_list = []
error_count = 0
paxnet_news_title(date_1)

title_df_1 = pd.DataFrame(result_list, columns=['뉴스제목'])
title_df_1['주가변동'] = 1
title_df_1.head()

title_df = pd.concat([title_df_0, title_df_1])
title_df.to_csv('팍스넷_뉴스타이틀.csv', index=False, encoding='utf-8')


result_list = []
error_cnt = 0


def naver_news_title(dates):
    base_url = 'https://news.naver.com/main/ranking/popularDay.nhn?rankingType=popular_day&sectionId=101&date={}'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36'
    }

    for date in dates:
        url = base_url.format(date)
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, features='html.parser')
            title_list = soup.select('.rankingnews_list')
            # print(title_list)
            # print(len(title_list))

            for title in title_list:
                try:
                    news_title = title.select_one('.list_title').text.strip()
                    result_list.append([news_title])
                    print(date, news_title)
                except:
                    error_cnt += 1
naver_news_title(date_0)

title_df_2 = pd.DataFrame(result_list, columns=['뉴스제목'])
title_df_2['주가변동'] = 0
title_df_2.head()

result = []
error_cnt = 0
naver_news_title(date_1)
title_df_3 = pd.DataFrame(result_list, columns=['뉴스제목'])
title_df_3['주가변동'] = 1
title_df_3.head()

title_df_2 = pd.concat([title_df_2, title_df_3])
title_df_2.to_csv('네이버_뉴스타이틀.csv', index=False, encoding='utf-8')

print(title_df_2)

all_title = pd.concat([title_df, title_df_2])

# '팍스넷 & 네이버_뉴스타이틀' 이름으로 파일 저장
all_title.to_csv('팍스넷&네이버_뉴스타이틀.csv')

