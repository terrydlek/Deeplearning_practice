import spicy as sp
import pandas as pd
import numpy as np

from konlpy.tag import Kkma
from konlpy.tag import Hannanum
from konlpy.tag import Okt
from konlpy.tag import *

import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import os
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
from sklearn.tree import export_graphviz

import matplotlib.font_manager as fm
plt.rc('font', family='NanumGothic')

import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = False

import warnings
warnings.filterwarnings('ignore')

# 팍스넷 + 네이버 뉴스데이터 연결하기
news_df = pd.read_csv('팍스넷&네이버_뉴스타이틀.csv')

# print(news_df.shape)
# print(news_df)

title_list = news_df.뉴스제목.values.tolist()
# print(len(title_list))
# print(title_list)

title_text = ''
for each_line in title_list:
    title_text += each_line + '\n'

# print(title_text)

'''형태소 분석'''
tokens_ko = Okt().morphs(title_text)
# print(tokens_ko)

# nltk를 사용한 불용어 제거
import nltk
ko = nltk.Text(tokens_ko)
# print(ko)
# print(len(ko.tokens)) # 토큰 전체 개수
# print(len(set(ko.tokens))) # 토큰 unique 개수

stop_words = [word[0] for word in ko.vocab().most_common(200)]

tokens_ko = [each_word for each_word in tokens_ko if each_word not in stop_words]

ko = nltk.Text(tokens_ko)
# print("-------------------------------------------")
# print(len(ko.tokens)) # 토큰 전체 개수
# print(len(set(ko.tokens))) # 토큰 unique 개수
# print(ko.vocab().most_common(100))

# 그래프에서 한글 폰트 깨지는 문제에 대한 대처(전역 글꼴 설정)
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname='c:/Windows/Fonts/malgunsl.ttf').get_name()
rc('font', family=font_name)

import graphviz
plt.figure(figsize=(15, 6))
ko.plot(50)
plt.show()

from wordcloud import WordCloud, STOPWORDS
from PIL import Image

data = ko.vocab().most_common(300)
# print(data)

wordcloud = WordCloud(font_path='c:/Windows/Fonts/malgunsl.ttf', relative_scaling=0.2, background_color='white').generate_from_frequencies(dict(data))

plt.figure(figsize=(16, 8))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

# 형태소 분석을 위한 함수
def tokenizer(text):
    okt = Okt()
    return okt.morphs(text)

def data_preprocessing():
    title_list = news_df['뉴스제목'].tolist()
    price_list = news_df['주가변동'].tolist()
    from sklearn.model_selection import train_test_split

    # 데이터의 80%는 학습셋, 20%는 테스트셋
    title_train, title_test, price_train, price_test = train_test_split(title_list, price_list, test_size=0.2, random_state=0)
    return title_train, title_test, price_train, price_test

def learning(x_train, y_train, x_test, y_test):
    # 전처리가 끝난 데이터를 단어 사전으로 만들고
    # 리뷰별로 나오는 단어를 파악해서 수치화(벡터화)해서 학습
    # tfidf, 로지스틱 회귀 이용
    tfidf = TfidfVectorizer(lowercase=False, tokenizer=tokenizer)
    # 로지스틱
    logistic = LogisticRegression(C=2, penalty='l2', random_state=0) # C의 숫자가 너무 크면 과적합 (기본 1), penalty로 과적합 방지

    pipe = Pipeline([('vect', tfidf), ('clf', logistic)])

    # 학습
    pipe.fit(x_train, y_train)

    # 학습 정확도 측정
    y_fred = pipe.predict(x_test)
    print(accuracy_score(y_test, y_fred))

    # 학습한 모델을 저장
    with open('pipe.dat', 'wb') as fp: # 쓰기, 바탕화면에 저장됨
        pickle.dump(pipe, fp)
    print("저장 완료") # 학습된 모델 저장 완료

def using():
    # 객체를 복원, 저장된 모델 불러오기
    with open('pipe.dat', 'rb') as fp:
        pipe = pickle.load(fp)

    while True:
        text = input('뉴스 타이틀을 입력해 주세요 : ') # 인풋

        str = [text]

        # 예측 정확도
        r1 = np.max(pipe.predict_proba(str)*100) # 확률값을 구해서 *100..?

        # 예측 결과
        r2 = pipe.predict(str)[0]  # 긍정('1'), 부정('0')

        if r2 == "1":
            print("코스피 지수는 상승할 것으로 예상됩니다.")
        else:
            print("코스피 지수는 하락할 것으로 예상됩니다.")
        print('정확도 : %.3f' % r1)
        print('----------------------------------------')


# 학습 함수
def model_learning(): # 감성분석 모델 생성
    title_train, title_test, price_train, price_test = data_preprocessing()
    learning(title_train, price_train, title_test, price_test)


# 사용 함수
def model_using(): # 감성분석 모델 사용
    using()

model_learning()

model_using()
