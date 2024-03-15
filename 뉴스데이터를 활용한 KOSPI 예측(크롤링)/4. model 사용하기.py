import pickle
import nltk
import numpy as np
from konlpy.tag import Okt


def tokenizer(text):
    okt = Okt()
    return okt.morphs(text)


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


# 사용 함수
def model_using(): # 감성분석 모델 사용
    using()


model_using()
