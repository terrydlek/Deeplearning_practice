'''TOPIC : 대학원 붙을 확률 딥러닝'''
import tensorflow as tf
import pandas as pd
import numpy as np

data = pd.read_csv('gpascore.csv')
# print(data.isnull().sum())
# 빈칸 채우기
# data.fillna(0, inplace=True)
data = data.dropna()

ydata = data['admit'].values
xdata = []

for i, rows in data.iterrows():
    li = []
    for j in range(1, len(rows)):
        li.append(rows[j])
    xdata.append(li)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu'), # 노드의 개수
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# optimizer은 최적화기계 ex) 경사하강법 등
# binary_crossentropu는 결과가 0과 1 사이의 분류 / 확률 문제에서 씀
# metrics는 모델을 평가할 때 어떤 요소로 평가할건가?
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 학습시키기 (x데이터, y데이터)
# x 데이터에는 학습 데이터, y데이터는 정답 데이터
# earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss')
model.fit(np.array(xdata), np.array(ydata), epochs=1000)

# 예측
predict = model.predict(np.array([[750, 3.70, 3], [400, 2.2, 1]]))
print(predict)
