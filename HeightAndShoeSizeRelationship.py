import tensorflow as tf

height = [170, 180, 175, 160]
shoe_size = [260, 270, 265, 255]

# shoe_size = height * a + b

a = tf.Variable(0.1)
b = tf.Variable(0.2)
def lossfunc():
    predict = height * a + b
    return tf.square(260 - predict)

# a, b를 경사하강법으로 구하는걸 도와주는 고마운 친구
# adam은 gradient를 알아서 smart하게 바꿔줌
opt = tf.keras.optimizers.Adam(learning_rate=0.1)
# var_list는 경사하강법으로 update할 weight variable 목록
for _ in range(300):
    opt.minimize(lossfunc, var_list=[a, b])
    print(a.numpy(), b.numpy())
