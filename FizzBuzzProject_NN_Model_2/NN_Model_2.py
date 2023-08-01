import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import coverage

#cov = coverage.Coverage()
#cov.start()

def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])

def fizz_buzz_encode(i):
    if   i % 15 == 0: return np.array([0, 0, 0, 1])
    elif i % 5  == 0: return np.array([0, 0, 1, 0])
    elif i % 3  == 0: return np.array([0, 1, 0, 0])
    else:             return np.array([1, 0, 0, 0])

NUM_DIGITS = 16
trX = np.array([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
trY = np.array([fizz_buzz_encode(i)          for i in range(101, 2 ** NUM_DIGITS)])
NUM_HIDDEN = 100

X = tf.placeholder("float", [None, NUM_DIGITS])
Y = tf.placeholder("float", [None, 4])

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

w_h = init_weights([NUM_DIGITS, NUM_HIDDEN])
w_o = init_weights([NUM_HIDDEN, 4])

def model(X, w_h, w_o):
    h = tf.nn.relu(tf.matmul(X, w_h))
    return tf.matmul(h, w_o)

py_x = model(X, w_h, w_o)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
predict_op = tf.argmax(py_x, 1)
def fizz_buzz(i, prediction):
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]

NUM_EPOCHS = 10000
BATCH_SIZE = 128

sess = tf.Session()

with sess.as_default():
    tf.global_variables_initializer().run()

    for epoch in range(NUM_EPOCHS):
        p = np.random.permutation(range(len(trX)))
        trX, trY = trX[p], trY[p]

        for start in range(0, len(trX), BATCH_SIZE):
            end = start + BATCH_SIZE
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})

        accuracy = np.mean(np.argmax(trY, axis=1) ==
                           sess.run(predict_op, feed_dict={X: trX, Y: trY}))
        print(epoch, accuracy)

numbers = np.arange(1, 101)
teX = np.transpose(binary_encode(numbers, NUM_DIGITS))

teY = sess.run(predict_op, feed_dict={X: teX})
output = np.vectorize(fizz_buzz)(numbers, teY)

actuals = [fizz_buzz(i, fizz_buzz_encode(i).argmax()) for i in numbers]

for i, (predicted, actual) in enumerate(zip(output, actuals)):
    if predicted != actual:
        print("{0} {1} {2}".format(i+1, predicted, actual))

#cov.stop()
#cov.save()

#cov.html_report()
