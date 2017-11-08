import tensorflow as tf
from test_data import *
from settings import *

x_data = getX()
y_data = getY()

def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def train():
    xs = tf.placeholder(tf.float32, shape=[None, d_of_word*2+1])
    # [0 : d_of_word-1] 是情感词的向量
    # [d_of_word: 2*d_of_word-1] 是某个词的向量
    # [2*d_of_word] 是距离
    ys = tf.placeholder(tf.float32, shape=[None, 1])
    # 1-0 是情感词还是不是情感词

    l1 = add_layer(xs, xs_width, l1_output_width, activation_function=tf.nn.sigmoid)
    l2 = add_layer(l1, l1_output_width, 1, activation_function=tf.nn.sigmoid)

    # -(1/m)*sum(sum((Y .* log(h_theta) + (1-Y) .* log(1- h_theta))))
    loss = tf.reduce_mean(
        tf.reduce_sum(
            tf.reduce_sum(
                - ys * tf.log(l2) - (1 - ys) * tf.log(1 - l2)
            )
        )
    )

    #elem1 = -ys * tf.log(l2)
    #elem2 = - (1 - ys) * tf.log(1 - l2)

    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(loop_round):
        sess.run(train_step, feed_dict={xs:x_data, ys: y_data})
        if i % 50 == 0:
            # to see the step improvement
            # print(sess.run(l2, feed_dict={xs:x_data}))
            # print(sess.run(elem1, feed_dict={xs: x_data, ys: y_data}))
            # print(sess.run(elem2, feed_dict={xs: x_data, ys: y_data}))
            print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))

if __name__ == '__main__':
    train()