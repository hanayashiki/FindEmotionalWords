import tensorflow as tf
from get_data import *
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
    return (outputs, Weights, biases)

def train():

    xs = tf.placeholder(tf.float32, shape=[None, xs_width])
    # [0 : d_of_word-1] 是情感词的向量
    # [d_of_word: 2*d_of_word-1] 是某个词的向量
    # [2*d_of_word] 是距离
    ys = tf.placeholder(tf.float32, shape=[None, 1])
    # 1-0 是情感词还是不是情感词

    (l1, W1, b1) = add_layer(xs, xs_width, l1_output_width, activation_function=tf.nn.sigmoid)

    (l2, W2, b2) = add_layer(l1, l1_output_width, l2_output_width, activation_function=tf.nn.sigmoid)

    (l3, W3, b3) = add_layer(l2, l2_output_width, 1, activation_function=None)

    # -(1/m)*sum(sum((Y .* log(h_theta) + (1-Y) .* log(1- h_theta))))
    loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=ys, logits=l3))

    #elem1 = -ys * tf.log(l2)
    #elem2 = - (1 - ys) * tf.log(1 - l2)

    train_step = tf.train.AdamOptimizer(rate).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    threshold = tf.placeholder(tf.float32)
    train_output = tf.cast(tf.greater(l3, threshold), tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(train_output, ys) , tf.float32))
    precision = tf.div(tf.reduce_sum(train_output * ys) , tf.reduce_sum(train_output))
    recall = tf.div(tf.reduce_sum(train_output * ys), tf.reduce_sum(ys))

    saver2 = tf.train.Saver()
    try:
        saver2.restore(sess, model_path)
    except:
        print("nothing to load")
    for i in range(loop_round):
        sess.run(train_step, feed_dict={xs:x_data, ys: y_data, threshold: default_threshold})

        if i % 50 == 0:
            # to see the step improvement
            # print(sess.run(l2, feed_dict={xs:x_data}))
            # print(sess.run(elem1, feed_dict={xs: x_data, ys: y_data}))
            # print(sess.run(elem2, feed_dict={xs: x_data, ys: y_data}))
            
            print(sess.run(loss, feed_dict={xs: x_data, ys: y_data, threshold: default_threshold}))
            print("loop %d accuracy: " % (i))
            print(sess.run(accuracy, feed_dict={xs: x_data, ys: y_data, threshold: default_threshold}))
            print("loop %d precision: " % (i))
            print(sess.run(precision, feed_dict={xs: x_data, ys: y_data, threshold: default_threshold}))
            print("loop %d recall: " % (i))
            print(sess.run(recall, feed_dict={xs: x_data, ys: y_data, threshold: default_threshold}))


    saver = tf.train.Saver()
    save_path = saver.save(sess, model_path)
    print("W1:")
    print(sess.run(W1))
    print("b1:")
    print(sess.run(b1))
    print("W2:")
    print(sess.run(W2))
    print("b2:")
    print(sess.run(b2))
    print("W3:")
    print(sess.run(W3))
    print("b3:")
    print(sess.run(b3))

    test_x = getTestX()
    test_y = getTestY()

    for t in threshold_list:
        print("with threshold = " + str(t))
        print("accuracy on test set: ")
        print(sess.run(accuracy, feed_dict={xs: test_x, ys: test_y, threshold:t}))
        print("precision: ")
        print(sess.run(precision, feed_dict={xs: test_x , ys: test_y, threshold:t}))
        print("recall: ")
        print(sess.run(recall, feed_dict={xs: test_x, ys: test_y, threshold:t}))


if __name__ == '__main__':
    train()
