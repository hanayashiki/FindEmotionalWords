import tensorflow as tf
from settings_cnn import *
from get_data_cnn import *

sess = tf.Session()
xs = tf.placeholder(tf.float32, shape=[None, sequence_length, embedding_size, 1], name="xs")
ys = tf.placeholder(tf.float32, shape=[None, category_count], name="ys")

def add_filters_and_pools():
    pooled_outputs = list()
    for filter_size, filter_num in list(zip(filter_sizes, filter_nums)):
        filter_shape = [filter_size, embedding_size, 1, filter_num]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=1), name="W")
        b = tf.Variable(tf.constant(1., shape=[filter_num]), name="b")

        conv = tf.nn.conv2d(
            xs,
            W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv"
        )

        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        pooled = tf.nn.max_pool(
            h,
            ksize=[1, sequence_length - filter_size + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool"
        )
        pooled_outputs.append(pooled)

    num_filters_total = sum(filter_nums)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

    return (h_pool_flat, num_filters_total)

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

def build():
    h_pool_flat, h_pool_flat_width = add_filters_and_pools()

    (l1, W1, b1) = add_layer(h_pool_flat, h_pool_flat_width, category_count,
                             activation_function=None)

    sigmoid_output = tf.nn.sigmoid(l1, name="sigmoid_output")

    losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=l1, labels=ys)

    loss = tf.reduce_sum(losses)

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    one_zero = tf.cast(tf.greater(sigmoid_output, threshold), tf.float32)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(one_zero, ys), tf.float32))

    recall = tf.reduce_sum(tf.multiply(one_zero, ys)) / tf.reduce_sum(ys)

    print("model built")
    return (optimizer, loss, sigmoid_output, accuracy, recall, one_zero)

def train(loss, optimizer, accuracy, recall):
    print("start training")
    saver = tf.train.Saver()
    try:
        saver.restore(sess, model_path)
    except:
        print("nothing to load")

    for i in range(epoch):
        print(".", end='')
        x_data, y_data = data.getNextXY()
        if i % 1 == 0:
            current_losses = sess.run(loss, feed_dict={xs: x_data, ys: y_data})
            current_training_accuracy = sess.run(accuracy, feed_dict={xs: x_data, ys: y_data})
            current_test_accuracy = sess.run(accuracy, feed_dict={xs: x_test, ys: y_test})
            current_recall = sess.run(recall, feed_dict={xs: x_test, ys: y_test})
            print("Loop " + str(i) + " entropy = "+str(current_losses))
            print("accuracy on training set: " + str(current_training_accuracy))
            print("recall on test set: " + str(current_recall))
            print("accuracy on test set: " + str(current_test_accuracy))
            saver.save(sess, model_path)
        sess.run(optimizer, feed_dict={xs: x_data, ys: y_data})


def main():
    optimizer, losses, sigmoid_output, accuracy, recall, one_zero = build()
    sess.run(tf.global_variables_initializer())
    train(losses, optimizer, accuracy, recall)

if __name__ == '__main__':
    x_data_all, y_data_all = getDataXY()
    print(len(x_data_all))
    data = Data(x_data_all, y_data_all, batch_size)
    x_test, y_test = getDataXY(begin=15000, end=17000)
    print(len(x_test))
    main()