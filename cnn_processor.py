import tensorflow as tf
from settings_cnn import *
import test_cnn

optimizer, losses, sigmoid_output, accuracy, one_zero = test_cnn.build()

def process(x_data):
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, model_path)

    y = sess.run(one_zero, feed_dict={test_cnn.xs: x_data})
    to_ret = []
    for y_line in y:
        emotion_obj_list = []
        for idx in range(len(y_line)):
            if y_line[idx] == 1:
                emotion_obj_list.append((idx // sequence_length + 1, idx % sequence_length + 1))
        to_ret.append(emotion_obj_list)
        print(emotion_obj_list)

    #print(y)
    #print(to_ret)
    return to_ret

if __name__ == '__main__':
    process()

