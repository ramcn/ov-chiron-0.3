import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

num_classes = 4
dropout = 0.75

weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    'out': tf.Variable(tf.random_normal([1024, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def modelcnn(x):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


def modelrnn(graph, input_tensor):
    #cell = tf.nn.rnn_cell.GRUCell(10)
    cell = tf.nn.rnn_cell.LSTMCell(10)
    with graph.as_default():
        ((fw_outputs, bw_outputs), (fw_state, bw_state)) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell,
            cell_bw=cell,
            inputs=input_tensor,
            sequence_length=[10] * 32,
            dtype=tf.float32,
            swap_memory=True,
            scope=None)
        outputs = tf.concat((fw_outputs, bw_outputs), 2)
        mean = tf.reduce_mean(outputs, axis=1)
        dense = tf.layers.dense(mean, 5, activation=None)
        return dense

if __name__ == '__main__':
    np.random.seed(0)
    features = np.random.randn(64, 10, 30)
    labels = np.eye(5)[np.random.randint(0, 5, (64,))]

    graph1 = tf.Graph()
    with graph1.as_default():
        tf.set_random_seed(0)
        batch_size_ph = tf.placeholder(tf.int64, name='batch_size_ph')
        features_data_ph = tf.placeholder(tf.float32, [None, None, 30], 'features_data_ph')
        labels_data_ph = tf.placeholder(tf.int32, [None, 5], 'labels_data_ph')
        # Dataset
        dataset = tf.data.Dataset.from_tensor_slices((features_data_ph, labels_data_ph))
        dataset = dataset.batch(batch_size_ph)
        iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
        dataset_init_op = iterator.make_initializer(dataset, name='dataset_init')
        input_tensor, labels_tensor = iterator.get_next()

        # Model
        logits = modelrnn(graph1, input_tensor)
        saver = tf.train.Saver()
        global_step=tf.get_variable('global_step',trainable=False,shape=(),dtype = tf.int32,initializer = tf.zeros_initializer())

        with tf.Session(graph=graph1) as sess:
            # Initialize variables
            tf.global_variables_initializer().run(session=sess)
            batch = 0
            sess.run( dataset_init_op, feed_dict={ features_data_ph: features, labels_data_ph: labels, batch_size_ph: 32 })
            value = sess.run(logits)
            batch += 1
            frozen = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["dense/BiasAdd"])
            graph_io.write_graph(frozen, './', 'train_graph.pb', as_text=False)

