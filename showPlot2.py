import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import tensorflow as tf
import os


max_step = 1000
rnn_unit = 30
input_size = 6
output_size = 1
lr = 0.0006

stockNo = 'main'
stockCsvPath = '{}-all.csv'.format(stockNo)
modelPath = 'model/{}/'.format(stockNo)
if not os.path.exists(modelPath):
    os.mkdir(modelPath)


def get_train_data(batch_size=60, time_step=20, train_begin=0, train_end=3400):
    batch_index = []
    data_train = data[train_begin:train_end]

    normalized_train_data = (data_train - np.mean(data_train, axis=0)) / np.std(data_train, axis=0)

    train_x, train_y = [], []
    for i in range(len(normalized_train_data) - time_step):
        if i % batch_size == 0:
            batch_index.append(i)
        x = normalized_train_data[i:i + time_step, :input_size]
        y = normalized_train_data[i:i + time_step, input_size, np.newaxis]

        train_x.append(x.tolist())
        train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data) - time_step))

    return batch_index, train_x, train_y


def get_test_data(time_step=20, test_begin=3401):
    data_test = data[test_begin:]
    mean = np.mean(data_test, axis=0)
    std = np.std(data_test, axis=0)
    normalized_test_data = (data_test - mean) / std
    size = (len(normalized_test_data) + time_step - 1) // time_step
    test_x, test_y = [], []
    for i in range(size - 1):
        x = normalized_test_data[i * time_step:(i + 1) * time_step, :input_size]
        y = normalized_test_data[i * time_step:(i + 1) * time_step, input_size]
        test_x.append(x.tolist())
        test_y.extend(y)
    test_x.append((normalized_test_data[(i + 1) * time_step:, :input_size]).tolist())
    test_y.extend((normalized_test_data[(i + 1) * time_step:, input_size]).tolist())
    return mean, std, test_x, test_y


def get_test_data2(time_step=20):
    data_test = data[-time_step:]
    print(data_test)
    print("----")
    mean = np.mean(data_test, axis=0)
    std = np.std(data_test, axis=0)
    print(std)
    normalized_test_data = (data_test - mean) / std
    test_x, test_y = [], []

    test_x.append((normalized_test_data[:, :input_size]).tolist())
    test_y.extend((normalized_test_data[:, input_size]).tolist())
    return mean, std, test_x, test_y


weights = {
    'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),
    'out': tf.Variable(tf.random_normal([rnn_unit, 1]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
}


def lstm(X):
    batch_size = tf.shape(X)[0]
    time_step = tf.shape(X)[1]
    w_in = weights['in']
    b_in = biases['in']
    input = tf.reshape(X, [-1, input_size])
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit, reuse=tf.AUTO_REUSE)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state,
                                                 dtype=tf.float32)
    output = tf.reshape(output_rnn, [-1, rnn_unit])
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output, w_out) + b_out

    return pred, final_states


def train_lstm(batch_size=80, time_step=20, train_begin=0, train_end=3400):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    Y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])
    batch_index, train_x, train_y = get_train_data(batch_size, time_step, train_begin, train_end)
    pred, _ = lstm(X)

    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))

    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    saver = tf.train.Saver(tf.global_variables())  # , max_to_keep=15
    # module_file = tf.train.latest_checkpoint('model2/')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, module_file)

        for i in range(max_step):
            for step in range(len(batch_index) - 1):

                _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x[batch_index[step]:batch_index[step + 1]],
                                                                 Y: train_y[batch_index[step]:batch_index[step + 1]]})
            print(i, loss_)
            if i % 10 == 0:
                print("----", saver.save(sess, './{}graph.ckpt'.format(modelPath), global_step=i))


def prediction(time_step=20):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    # Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    mean, std, test_x, test_y = get_test_data(time_step)

    pred, _ = lstm(X)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:

        module_file = tf.train.latest_checkpoint(modelPath)
        saver.restore(sess, module_file)
        test_predict = []
        for step in range(len(test_x) - 1):
            prob = sess.run(pred, feed_dict={X: [test_x[step]]})
            predict = prob.reshape((-1))
            test_predict.extend(predict)

        # lastList = test_x[-2][-10:] + test_x[-1]
        # prob = sess.run(pred, feed_dict={X: [lastList]})
        # predict = prob.reshape((-1))
        # test_predict.extend(predict[-10:])

        test_y = np.array(test_y) * std[input_size] + mean[input_size]
        # test_predict = np.array(test_predict) * std[input_size] + mean[input_size]
        test_predict = np.array(test_predict) * std[4] + mean[4]
        temp = np.abs(test_predict - test_y[:len(test_predict)]) / test_y[:len(test_predict)]
        acc = np.average(temp)
        print(acc)

        plt.figure()

        red_patch = mpatches.Patch(color='blue', label='origin')
        red_patch2 = mpatches.Patch(color='red', label='prediction')
        plt.legend(handles=[red_patch, red_patch2])

        plt.plot(list(range(len(test_predict))), test_predict, color='r')
        plt.plot(list(range(len(test_y))), test_y, color='b')
        plt.show()

        mean2, std2, test_x2, test_y2 = get_test_data2(time_step)
        test_predict2 = []
        prob = sess.run(pred, feed_dict={X: [test_x2[0]]})
        predict = prob.reshape((-1))
        test_predict2.extend(predict)

        # print(test_x2[:, 6])
        # test_predict21 = np.array(test_predict2) * std2[6] + mean2[6]
        # print(test_predict21)

        test_predict22 = np.array(test_predict2) * std2[4] + mean2[4]
        print(test_predict22)


if __name__ == '__main__':
    f = open(stockCsvPath)

    df = pd.read_csv(f)
    print(len(df))
    closeList = df['close']
    x = list(range(len(closeList)))
    y = closeList

    plt.figure()
    plt.plot(x, y, color='b')
    plt.show()

    f.close()

    data = df.iloc[:, 1:8].values

    # train_lstm()

    prediction()




