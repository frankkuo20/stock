import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os


class RnnLSTM:
    max_step = 1000
    rnn_unit = 30
    input_size = 6
    output_size = 1
    lr = 0.0006
    testNum = 20
    #
    time_step = 20
    batch_size = 80
    weights = {
        'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),
        'out': tf.Variable(tf.random_normal([rnn_unit, 1]))
    }
    biases = {
        'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
        'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
    }

    HIGH = 'high'
    CLOSE = 'close'

    def __init__(self, stockNo, type):
        self.stockNo = stockNo

        self._setType(type)
        self._getData()

        self._setModelPath()

    def _setType(self, type):

        if type not in [self.HIGH, self.CLOSE]:
            print()
        self.type = type

    def _setModelPath(self):

        modelPath = 'model/{}-{}/'.format(self.stockNo, self.type)
        if not os.path.exists(modelPath):
            os.mkdir(modelPath)
        self.modelPath = modelPath

    def _getData(self):
        # stockCsvPath = '{}-all.csv'.format(self.stockNo)
        stockCsvPath = '{}-all-{}.csv'.format(self.stockNo, self.type)

        f = open(stockCsvPath)
        stockPd = pd.read_csv(f)
        f.close()

        self.stockPd = stockPd
        self.data = stockPd.iloc[:, 1:8].values

    def showCloseChart(self):
        closeList = self.stockPd['close']
        x = list(range(len(closeList)))
        y = closeList

        plt.figure()
        plt.plot(x, y, color='b')
        plt.show()

    def get_train_data(self, train_begin=0):
        input_size = self.input_size
        data = self.data
        time_step = self.time_step
        batch_size = self.batch_size

        batch_index = []
        data_train = data[train_begin:-20]

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

    def get_test_data(self, test_begin=3401):
        input_size = self.input_size
        data = self.data
        time_step = self.time_step

        data_test = data[test_begin:test_begin + 40]
        print(len(data_test))
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

        # test_x.append((normalized_test_data[(i + 1) * time_step:, :input_size]).tolist())
        # test_y.extend((normalized_test_data[(i + 1) * time_step:, input_size]).tolist())
        return mean, std, test_x, test_y

    def get_test_data2(self):
        input_size = self.input_size
        data = self.data
        time_step = self.time_step

        data_test = data[-20:]
        # data_test = data[-21:-1]

        mean = np.mean(data_test, axis=0)
        std = np.std(data_test, axis=0)
        normalized_test_data = (data_test - mean) / std
        size = (len(normalized_test_data) + time_step - 1) // time_step
        test_x, test_y = [], []
        for i in range(size):
            x = normalized_test_data[i * time_step:(i + 1) * time_step, :input_size]
            y = normalized_test_data[i * time_step:(i + 1) * time_step, input_size]
            test_x.append(x.tolist())
            test_y.extend(y)
        return mean, std, test_x, test_y

    def lstm(self, X):
        weights = self.weights
        biases = self.biases
        input_size = self.input_size
        rnn_unit = self.rnn_unit

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

    def train_lstm(self, time_step=20, train_begin=0):
        input_size = self.input_size
        output_size = self.output_size
        lr = self.lr
        max_step = self.max_step
        modelPath = self.modelPath

        X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
        Y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])
        batch_index, train_x, train_y = self.get_train_data(train_begin)

        pred, _ = self.lstm(X)

        loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))

        train_op = tf.train.AdamOptimizer(lr).minimize(loss)
        saver = tf.train.Saver(tf.global_variables())  # , max_to_keep=15
        # module_file = tf.train.latest_checkpoint('model2/')

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # saver.restore(sess, module_file)

            for i in range(max_step):
                for step in range(len(batch_index) - 1):
                    _, loss_ = sess.run([train_op, loss],
                                        feed_dict={X: train_x[batch_index[step]:batch_index[step + 1]],
                                                   Y: train_y[batch_index[step]:batch_index[step + 1]]})
                print(i, loss_)
                if i % 10 == 0:
                    print("----", saver.save(sess, './{}graph.ckpt'.format(modelPath), global_step=i))

    def prediction(self, time_step=20):
        input_size = self.input_size
        modelPath = self.modelPath

        X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
        mean, std, test_x, test_y = self.get_test_data(time_step)

        pred, _ = self.lstm(X)
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            module_file = tf.train.latest_checkpoint(modelPath)
            saver.restore(sess, module_file)
            test_predict = []
            # for step in range(len(test_x) - 1):
            for step in range(len(test_x)):
                prob = sess.run(pred, feed_dict={X: [test_x[step]]})
                predict = prob.reshape((-1))
                test_predict.extend(predict)

            test_y = np.array(test_y) * std[input_size] + mean[input_size]

            # test_predict = np.array(test_predict) * std[input_size] + mean[input_size]
            test_predict = np.array(test_predict) * std[4] + mean[4]
            # temp = np.abs(test_predict - test_y[:len(test_predict)]) / test_y[:len(test_predict)]
            # acc = np.average(temp)
            # print(acc)

            # print(test_y)
            # print(test_predict)
            plt.figure()

            test_y2 = self.stockPd.iloc[3400:3421, 5:6].values
            print(test_y2)

            blue_patch = mpatches.Patch(color='blue', label='origin')
            red_patch = mpatches.Patch(color='red', label='prediction')
            plt.legend(handles=[blue_patch, red_patch])

            # test_y = list(test_y)
            # test_y.insert(0, "0")
            plt.plot(list(range(len(test_predict))), test_predict, color='r')
            # plt.plot(list(range(len(test_y))), test_y, color='b')
            plt.plot(list(range(len(test_y2))), test_y2, color='black')
            plt.show()

            # mean2, std2, test_x2, test_y2 = self.get_test_data2(time_step)
            # test_predict2 = []
            # prob = sess.run(pred, feed_dict={X: [test_x2[0]]})
            # predict = prob.reshape((-1))
            # test_predict2.extend(predict)

            # print(test_x2[:, 6])
            # test_predict21 = np.array(test_predict2) * std2[6] + mean2[6]
            # print(test_predict21)

            # test_predict22 = np.array(test_predict2) * std2[4] + mean2[4]
            # print(test_predict22)

    def get_date(self):
        stockPd = self.stockPd
        print(stockPd.iloc[3400:3421, 0:8].values)

    def getLastResult(self):
        input_size = self.input_size
        modelPath = self.modelPath
        time_step = 20

        X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
        mean, std, test_x, test_y = self.get_test_data2()

        pred, _ = self.lstm(X)
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            module_file = tf.train.latest_checkpoint(modelPath)
            saver.restore(sess, module_file)
            test_predict = []
            for step in range(len(test_x)):
                prob = sess.run(pred, feed_dict={X: [test_x[step]]})
                predict = prob.reshape((-1))
                test_predict.extend(predict)

            test_y = np.array(test_y) * std[input_size] + mean[input_size]

            # test_predict = np.array(test_predict) * std[4] + mean[4]
            test_predict = np.array(test_predict) * std[2] + mean[2]
            print(test_predict)

            plt.figure()

            blue_patch = mpatches.Patch(color='blue', label='origin')
            red_patch = mpatches.Patch(color='red', label='prediction')
            plt.legend(handles=[blue_patch, red_patch])

            plt.plot(list(range(len(test_predict))), test_predict, color='r')
            plt.plot(list(range(len(test_y))), test_y, color='b')
            plt.show()
