from RnnLSTM import RnnLSTM

if __name__ == '__main__':

    # main stock
    rnnLSTM = RnnLSTM(stockNo='main')
    rnnLSTM.max_step = 101
    rnnLSTM.rnn_unit = 30
    rnnLSTM.input_size = 6
    rnnLSTM.output_size = 1
    rnnLSTM.lr = 0.0006

    # rnnLSTM.showCloseChart()
    rnnLSTM.train_lstm()
    rnnLSTM.prediction()


