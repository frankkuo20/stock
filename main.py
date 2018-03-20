from RnnLSTM import RnnLSTM

if __name__ == '__main__':

    # main stock
    # rnnLSTM = RnnLSTM(stockNo='main', type='close')
    rnnLSTM = RnnLSTM(stockNo='100000', type='close')
    rnnLSTM.max_step = 1000
    rnnLSTM.rnn_unit = 30
    rnnLSTM.input_size = 6
    rnnLSTM.output_size = 1
    rnnLSTM.lr = 0.0006

    # rnnLSTM.showCloseChart()
    # rnnLSTM.train_lstm()
    # rnnLSTM.prediction()

    # rnnLSTM.get_date()
    rnnLSTM.getLastResult()

    # rnnLSTM = RnnLSTM(stockNo='0050')
    # rnnLSTM.max_step = 200
    # rnnLSTM.rnn_unit = 30
    # rnnLSTM.input_size = 8
    # rnnLSTM.output_size = 1
    # rnnLSTM.lr = 0.0006
    # rnnLSTM.train_lstm()
    # rnnLSTM.prediction()

