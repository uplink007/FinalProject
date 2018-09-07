import sys
import logging
from keras.models import Sequential
from keras.layers import Dense, Activation, Bidirectional, Dropout, Conv1D, MaxPooling1D, Flatten, LSTM

module_logger = logging.getLogger('auto_de.DLModule')


class DLClass(object):
    def __init__(self,
                 kernel_size=3,
                 filters=100,
                 pool_size=4,
                 strides=1,
                 epochs=10,
                 batch_size=100,
                 Conv1D_padding='valid',
                 Conv1D_activation='relu',
                 dropout=0.5,
                 kwargs={}):
        """
        Init Model base params
        :param kernel_size:
        :param filters:
        :param pool_size:
        :param strides:
        :param epochs:
        :param batch_size:
        :param Conv1D_padding:
        :param Conv1D_activation:
        :param dropout:
        """
        self.logger = logging.getLogger("DLClass")
        self.logger.critical("Build model init")
        self.kernel_size = kernel_size
        self.filters = filters
        self.pool_size = pool_size
        self.strides = strides
        self.epochs = epochs
        self.batch_size = batch_size
        self.Conv1D_padding = Conv1D_padding
        self.Conv1D_activation = Conv1D_activation
        self.dropout = dropout
        self.model = Sequential()
        self.logger.critical("Build model init finished")

    def build_model(self, x, y, model_type, lstm_units=100, validation_data=''):
        """
        Train model
        :param x: preprocessed instances
        :param y: array of labels
        :param model_type: type of model, can be cnn or cblstm
        :param lstm_units: lstm units, default value 100
        :param validation_data:
        """
        self.logger.critical("Build {0} model start".format(model_type))
        self.model.add(Conv1D(self.filters,
                              self.kernel_size,
                              padding=self.Conv1D_padding,
                              activation=self.Conv1D_activation,
                              strides=self.strides,
                              input_shape=(x.shape[1], x.shape[2])))
        self.model.add(MaxPooling1D(pool_size=self.pool_size))
        if model_type == 'cnn':
            self.model.add(Flatten())
            self.model.add(Dropout(self.dropout))
        elif model_type == 'cblstm':
            self.model.add(Bidirectional(LSTM(lstm_units)))
            self.model.add(Dropout(self.dropout))
        else:
            sys.exit('Model type must be "cnn" or "blstm"')
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))
        self.model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])
        print('Train with ', len(x))
        print(self.model.summary())
        self.logger.critical("Model summary\n{0}".format(self.model.summary()))
        self.model.fit(x, y, epochs=self.epochs, batch_size=self.batch_size, validation_data=validation_data)
        self.logger.critical("Build {0} model finished".format(model_type))
