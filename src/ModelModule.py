from sklearn.utils import shuffle
from src.DLModule import DLClass
from src.DataModule import DataClass
from src.word2vec_module import MyWord2vec
from src.loggerModule import LoggerClass
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import load_model


class Model(object):
    def __init__(self, preprocessed_data):
        """
        Init model for training
        :param preprocessed_data: data that was already preprocessed
        """
        self.logger = LoggerClass(name="auto_de")
        self.data = preprocessed_data
        if self.data.preprocessed is not True:
            raise ValueError
        self.nnmodel = None

    def train(self, model_name, model_type, test_size=0.33, **kwargs):
        """
        get all the data preprocess it and train the model and save
        :param model_name: Name of the model for saving
        :param model_type: Type of the model, can be cnn or cblstm
        """
        x, y = shuffle(self.data.preprocess["X"], self.data.labels, random_state=0)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
        self.nnmodel = DLClass(kwargs=kwargs)
        self.nnmodel.build_model(X_train, y_train, model_type)
        predicts = np.array([i[0] for i in self.nnmodel.model.predict_classes(X_test)])
        print(classification_report(y_test, predicts))
        self.logger.logger.critical("Classification Report: \n{0}".format(classification_report(y_test, predicts)))
        self.nnmodel.model.save("../model/{0}.model".format(model_name))

    # def predict(self, model_name, data_type=None, my_list=None, threshold=0.8):
    #     """
    #     take the data from my_list , create data object , preprocess it and predict with our model
    #     :param my_list:  data from user
    #     :param data_type: data from dataset
    #     :param model_name: Name of the model to be used for prediction
    #     :param threshold: default 0.8
    #     """
    #     self.data = DataClass(config=self.logger.config)
    #     if my_list is not None:
    #         self.data.data_init(my_list=my_list)
    #     elif data_type is not None:
    #         self.data.data_init(data_type=data_type)
    #     self.preprocess = PreprocessClass(get_data=self.data, get_word2vec=self.word2vec, config=self.logger.config)
    #     self.preprocess.load_all(model_name)
    #     self.preprocess.preprocessing_data()
    #     self.nnmodel = load_model("../model/{0}.model".format(model_name))
    #     for idx, sent in enumerate(self.data.instances):
    #         preds = self.nnmodel.predict(np.array([self.preprocess.X[idx]]))[0][0]
    #         if preds > threshold:
    #             print('{0} Sent: '.format(idx), sent, ' -> ', preds)


if __name__ == "__main__":
    word2vec = MyWord2vec()
    data_wcl = DataClass("wcl", depth="ml")
    data_wcl.getMaxLength(save_stats=True)
    data_wcl.preprocessing_data(word2vec)
    data_w00 = DataClass("w00", depth="ml")
    data_w00.getMaxLength(save_stats=True)
    data_w00.preprocessing_data(word2vec)
    data_wolfram = DataClass("wolfram", depth="ml")
    data_wolfram.getMaxLength(save_stats=True)
    data_wolfram.preprocessing_data(word2vec)
    data_all = DataClass("", depth="ml")
    data_all.getMaxLength(save_stats=True)
    data_all.preprocessing_data(word2vec)
    model_wcl = Model(data_wcl)
    model_w00 = Model(data_wcl)
    model_wolfram = Model(data_wcl)
    model_all = Model(data_wcl)
    model_wcl.train("wcl", "cblstm", test_size=0.33)
    model_w00.train("wcl", "cblstm", test_size=0.33)
    model_wolfram.train("wcl", "cblstm", test_size=0.33)
    model_all.train("wcl", "cblstm", test_size=0.33)
    pass

