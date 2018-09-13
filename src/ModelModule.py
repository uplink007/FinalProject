from sklearn.utils import shuffle
from DLModule import DLClass
from DataModule import DataClass
from word2vec_module import MyWord2vec
from loggerModule import LoggerClass
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score as precision
from sklearn.metrics import recall_score as recall
from sklearn.metrics import f1_score
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import load_model
import gc
import sys
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
        self.preds = None

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
        self.logger.logger.critical("10 trainings average score START")

    def train_10_avg_score(self, model_name, model_type,  **kwargs):
        self.logger.logger.critical("10 trainings of {0} average score START".format(model_name))
        x, y = shuffle(self.data.preprocess["X"], self.data.labels, random_state=0)
        seed = 7
        n_splits = 10
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        scores = defaultdict(int)
        for train, test in kfold.split(x, y):
            self.nnmodel = DLClass(kwargs=kwargs)
            self.nnmodel.build_model(x[train], y[train], model_type)
            # nnmodel.fit(X_wcl_enriched[train],y_wcl[train],epochs=epochs,batch_size=100)
            print('Predicting...')
            preds = np.array([i[0] for i in self.nnmodel.model.predict_classes(x[test])])
            p = precision(preds, y[test])
            r = recall(preds, y[test])
            f1 = f1_score(preds, y[test])
            print('(Fold) Precision: ', p, ' | Recall: ', r, ' | F: ', f1)
            scores['Precision'] += p
            scores['Recall'] += r
            scores['F1'] += f1
        sys.stdout = open('../logs/{0}_scores.txt'.format(model_name), 'wt')
        print('Overall scores for model {0}:'.format(model_name))
        for n, sc in scores.items():
            print(n, '-> ', sc / n_splits * 1.0)
        sys.stdout = sys.__stdout__

    def predict_on_others(self, model_name, data_to_predict_name, labeled=True, threshold=0.5):
        self.logger.logger.critical('Starting predicting with {0} data {1} with threshold {2} '
                                    .format(model_name, data_to_predict_name, threshold))
        if labeled:
            x, y = shuffle(self.data.preprocess["X"], self.data.labels, random_state=0)
        else:
            x, y = self.data.preprocess['X'], self.data.labels
        self.nnmodel = load_model('../model/{0}.model'.format(model_name))
        self.preds = np.array([i[0] for i in self.nnmodel.predict_classes(x)])
        sys.stdout = open('../logs/{0}_predicted_{1}.txt'.format(model_name,data_to_predict_name), 'wt')
        if labeled:
            print(classification_report(y, self.preds))
        else:
            for idx, prediction in enumerate(self.preds):
                print(self.data.instances, '-->', prediction, '\n')
                print('Threshold 0.7')
                if prediction > threshold:
                    print('---------DEFINITION---------')
                else:
                    print('-------NOT DEFINITION-------')
        sys.stdout = sys.__stdout__



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


def run_module():
    word2vec = MyWord2vec()

    # wcl_w00
    data_wcl_w00 = DataClass("wcl_w00", depth="m")
    data_wcl_w00.getMaxLength(save_stats=True)
    data_wcl_w00.preprocessing_data(word2vec)
    model_wcl_w00 = Model(data_wcl_w00)
    model_wcl_w00.train("wcl_w00_cblstm_m", "cblstm", test_size=0.33)
    model_wcl_w00.train_10_avg_score("wcl_w00_cblstm_m", "cblstm")
    model_wcl_w00.train("wcl_w00_cnn_m", "cnn", test_size=0.33)
    model_wcl_w00.train_10_avg_score("wcl_w00_cnn_m", "cnn")
    data_wcl_w00 = DataClass("wcl_w00", depth="")
    data_wcl_w00.getMaxLength(save_stats=True)

    data_wcl_w00.preprocessing_data(word2vec)
    model_wcl_w00 = Model(data_wcl_w00)
    model_wcl_w00.train("wcl_w00_cblstm", "cblstm", test_size=0.33)
    model_wcl_w00.train_10_avg_score("wcl_w00_cblstm", "cblstm")
    model_wcl_w00.train("wcl_w00_cnn", "cnn", test_size=0.33)
    model_wcl_w00.train_10_avg_score("wcl_w00_cnn", "cnn")
    
    data_wcl_w00 = DataClass("wcl_w00", depth="ml")
    data_wcl_w00.getMaxLength(save_stats=True)
    data_wcl_w00.preprocessing_data(word2vec)
    model_wcl_w00 = Model(data_wcl_w00)
    model_wcl_w00.train("wcl_w00_cblstm_ml", "cblstm", test_size=0.33)
    model_wcl_w00.train_10_avg_score("wcl_w00_cblstm_ml", "cblstm")

    del data_wcl_w00
    del model_wcl_w00
    gc.collect()

    # wolfram_w00
    data_wolfram_w00 = DataClass("wolfram_w00", depth="ml")
    data_wolfram_w00.getMaxLength(save_stats=True)
    data_wolfram_w00.preprocessing_data(word2vec)
    model_wolfram_w00 = Model(data_wolfram_w00)
    model_wolfram_w00.train("wolfram_w00_cblstm_ml", "cblstm", test_size=0.33)
    model_wolfram_w00.train_10_avg_score("wolfram_w00_cblstm_ml", "cblstm")
    # data_wolfram_w00 = DataClass("wolfram_w00", depth="m")
    # data_wolfram_w00.getMaxLength(save_stats=True)
    # data_wolfram_w00.preprocessing_data(word2vec)
    # model_wolfram_w00 = Model(data_wolfram_w00)
    # model_wolfram_w00.train("wolfram_w00_cblstm_m", "cblstm", test_size=0.33)
    # model_wolfram_w00.train_10_avg_score("wolfram_w00_cblstm_m", "cblstm")
    # model_wolfram_w00.train("wolfram_w00_cnn_m", "cnn", test_size=0.33)
    # model_wolfram_w00.train_10_avg_score("wolfram_w00_cnn_m", "cnn")
    # data_wolfram_w00 = DataClass("wolfram_w00", depth="")
    # data_wolfram_w00.getMaxLength(save_stats=True)
    # data_wolfram_w00.preprocessing_data(word2vec)
    # model_wolfram_w00 = Model(data_wolfram_w00)
    # model_wolfram_w00.train("wolfram_w00_cblstm", "cblstm", test_size=0.33)
    # model_wolfram_w00.train_10_avg_score("wolfram_w00_cblstm", "cblstm")
    # model_wolfram_w00.train("wolfram_w00_cnn", "cnn", test_size=0.33)
    # model_wolfram_w00.train_10_avg_score("wolfram_w00_cnn", "cnn")
    del data_wolfram_w00
    del model_wolfram_w00
    gc.collect()
    # wolfram_wcl
    data_wolfram_wcl = DataClass("wolfram_wcl", depth="ml")
    data_wolfram_wcl.getMaxLength(save_stats=True)
    data_wolfram_wcl.preprocessing_data(word2vec)
    model_wolfram_wcl = Model(data_wolfram_wcl)
    model_wolfram_wcl.train("wolfram_wcl_cblstm_ml", "cblstm", test_size=0.33)
    model_wolfram_wcl.train_10_avg_score("wolfram_wcl_cblstm_ml", "cblstm")
    # data_wolfram_wcl = DataClass("wolfram_wcl", depth="m")
    # data_wolfram_wcl.getMaxLength(save_stats=True)
    # data_wolfram_wcl.preprocessing_data(word2vec)
    # model_wolfram_wcl = Model(data_wolfram_wcl)
    # model_wolfram_wcl.train("wolfram_wcl_cblstm_m", "cblstm", test_size=0.33)
    # model_wolfram_wcl.train_10_avg_score("wolfram_wcl_cblstm_m", "cblstm")
    # model_wolfram_wcl.train("wolfram_wcl_cnn_m", "cnn", test_size=0.33)
    # model_wolfram_wcl.train_10_avg_score("wolfram_wcl_cnn_m", "cnn")
    # data_wolfram_wcl = DataClass("wolfram_wcl", depth="")
    # data_wolfram_wcl.getMaxLength(save_stats=True)
    # data_wolfram_wcl.preprocessing_data(word2vec)
    # model_wolfram_wcl = Model(data_wolfram_wcl)
    # model_wolfram_wcl.train("wolfram_wcl_cblstm", "cblstm", test_size=0.33)
    # model_wolfram_wcl.train_10_avg_score("wolfram_wcl_cblstm", "cblstm")
    # model_wolfram_wcl.train("wolfram_wcl_cnn", "cnn", test_size=0.33)
    # model_wolfram_wcl.train_10_avg_score("wolfram_wcl_cnn", "cnn")
    del data_wolfram_wcl
    del model_wolfram_wcl
    gc.collect()

    # wcl_w00_cblstm -> wolfram ml
    data_wcl_w00_wolfram = DataClass("wolfram", depth="ml", model_name='wcl_w00')
    data_wcl_w00_wolfram.preprocessing_data(word2vec, model=True)
    model_wcl_w00_wolfram = Model(data_wcl_w00_wolfram)
    model_wcl_w00_wolfram.predict_on_others('wcl_w00_cblstm_ml', 'wolfram')
    # wcl_w00_cnn -> wolfram ml
    model_wcl_w00_wolfram.predict_on_others('wcl_w00_cnn_ml', 'wolfram')
    # wcl_w00_cblstm -> wolfram m
    data_wcl_w00_wolfram = DataClass("wolfram", depth="m", model_name='wcl_w00')
    data_wcl_w00_wolfram.preprocessing_data(word2vec, model=True)
    model_wcl_w00_wolfram = Model(data_wcl_w00_wolfram)
    model_wcl_w00_wolfram.predict_on_others('wcl_w00_cblstm_m', 'wolfram')
    # wcl_w00_cnn -> wolfram ml
    model_wcl_w00_wolfram.predict_on_others('wcl_w00_cnn_m', 'wolfram')
    # wcl_w00_cblstm -> wolfram
    data_wcl_w00_wolfram = DataClass("wolfram", depth="", model_name='wcl_w00')
    data_wcl_w00_wolfram.preprocessing_data(word2vec, model=True)
    model_wcl_w00_wolfram = Model(data_wcl_w00_wolfram)
    model_wcl_w00_wolfram.predict_on_others('wcl_w00_cblstm', 'wolfram')
    # wcl_w00_cnn -> wolfram ml
    model_wcl_w00_wolfram.predict_on_others('wcl_w00_cnn', 'wolfram')
    del data_wcl_w00_wolfram
    del model_wcl_w00_wolfram
    gc.collect()
    
    # wolfram_wcl_cblstm -> w00 ml
    data_wolfram_wcl_w00 = DataClass("w00", depth="ml", model_name='wolfram_wcl')
    data_wolfram_wcl_w00.preprocessing_data(word2vec, model=True)
    model_wolfram_wcl_w00 = Model(data_wolfram_wcl_w00)
    model_wolfram_wcl_w00.predict_on_others('wolfram_wcl_cblstm_ml', 'w00')
    # wolfram_wcl_cnn -> w00 ml
    model_wolfram_wcl_w00.predict_on_others('wolfram_wcl_cnn_ml', 'w00')
    # wolfram_wcl_cblstm -> w00 m
    data_wolfram_wcl_w00 = DataClass("w00", depth="m", model_name='wolfram_wcl')
    data_wolfram_wcl_w00.preprocessing_data(word2vec, model=True)
    model_wolfram_wcl_w00 = Model(data_wolfram_wcl_w00)
    model_wolfram_wcl_w00.predict_on_others('wolfram_wcl_cblstm_m', 'w00')
    # wolfram_wcl_cnn -> w00 ml
    model_wolfram_wcl_w00.predict_on_others('wolfram_wcl_cnn_m', 'w00')
    # wolfram_wcl_cblstm -> w00
    data_wolfram_wcl_w00 = DataClass("w00", depth="", model_name='wolfram_wcl')
    data_wolfram_wcl_w00.preprocessing_data(word2vec, model=True)
    model_wolfram_wcl_w00 = Model(data_wolfram_wcl_w00)
    model_wolfram_wcl_w00.predict_on_others('wolfram_wcl_cblstm', 'w00')
    # wolfram_wcl_cnn -> w00 ml
    model_wolfram_wcl_w00.predict_on_others('wolfram_wcl_cnn', 'w00')
    del data_wolfram_wcl_w00
    del model_wolfram_wcl_w00
    gc.collect()

    # wolfram_w00_cblstm -> wcl ml
    data_wolfram_w00_wcl = DataClass("wcl", depth="ml", model_name='wolfram_w00')
    data_wolfram_w00_wcl.preprocessing_data(word2vec, model=True)
    model_wolfram_w00_wcl = Model(data_wolfram_w00_wcl)
    model_wolfram_w00_wcl.predict_on_others('wolfram_w00_cblstm_ml', 'wcl')
    # wolfram_w00_cnn -> wcl ml
    model_wolfram_w00_wcl.predict_on_others('wolfram_w00_cnn_ml', 'wcl')
    # wolfram_w00_cblstm -> wcl m
    data_wolfram_w00_wcl = DataClass("wcl", depth="m", model_name='wolfram_w00')
    data_wolfram_w00_wcl.preprocessing_data(word2vec, model=True)
    model_wolfram_w00_wcl = Model(data_wolfram_w00_wcl)
    model_wolfram_w00_wcl.predict_on_others('wolfram_w00_cblstm_m', 'wcl')
    # wolfram_w00_cnn -> wcl ml
    model_wolfram_w00_wcl.predict_on_others('wolfram_w00_cnn_m', 'wcl')
    # wolfram_w00_cblstm -> wcl
    data_wolfram_w00_wcl = DataClass("wcl", depth="", model_name='wolfram_w00')
    data_wolfram_w00_wcl.preprocessing_data(word2vec, model=True)
    model_wolfram_w00_wcl = Model(data_wolfram_w00_wcl)
    model_wolfram_w00_wcl.predict_on_others('wolfram_w00_cblstm', 'wcl')
    # wolfram_w00_cnn -> wcl ml
    model_wolfram_w00_wcl.predict_on_others('wolfram_w00_cnn', 'wcl')
    del data_wolfram_w00_wcl
    del model_wolfram_w00_wcl
    gc.collect()


if __name__ == "__main__":
    run_module()

