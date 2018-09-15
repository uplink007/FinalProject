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
import sys
from keras.models import load_model
from argparse import ArgumentParser


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


def score_func(data_name, word2vec, test_size=0.33):
    """
    Function that create a models and save it and then it using StratifiedKFold (kfold cross validation )
    and creating score report in format 1_2_3_scores.txt :
        1 - model name
        2 - model type
        3 - depth of the prepossessing
        example :
            input : ("w00",word2vec,0.33)
            output:
                1 log file: w00_cblstm_ml_scores.txt
                2 log file: w00_cblstm_m_scores.txt
                3 log file: w00_cblstm_scores.txt
                4 log file: w00_cnn_scores.txt
                5 log file: w00_cnn_m_scores.txt
                6 log file: w00_cnn_ml_scores.txt
    :param data_name:training data for the model
    :param word2vec: word2vec to be used in the prepossessing step
    :param test_size: split data size
    """
    if word2vec is not "Google_300":
        word2vec = MyWord2vec(word2vec)
    else:
        word2vec = MyWord2vec()
    data = DataClass(data_name, depth="m")
    data.getMaxLength(save_stats=True)
    data.preprocessing_data(word2vec)
    model = Model(data)
    model.train("{0}_cblstm_m".format(data_name), "cblstm", test_size=test_size)
    model.train_10_avg_score("{0}_cblstm_m".format(data_name), "cblstm")
    model.train("{0}_cnn_m".format(data_name), "cnn", test_size=test_size)
    model.train_10_avg_score("{}_cnn_m".format(data_name), "cnn")

    data = DataClass(data_name, depth="")
    data.getMaxLength(save_stats=True)
    data.preprocessing_data(word2vec)
    model = Model(data)
    model.train("{0}_cblstm".format(data_name), "cblstm", test_size=test_size)
    model.train_10_avg_score("{0}_cblstm".format(data_name), "cblstm")
    model.train("{0}_cnn".format(data_name), "cnn", test_size=test_size)
    model.train_10_avg_score("{0}_cnn".format(data_name), "cnn")

    data = DataClass(data_name, depth="ml")
    data.getMaxLength(save_stats=True)
    data.preprocessing_data(word2vec)
    model = Model(data)
    model.train("{0}_cblstm_ml".format(data_name), "cblstm", test_size=test_size)
    model.train_10_avg_score("{0}_cblstm_ml".format(data_name), "cblstm")
    model.train("{0}_cnn_ml".format(data_name), "cnn", test_size=test_size)
    model.train_10_avg_score("{0}_cnn_ml".format(data_name), "cnn")


def prediction_func(model_name, word2vec, predict_data, depth, threshold=0.5):
    """
    Function that predict model scores on other data sets
    Function save in the log classification report in the format 1_2_3_predicted_4.txt:
            1 - model name
            2 - model type
            3 - depth of the postprocessing
            4 - predicted data name
            example :
                    input :("w00",wor2vec,"wcl","ml")
                    log file 1 : w00_cblstm_ml_predicted_wcl.txt
                    log file 2 : w00_cnn_ml_predicted_wcl.txt
    :param model_name: Name of the model to be used for the prediction
    :param word2vec: word2vec to be used for the postprocessing step
    :param predict_data: name of the data to be predicted by the model
    :param depth: the depth of the postprocessing stem can be ("ml","m","")
    :param threshold: the threshold for the prediction safety , default value is 0.5
    """
    if word2vec is not "Google_300":
        word2vec = MyWord2vec(word2vec)
    else:
        word2vec = MyWord2vec()
    data = DataClass(predict_data, depth=depth, model_name=model_name)
    data.preprocessing_data(word2vec, model=True)
    model = Model(data)
    model.predict_on_others('{0}_cblstm_{1}'.format(model_name, depth), predict_data, threshold=threshold)
    model.predict_on_others('{0}_cnn_{1}'.format(model_name, depth), predict_data, threshold=threshold)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-u', '--use', help="""What action need to be made:
    predict -   Function that predict model scores on other data sets
                Function save in the log classification report in the format 1_2_3_predicted_4.txt:
                    1 - model name
                    2 - model type
                    3 - depth of the postprocessing
                    4 - predicted data name
                    example :
                            input :("w00",wor2vec,"wcl","ml")
                            log file 1 : w00_cblstm_ml_predicted_wcl.txt
                            log file 2 : w00_cnn_ml_predicted_wcl.txt
    score -     Function that create a models and save it and then it using StratifiedKFold (kfold cross validation )
                Function creating score report in format 1_2_3_scores.txt :
                    1 - model name
                    2 - model type
                    3 - depth of the prepossessing
                    example :
                        input : ("w00",word2vec,0.33)
                        output:
                            1 log file: w00_cblstm_ml_scores.txt
                            2 log file: w00_cblstm_m_scores.txt
                            3 log file: w00_cblstm_scores.txt
                            4 log file: w00_cnn_scores.txt
                            5 log file: w00_cnn_m_scores.txt
                            6 log file: w00_cnn_ml_scores.txt""", required=True, choices=['score', 'predict'])
    parser.add_argument('-wv', '--word-vector', help="""Name of word2vec to be used;
Can be empty to use the default one in the config file.""", required=False)
    parser.add_argument('-m', '--model-name', help="""Model name for prediction function""", required=False)
    parser.add_argument('-dep', '--dependencies', help='Option for using dependencies', required=False,
                        choices=['ml', 'm', 'n'])
    parser.add_argument('-d', '--data', help='Data for train or prediction', required=True)
    parser.add_argument('-t', '--threshold', help='Threshold for predicting data ', type=float, required=False)
    parser.add_argument('-s', '--split', help='split size for split test size', type=float, required=False)

    args = vars(parser.parse_args())
    try:
        if args['threshold']is not None:
            if args['threshold'] <= 0 or args['threshold'] >= 1:
                parser.error('Threshold can be only float number between 0 and 1')
    except KeyError:
        args['threshold'] = 0.5
    try:
        if args['split'] is not None:
            if args['split'] <= 0 or args['split'] >= 1:
                parser.error('Split can be only float number between 0 and 1')
    except KeyError:
        args['split'] = 0.33
    try:
        if args['dependencies'] is not None and args['depth'] == 'n':
            args['dependencies'] = ''
    except KeyError:
        if args['use'] == 'predict':
            parser.error('predict requires --model-name, --data and --dependencies')
    if args['use'] is not None:
        try:
            if args['word-vector'] is not None:
                word2vec = MyWord2vec(args['word-vector'])
        except KeyError:
            word2vec = "Google_300"
        if args['use'] == 'predict':
            if args['model_name'] is not None and args['data'] is not None and args['dependencies'] is not None:
                prediction_func(args['model_name'], word2vec, args['data'], args['depth'], threshold=args['threshold'])
            else:
                parser.error("predict requires --model-name, --data and --dependencies")
        elif args['use'] == 'score':
            if args['data'] is not None:
                score_func(args['data'], word2vec, test_size=args['split'])
            else:
                parser.error("score requires --data")
        else:
            parser.error("use can be only score or predict")
