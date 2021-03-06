import configparser
import ctypes
import json
import os
import pickle
import numpy as np
import logging
import spacy
import gc
from stanfordcorenlp import StanfordCoreNLP
from word2vec_module import MyWord2vec
from loggerModule import LoggerClass
import pdb
# logger for DataModule
module_logger = logging.getLogger('auto_de.DataModule')
DATA_MODULE_CONFIG_SECTION = "DataModule"


class PyObject(ctypes.Structure):
    _fields_ = [("refcnt", ctypes.c_long)]


class DataClass(object):
    def __init__(self, name, depth="ml", model_name=None):
        """
        Class that contains data for model training \n
        Config file location is /src/configuration/config.ini \n
        :param name: name of the dataset to be loaded valid_data_names = {'wcl', 'w00', 'wolfram', ""}
        :param depth: The depth in the training process, can be ml ,m, or empty string
        :param model_name: If data need to be preprocessed for specific model, the model name need to be assign
        """
        self.model_name = model_name
        self.logger = logging.getLogger('auto_de.DataModule.DataClass')
        self.logger.info("Data Class init Start")
        self.name = name
        self.preprocessed = False
        self.instances = []
        self.labels = np.array([])
        self.config = None
        self.__init_config()
        self.init_data(name)
        self.preprocess = None
        self.__init_preprocess(depth)
        self.logger.info("Data Class init Finished")

    def __init_config(self, path="configuration/config.ini"):
        """
        Read config file
        :param path: path to config file
        :return: None
        """
        self.config = configparser.ConfigParser()
        self.config.read(path)

    def init_data(self, name):
        """
        Load specific data set to DataCLass object
        :param name: name of the dataset valid_data_names = {'wcl', 'w00', 'wolfram',
                                                            'wolfram_w00', 'wolfram_wcl',
                                                             'wcl_w00', "all"}
        :return: None
        """
        folder = None
        valid_data_names = {'wcl', 'w00', 'wolfram', 'wolfram_w00', 'wolfram_wcl', 'wcl_w00', "all"}
        path = self.config[DATA_MODULE_CONFIG_SECTION]["path"]
        if name not in valid_data_names:
            self.logger.error("results: status must be one of %r." % valid_data_names)
            raise ValueError("results: status must be one of %r." % valid_data_names)
        else:
            self.logger.critical("init {0}{1} dataset".format(path, name))
        if name != "all":
            folder = self.config[DATA_MODULE_CONFIG_SECTION][name]
        if name == "wcl":
            self.__load_wcl(path, folder)
        elif name == "w00":
            self.__load_w00(path, folder)
        elif name == "wolfram":
            self.__load_wolfram(path, folder)
        elif name == "wolfram_w00":
            self.__load_wolfram(path, "")
            self.__load_w00(path, "")
        elif name == "wolfram_wcl":
            self.__load_wolfram(path, "")
            self.__load_wcl(path, "")
        elif name == "wcl_w00":
            self.__load_wcl(path, "")
            self.__load_w00(path, "")
        else:
            self.__load_all(path)
        self.logger.info("Reading data form file succeeded")

    @staticmethod
    def __pars_wcl(label, lines):
        """
        Function that parse the wcl data lines to a separated instances and append there real label
        :param label: good or bad
        :param lines: all the data from file by lines
        :return: array of labels and instances for each instances : instances[i] have a classification labels[i]
        """
        labels, instances = [], []
        for idx, line in enumerate(lines):
            if line.startswith('#'):
                target = lines[idx + 1].split(':')[0]
                if target[0] == "!":
                    target = target[1:]
                sent = line[2:].replace('TARGET', target).strip().lower()
                if label == 'good':
                    labels.append(1)
                else:
                    labels.append(0)
                instances.append(sent)
        return labels, instances

    # TODO delete after tests
    def word2vec_temp(self):
        """
        Method that create structure for word2vec training
        Used for tem word2vec creation for debugging
        :return: [["word1","word2","word3"...],[sentence 2],[sentence 3]...]
        """
        nlp = spacy.load("en_core_web_sm")
        result_main = []
        for line in self.instances:
            sentenses = nlp(line)
            result = []
            for idx, token in enumerate(sentenses):
                result.append(token.orth_)
            result_main.append(result)
        return result_main

    @staticmethod
    def __pars_wolfram(label, lines):
        """
        Function that parse the w00 data lines to a separated instances and append there real label
        :param label: good or bad
        :param lines: all the data from file by lines
        :return: array of labels and instances for each instances : instances[i] have a classification labels[i]
        """
        labels, instances = [], []
        for idx, line in enumerate(lines):
            if label == 'good':
                labels.append(1)
            else:
                labels.append(0)
            instances.append(line)
        return labels, instances

    def __load_wcl(self, path, folder):
        """
        Load wcl dataset
        :param path: path to data folder
        :param folder: dataset folder name
        :return: init self.instances and self.labels
        """
        self.logger.critical("Reading wiki data")
        for root, subdirs, files in os.walk(path+folder):
            for filename in files:
                if filename.startswith('wiki_'):
                    label = filename.split('_')[-1].replace('.txt', '')
                    doc = os.path.join(root, filename)
                    lines = open(doc, 'r', encoding='utf-8').readlines()
                    try:
                        self.logger.info("Start pars_wcl")
                        labels, instances = self.__pars_wcl(label, lines)
                    except:
                        self.logger.error("__pars_wcl failed")
                        raise
                    self.logger.info("__pars_wcl succeeded")
                    self.instances = self.instances + instances
                    self.labels = np.concatenate((self.labels, np.array(labels)), axis=None)

    def __load_w00(self, path, folder):
        """
        Load w00 dataset
        :param path: path to data folder
        :param folder: dataset folder name
        :return: init self.instances and self.labels
        """
        self.logger.critical("Reading w00 data")
        error_count = 0
        for root, subdirs, files in os.walk(path+folder):
            for filename in files:
                if filename.startswith("annotated"):
                    if filename == 'annotated.word':
                        sents = open(os.path.join(root, filename), 'r').readlines()
                    elif filename == 'annotated.meta':
                        labels = open(os.path.join(root, filename), 'r').readlines()
                    try:
                        if (sents is not None) and (labels is not None):
                            for idx, sent in enumerate(sents):
                                sent = sent.strip().lower()
                                label = int(labels[idx].split(' $ ')[0])
                                self.instances.append(sent)
                                self.labels = np.concatenate((self.labels, np.array(label)), axis=None)
                    except UnboundLocalError:
                        if error_count < 10:
                            error_count = error_count+1
                            pass
                        else:
                            raise UnboundLocalError

    def __load_wolfram(self, path, folder):
        """
        Load wolfram dataset
        :param path: path to data folder
        :param folder: dataset folder name
        :return: init self.instances and self.labels
        """
        self.logger.critical("Reading wolfram data")
        for root, subdirs, files in os.walk(path+folder):
            for filename in files:
                if filename.startswith("wolfram"):
                    label = filename.split('_')[-1].replace('.txt', '')
                    doc = os.path.join(root, filename)
                    lines = open(doc, 'r', encoding='utf-8').readlines()
                    try:
                        self.logger.info("Start pars_wolfram")
                        labels, instances = self.__pars_wolfram(label, lines)
                    except:
                        self.logger.error("__pars_wolfram failed")
                        raise
                    self.logger.info("__pars_wolfram succeeded")
                    self.instances = self.instances + instances
                    self.labels = np.concatenate((self.labels, np.array(labels)), axis=None)

    def __load_all(self, path):
        """
        Load all datasets together
        :param path: path to datasets folder
        :return: None
        """
        self.__load_wcl(path, "")
        self.__load_w00(path, "")
        self.__load_wolfram(path, "")

    @staticmethod
    def get_pair_words(json_object):
        """
        Method that pars Stanford core nlp return jason
        :param json_object: json from Stanfor core nlp
        :return: parsed dictionary
        """
        result = {}
        res = {}
        try:
            [res.update({k['dependent']: k['dep']}) for k in json_object['sentences'][0]['basicDependencies']]
        except IndexError:
            raise IndexError
        for idx, k in enumerate(json_object['sentences'][0]['basicDependencies'][1:]):
            result[idx] = {'parent_word': json_object['sentences'][0]['basicDependencies'][1:][idx]['governorGloss'],
                           'parent_dep': res[json_object['sentences'][0]['basicDependencies'][1:][idx]['governor']],
                           'child_word': json_object['sentences'][0]['basicDependencies'][1:][idx]['dependentGloss'],
                           'child_dep': json_object['sentences'][0]['basicDependencies'][1:][idx]['dep']}
        return result

    def __set_depth(self, depth=""):
        """
        Need to be used for different prepossess methods
        :return:
        """
        try:
            if depth == 'ml':
                self.preprocess['X'] = np.concatenate([self.preprocess["X"], self.preprocess["X_deps"]], axis=1)
                self.logger.critical("Setting depth ml")
            elif depth == 'm':
                self.preprocess['X'] = np.concatenate([self.preprocess["X"], self.preprocess['X_wordpairs']], axis=1)
                self.logger.info("Setting depth m")
            else:
                self.logger.info("Setting no depth")
        except Exception:
            self.logger.error("Setting depth failed")
            raise

    def __init_preprocess(self, depth):
        self.preprocess = dict()
        self.preprocess["X"] = []
        self.preprocess["X_deps"] = []
        self.preprocess["X_wordpairs"] = []
        self.preprocess['depth'] = depth

    def __get_stanford_core_nlp(self):
        """
        Load StanfordCoreNlp jar to python wrapper
        :return: stanford core nlp wrapper instance
        """
        return StanfordCoreNLP(self.config[DATA_MODULE_CONFIG_SECTION]['nlp_path'])

    def getMaxLength(self, save_stats=False):
        """
        Get stats from data maxlen, deps2ids, ids2deps
        maxlen - the longest sentence or dependencies length
        :param save_stats: True or false for saving stats to pickle
        :type save_stats: Boolean
        """
        self.logger.critical('Getting maxlen')
        maxlen = 0
        deps2ids = {}
        depid = 0
        maxlen_dep = 0
        nlp = self.__get_stanford_core_nlp()
        for idx, sent in enumerate(self.instances):
            if idx % 20 == 0:
                self.logger.critical('Done {0} of {1}'.format(idx, len(self.instances)))
            try:
                sent_maxlen_dep = 0
                doc = nlp.dependency_parse(sent)
                if len(doc) > maxlen:
                    maxlen = len(doc)
                doc_set = set([k[0] for k in doc])
                for token in doc_set:
                    if token not in deps2ids:
                        deps2ids[token] = depid
                        depid += 1
                        sent_maxlen_dep += 1
                if sent_maxlen_dep > maxlen_dep:
                    maxlen_dep = sent_maxlen_dep
            except UnicodeDecodeError:
                self.logger.error('Cant process sentence: ', sent)
        maxlen = max(maxlen, maxlen_dep)
        ids2deps = dict([(idx, dep) for dep, idx in deps2ids.items()])
        self.preprocess["maxlen"] = maxlen
        self.preprocess["deps2ids"] = deps2ids
        self.preprocess["ids2deps"] = ids2deps
        nlp.close()
        nlp = None
        if save_stats:
            self.save_stats(self.name)
        self.logger.critical('Maxlen = {0}'.format(self.preprocess["maxlen"]))

    @staticmethod
    def save_obj(obj, name, path="../bin/"):
        """
        Static method that can save any object to pickle
        :param obj: The object need to be saved
        :param name: Name of saved file
        :param path: path to save the file
        :return: None
        """
        with open(path + name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_obj(name, path="../bin/"):
        """
        Static method that can load any python object
        :param name: name of the file
        :param path: path to the directory of the object
        :return: object from pickle
        """
        with open(path + name + '.pkl', 'rb') as f:
            return pickle.load(f)

    def save_stats(self, name):
        """
        Method that save to pickle ids2deps,deps2ids and maxlen
        :return: None
        """
        self.logger.critical("Saving stats of {0} to file".format(name))
        self.save_obj(self.preprocess['ids2deps'], "{0}_ids2deps".format(name))
        self.save_obj(self.preprocess['deps2ids'], "{0}_deps2ids".format(name))
        self.save_obj(self.preprocess['maxlen'], "{0}_maxlen".format(name))

    def load_all_stats(self, name):
        """
        Method that load from pickle ids2deps,deps2ids and maxlen
        :return: save the self instances
        """
        self.logger.critical("Load stats {0} from file".format(name))
        self.preprocess['ids2deps'] = self.load_obj("{0}_ids2deps".format(name))
        self.preprocess['deps2ids'] = self.load_obj("{0}_deps2ids".format(name))
        self.preprocess['maxlen'] = self.load_obj("{0}_maxlen".format(name))

    @staticmethod
    def pad_words(tokens, maxlen, append_tuple=False):
        """
        Static Method that is part of the Preprocess
        make the sentences arrays same length
        :param tokens: tokens to be modified
        :param maxlen: max length of the longest sentance or dependency length
        :param append_tuple:  Need to be true when tuple need to be append
        :return: return modified tokens
        """
        if len(tokens) > maxlen:
            return tokens[:maxlen]
        else:
            dif = maxlen - len(tokens)
            for i in range(dif):
                if not append_tuple:
                    tokens.append('UNK')
                else:
                    tokens.append(('UNK', 'UNK'))
            return tokens

    def preprocessing_data(self, word2vec, model=False):
        """
        Method that takes all the instances in the data and preprocess them.
        :param word2vec: word2vec that need to be used in the preprocess
        :param model: if model true mean that need to be used maxlen of this model and model dependencies too
        """
        self.logger.critical("Starting to preprocess data {}".format(self.name))
        nlp = self.__get_stanford_core_nlp()
        try:
            self.logger.info("Max length of the data is {0}".format(self.preprocess['maxlen']))
        except KeyError:
            if model:
                self.load_all_stats(self.model_name)
            else:
                self.load_all_stats(self.name)
        for idx, sent in enumerate(self.instances):
            if idx % 100 == 0:
                self.logger.critical('Words done {0} of  {1}'.format(idx, len(self.instances)))
            object_json_data = json.loads(nlp.annotate(sent, properties={'annotators': 'tokenize', 'outputFormat': 'json'}))
            tokens = [k['word'].lower() for k in object_json_data['tokens']]
            sent_matrix = []
            for token in self.pad_words(tokens, self.preprocess['maxlen']):
                if token in word2vec.vocab:
                    # each word vector is embedding dim + length of one-hot encoded label
                    vec = np.concatenate([word2vec.model[token], np.zeros(len(self.preprocess['ids2deps'])+1)])
                    sent_matrix.append(vec)
                else:
                    sent_matrix.append(np.zeros(word2vec.dims + len(self.preprocess['ids2deps']) + 1))
            sent_matrix = np.array(sent_matrix)
            try:
                self.preprocess['X'].append(sent_matrix)
            except AttributeError:
                np.concatenate([self.preprocess['X'], [sent_matrix]])

        for idx, sent in enumerate(self.instances):
            if idx % 10 == 0:
                self.logger.critical('Pairs done {0} of  {1}'.format(idx, len(self.instances)))
            idy = idx
            object_json_data = json.loads(nlp.annotate(sent, properties={'annotators': 'depparse', 'outputFormat': 'json'}))
            try:
                tokens = self.get_pair_words(object_json_data)
            except IndexError:
                pdb.set_trace()
                pass
            word_pairs = []
            dep_pairs = []
            for idx2, tok in tokens.items():
                word_pairs.append((tok['parent_word'], tok['child_word']))
                dep_pairs.append((tok['parent_dep'], tok['child_dep']))
            word_pairs = self.pad_words(word_pairs, self.preprocess['maxlen'], append_tuple=True)
            dep_pairs = self.pad_words(dep_pairs, self.preprocess['maxlen'], append_tuple=True)
            dep_labels = [j for i, j in dep_pairs]
            avg_sent_matrix = []
            avg_label_sent_matrix = []
            for idx, word_pair in enumerate(word_pairs):
                head, modifier = word_pair[0], word_pair[1]
                if head in word2vec.vocab and not head == 'UNK':
                    head_vec = word2vec.model[head]
                else:
                    head_vec = np.zeros(word2vec.dims)
                if modifier in word2vec.vocab and not modifier == 'UNK':
                    modifier_vec = word2vec.model[modifier]
                else:
                    modifier_vec = np.zeros(word2vec.dims)
                avg = np.mean(np.array([head_vec, modifier_vec]), axis=0)
                if dep_labels[idx] != 'UNK':
                    try:
                        dep_idx = self.preprocess['deps2ids'][dep_labels[idx]]
                    except KeyError:
                        dep_idx = -1
                else:
                    dep_idx = -1
                dep_vec = np.zeros(len(self.preprocess['deps2ids']) + 1)
                dep_vec[dep_idx] = 1
                avg_label_vec = np.concatenate([avg, dep_vec])
                avg_sent_matrix.append(np.concatenate([avg, np.zeros(len(self.preprocess['deps2ids']) + 1)]))
                avg_label_sent_matrix.append(avg_label_vec)
            wp = np.array(avg_sent_matrix)
            labs = np.array(avg_label_sent_matrix)
            try:
                self.preprocess['X_wordpairs'].append(wp)
            except AttributeError:
                np.concatenate([self.preprocess['X_wordpairs'], [wp]])
            try:
                self.preprocess['X_deps'].append(labs)
            except AttributeError:
                np.concatenate([self.preprocess['X_deps'], [labs]])
        nlp.close()
        del nlp
        del word2vec
        del self.instances
        try:
            self.logger.critical('2 Garbage Collector: Collected amount {0}'.format(gc.collect()))
            if self.preprocess['X'].__class__ is not np.array([]).__class__:
                self.preprocess['X'] = np.array(self.preprocess['X'])
            if self.preprocess['X_wordpairs'].__class__ is not np.array([]).__class__ and self.preprocess['depth'] == 'm':
                del self.preprocess['X_deps']
                self.preprocess['X_wordpairs'] = np.array(self.preprocess['X_wordpairs'])
            elif self.preprocess['X_deps'].__class__ is not np.array([]).__class__ and self.preprocess['depth'] == 'ml':
                del self.preprocess['X_wordpairs']
                self.preprocess['X_deps'] = np.array(self.preprocess['X_deps'])
            if self.preprocess['depth'] == '':
                del self.preprocess['X_deps']
                del self.preprocess['X_wordpairs']
        except MemoryError:
            pdb.set_trace()
            pass
        self.logger.critical('3 Garbage Collector: Collected amount {0}'.format(gc.collect()))
        self.__set_depth(self.preprocess['depth'])
        self.preprocessed = True
        self.logger.critical("Data prepossess succeeded ")


if __name__ == '__main__':
    logger = LoggerClass()
    word2vec = MyWord2vec()
    data_wcl = DataClass("wcl")
    data_wcl.getMaxLength()
    data_wcl.preprocessing_data(word2vec)
    data_w00 = DataClass("w00")
    data_w00.getMaxLength()
    data_w00.preprocessing_data(word2vec)
    data_wolfram = DataClass("wolfram")
    data_wolfram.getMaxLength()
    data_wolfram.preprocessing_data(word2vec)
    data_all = DataClass("")
    data_all.getMaxLength()
    data_all.preprocessing_data(word2vec)
    pass





