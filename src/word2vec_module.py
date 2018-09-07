import configparser
import gensim
import pickle
import logging
import sys

# logger for word2vec_module
module_logger = logging.getLogger('auto_de.word2vecModule')
# SECTION NAME in config file
WORD2VEC_CONFIG_SECTION = "word2vec"


class MyWord2vec(object):
    def __init__(self, manual_path_name=None, train_temp_word2vec = False):
        """
        Class that hold the word2vec models from config file or manual input
        :param manual_path_name: Manual option for using another(not default) word2vec model
        :param train_temp_word2vec: Create word2vec for tests
        """
        self.logger = logging.getLogger('auto_de.word2vecModule.MyWord2vec')
        self.config = None
        self.logger.info("Reading configuration from config file")
        self.__init_config()
        if manual_path_name is not None:
            self.path_and_name = manual_path_name
        else:
            if self.config[WORD2VEC_CONFIG_SECTION]["path"][-1:] == '/':
                self.path_and_name = "{0}{1}".format(self.config[WORD2VEC_CONFIG_SECTION]["path"],
                                                     self.config[WORD2VEC_CONFIG_SECTION]["name"])
            else:
                self.path_and_name = "{0}/{1}".format(self.config[WORD2VEC_CONFIG_SECTION]["path"],
                                                      self.config[WORD2VEC_CONFIG_SECTION]["name"])
        self.logger.critical("init word2vec {0}".format(self.path_and_name))
        self.model = None
        self.vocab = None
        self.dims = None
        if not train_temp_word2vec:
            self.__load_embeddings()

    def __load_embeddings(self):
        """
        Method that trying to load the word2vec in different formats
        :return: init self.vocab , self.dims , self.model
        """
        try:
            self.logger.info("Loading word2vec from path {0}".format(self.path_and_name))
            self.model = gensim.models.Word2Vec.load(self.path_and_name)
        except:
            try:
                self.logger.info("load failed")
                self.model = gensim.models.KeyedVectors.load_word2vec_format(self.path_and_name)
            except:
                try:
                    self.logger.info("load_word2vec_format failed")
                    self.model = gensim.models.KeyedVectors.load_word2vec_format(self.path_and_name, binary=True)
                except:
                    try:
                        self.logger.info("load_word2vec_format_binary failed")
                        with open(self.path_and_name, 'rb') as handle:
                            self.model = pickle.load(handle)
                    except:
                        self.logger.error("Couldn't load pickle")
                        raise Exception
        self.logger.critical("load word2vec succeeded")
        try:
            self.vocab = self.model.index2word
        except AttributeError:
            try:
                self.vocab = self.model.wv.index2word
            except:
                self.logger.error("Can't create vocab")
                sys.exit()
        self.logger.info("vocab init succeeded")
        self.dims = self.model.__getitem__(self.vocab[0]).shape[0]
        self.logger.info("dims init succeeded")
        self.vocab = set(self.vocab)

    def __init_config(self, path="configuration/config.ini"):
        """
        Read config file
        :param path: path to config file
        :return: None
        """
        self.config = configparser.ConfigParser()
        self.config.read(path)

    @staticmethod
    def train_word_2_vec():
        """
        Class that train word2vec on the instances from DataClass
        :return: Save the model as a pickle
        """
        from src.DataModule import DataClass
        data_all = DataClass("")
        sentences = data_all.word2vec_temp()
        model = gensim.models.Word2Vec(sentences, min_count=1)
        with open('../bin/word2vec.pickle', 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    # logger = LoggerClass()
    word2vec = MyWord2vec()
    pass

