from data.loader import FileIO
from data.data import Data
from util.conf import OptionConf
from util.logger import Log
from os.path import abspath
from time import strftime, localtime, time
from util.algorithm import find_k_largest


class Recommender(object):
    def __init__(self, conf, training_set, test_set):
        self.config = conf
        self.data = Data(self.config, training_set, test_set)
        self.model_name = self.config['model.name']
        self.ranking = OptionConf(self.config['item.ranking'])
        self.emb_size = int(self.config['embbedding.size'])
        self.maxEpoch = int(self.config['num.max.epoch'])
        self.lRate = float(self.config['learnRate'])
        self.reg = float(self.config['reg.lambda'])
        self.output = OptionConf(self.config['output.setup'])
        current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        self.model_log = Log(self.model_name, self.model_name + ' ' + current_time)
        self.result = []
        self.recOutput = []

    def initializing_log(self):
        # save configuration
        self.model_log.add('### model configuration ###')
        for k in self.config.config:
            self.model_log.add(k + '=' + self.config[k])

    def print_model_info(self):
        "show model's configuration"
        # print specific parameters if applicable
        if self.config.contain(self.config['model.name']):
            par_str = ''
            args = OptionConf(self.config[self.config['model.name']])
            for key in args.keys():
                par_str += key[1:] + ':' + args[key] + '  '
            print('Specific parameters:', par_str)
            print('=' * 80)
        print('Model:', self.config['model.name'])
        print('Training set:', abspath(self.config['training.set']))
        print('Test set:', abspath(self.config['test.set']))
        print('Embedding Dimension:', self.emb_size)
        print('Maximum Epoch:', self.maxEpoch)
        print('Regularization parameter: reg %.4f' % self.reg)

    def init(self):
        pass

    def train(self):
        'build the model (for model-based Models )'
        pass

    def predict(self, u):
        pass

    def test(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def evaluate(self,rec_list):
        pass
    # # for rating prediction
    # def predictForRating(self, u, i):
    #     pass

    # for item prediction

    def execute(self):
        self.initializing_log()
        self.print_model_info()
        print('Initializing model...')
        self.init()
        print('Training Model...')
        self.train()
        # rating prediction or item ranking
        print('Testing...')
        rec_list = self.test()
        self.evaluate(rec_list)

