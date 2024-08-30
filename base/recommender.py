from data.data import Data
from util.logger import Log
from os.path import abspath
from time import strftime, localtime, time


class Recommender:
    def __init__(self, conf, training_set, test_set, **kwargs):
        self.config = conf
        self.data = Data(self.config, training_set, test_set)

        # Store references in class attributes for repeated use
        model_config = self.config['model']
        self.model_name = model_config['name']
        self.ranking = self.config['item.ranking.topN']
        self.emb_size = int(self.config['embedding.size'])
        self.maxEpoch = int(self.config['max.epoch'])
        self.batch_size = int(self.config['batch.size'])
        self.lRate = float(self.config['learning.rate'])
        self.reg = float(self.config['reg.lambda'])
        self.output = self.config['output']

        # Use f-string for better readability
        current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        self.model_log = Log(self.model_name, f"{self.model_name} {current_time}")

        self.result = []
        self.recOutput = []

    def initializing_log(self):
        self.model_log.add('### model configuration ###')
        config_items = self.config.config
        for k in config_items:
            self.model_log.add(f"{k}={str(config_items[k])}")

    def print_model_info(self):
        print('Model:', self.model_name)
        print('Training Set:', abspath(self.config['training.set']))
        print('Test Set:', abspath(self.config['test.set']))
        print('Embedding Dimension:', self.emb_size)
        print('Maximum Epoch:', self.maxEpoch)
        print('Learning Rate:', self.lRate)
        print('Batch Size:', self.batch_size)
        print('Regularization Parameter:', self.reg)

        model_name = self.config['model']['name']
        if self.config.contain(model_name):
            args = self.config[model_name]
            parStr = '  '.join(f"{key}:{args[key]}" for key in args)
            print('Specific parameters:', parStr)

    def build(self):
        pass

    def train(self):
        pass

    def predict(self, u):
        pass

    def test(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def evaluate(self, rec_list):
        pass

    def execute(self):
        self.initializing_log()
        self.print_model_info()
        print('Initializing and building model...')
        self.build()
        print('Training Model...')
        self.train()
        print('Testing...')
        rec_list = self.test()
        print('Evaluating...')
        self.evaluate(rec_list)
