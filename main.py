from SELFRec import SELFRec
from util.conf import ModelConf

if __name__ == '__main__':
    # Register your model here
    baseline = ['LightGCN','DirectAU','MF']
    graph_models = ['SGL', 'SimGCL', 'SEPT', 'MHCN', 'BUIR', 'SelfCF', 'SSL4Rec', 'XSimGCL', 'NCL','MixGCF']
    sequential_models = []

    print('=' * 80)
    print('   SELFRec: A library for self-supervised recommendation.   ')
    print('=' * 80)

    print('Baseline Models:')
    print('   '.join(baseline))
    print('-' * 80)
    print('Graph-Based Models:')
    print('   '.join(graph_models))

    print('=' * 80)
    model = input('Please enter the model you want to run:')
    import time

    s = time.time()
    if model in baseline or model in graph_models or model in sequential_models:
        conf = ModelConf('./conf/' + model + '.conf')
    else:
        print('Wrong model name!')
        exit(-1)
    rec = SELFRec(conf)
    rec.execute()
    e = time.time()
    print("Running time: %f s" % (e - s))
