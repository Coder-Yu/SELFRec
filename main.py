from SELFRec import SELFRec
from util.conf import ModelConf
import time

def print_models(title, models):
    print(f"{'=' * 80}\n{title}\n{'-' * 80}")
    for category, model_list in models.items():
        print(f"{category}:\n   {'   '.join(model_list)}\n{'-' * 80}")

if __name__ == '__main__':
    models = {
        'Graph-Based Baseline Models': ['LightGCN', 'DirectAU', 'MF'],
        'Self-Supervised Graph-Based Models': ['SGL', 'SimGCL', 'SEPT', 'MHCN', 'BUIR', 'SelfCF', 'SSL4Rec', 'XSimGCL', 'NCL', 'MixGCF'],
        'Sequential Baseline Models': ['SASRec'],
        'Self-Supervised Sequential Models': ['CL4SRec', 'BERT4Rec']
    }

    print('=' * 80)
    print('   SELFRec: A library for self-supervised recommendation.   ')
    print_models("Available Models", models)

    model = input('Please enter the model you want to run:')

    s = time.time()
    all_models = sum(models.values(), [])
    if model in all_models:
        conf = ModelConf(f'./conf/{model}.yaml')
        rec = SELFRec(conf)
        rec.execute()
        e = time.time()
        print(f"Running time: {e - s:.2f} s")
    else:
        print('Wrong model name!')
        exit(-1)
