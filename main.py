import argparse
import pickle
import time
import numpy as np
import torch
from model4 import AttenMixer, train_test
from utils4 import Data, split_validation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='yoochoose1_64', help='yoochoose1_64/diginetica/Gowalla/LastFM')
    parser.add_argument('--batchSize', type=int, default=100)
    parser.add_argument('--hiddenSize', type=int, default=100)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_dc', type=float, default=0.5)
    parser.add_argument('--lr_dc_step', type=int, default=3)
    parser.add_argument('--l2', type=float, default=1e-5)
    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--valid_portion', type=float, default=0.1)
    parser.add_argument('--len_max', type=int, default=70)
    opt = parser.parse_args()
    print(opt)

    # Загрузка данных
    train_data = pickle.load(open(f'datasets/{opt.dataset}/train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open(f'datasets/{opt.dataset}/test.txt', 'rb'))

    train_data = Data(train_data, opt.len_max, shuffle=True)
    test_data = Data(test_data, opt.len_max, shuffle=False)

    # Определение n_node
    n_node = max(max(seq) for seq in train_data.inputs + test_data.inputs) + 1
    print(f'n_node = {n_node}')

    # Модель
    model = AttenMixer(opt, n_node).cuda() if torch.cuda.is_available() else AttenMixer(opt, n_node)

    # Обучение
    best_hit, best_mrr = 0.0, 0.0
    for epoch in range(opt.epoch):
        print(f'\n=== Epoch {epoch}/{opt.epoch - 1} ===')
        hit, mrr = train_test(model, train_data, test_data, epoch)
        print(f'Hit@20: {hit:.4f} | MRR@20: {mrr:.4f}')
        if hit > best_hit:
            best_hit = hit
        if mrr > best_mrr:
            best_mrr = mrr

    print(f'\nBest Hit@20: {best_hit:.4f}')
    print(f'Best MRR@20: {best_mrr:.4f}')

if __name__ == '__main__':
    main()