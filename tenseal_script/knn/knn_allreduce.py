import time
import sys

import numpy as np
import argparse

import torch
import torch.distributed as dist
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.append("../../")
from data_loader.data_partition import load_dummy_partition_with_label
from tenseal_trainer.knn.all_reduce_trainer import Trainer

N_FEATURES = 28
N_CLASSES = 2
N_PARTS = 7
K = 10
N_TEST = 20


def dist_is_initialized():
    if dist.is_available():
        if dist.is_initialized():
            return True
    return False


def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    file_name = "{}/{}_{}".format(args.root, args.rank, args.world_size)
    print("read file {}".format(file_name))

    load_start = time.time()
    data, targets = load_dummy_partition_with_label(file_name)
    print("load data part cost {} s".format(time.time()-load_start))
    n_data = len(data)
    print("number of data = {}".format(n_data))

    # shuffle the data to split train data and test data
    shuffle_ind = np.arange(n_data)
    np.random.shuffle(shuffle_ind)
    print("test data indices: {}".format(shuffle_ind[:args.n_test]))
    data = data[shuffle_ind]
    targets = targets[shuffle_ind]
    train_dataset = data[args.n_test:]
    test_dataset = data[:args.n_test]
    train_targets = targets[args.n_test:]
    test_targets = targets[:args.n_test]

    trainer = Trainer(args, train_dataset, train_targets)

    pred_targets = []
    pred_probs = []
    true_targets = []

    test_start = time.time()

    for i in range(args.n_test):
        print(">>>>>> test[{}] <<<<<<".format(i))
        one_test_start = time.time()
        test_data = test_dataset[i]
        test_target = test_targets[i]
        true_targets.append(test_target)

        pred_target, pred_prob = trainer.find_top_k(test_data, test_target, args.k)
        pred_targets.append(pred_target)
        pred_probs.append(pred_prob)

        one_test_time = time.time() - one_test_start

        print("one test finish: target = {}, prediction = {}, cost {} s"
              .format(test_target, pred_target, one_test_time))

    print("======finish======")
    print("test {} data cost {:.4f} s".format(args.n_test, time.time() - test_start))
    #print("targets = {}".format(true_targets))
    #print("predictions = {}".format(pred_targets))
    #print("pred probs = {}".format(pred_probs))
    print("accuracy = {:.6f}".format(accuracy_score(true_targets, pred_targets)))
    # two-class
    print("auc = {:.6f}".format(roc_auc_score(true_targets, np.array(pred_probs)[:, 1])))
    # multi-class
    # print("auc = {}".format(roc_auc_score(true_targets, np.array(pred_probs), multi_class="ovr")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', type=str, default='gloo', help='Name of the backend to use.')
    parser.add_argument(
        '-i',
        '--init-method',
        type=str,
        default='tcp://127.0.0.1:23456',
        help='URL specifying how to initialize the package.')
    parser.add_argument('-s', '--world-size', type=int, default=N_PARTS, help='Number of processes participating in the job.')
    parser.add_argument('-r', '--rank', type=int, default=0, help='Rank of the current process.')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--config', type=str, default='ts_ckks.config')
    parser.add_argument('--n-features', type=int, default=N_FEATURES)
    parser.add_argument('--n-classes', type=int, default=N_CLASSES)
    parser.add_argument('--k', type=int, default=K)
    parser.add_argument('--n-test', type=int, default=N_TEST)
    parser.add_argument('--seed', type=int, default=100)
    args = parser.parse_args()
    print(args)

    if args.world_size > 1:
        dist.init_process_group(backend=args.backend, init_method=args.init_method, world_size=args.world_size, rank=args.rank)

    run(args)


if __name__ == '__main__':
    main()
