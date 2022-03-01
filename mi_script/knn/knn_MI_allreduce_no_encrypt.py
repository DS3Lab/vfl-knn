import time
import sys
import math
import numpy as np
import argparse

import torch
import torch.distributed as dist
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.append("../../")
from data_loader.data_partition import load_dummy_partition_with_label
from mi_trainer.knn.allreduce_trainer_no_encrypt import AllReduceTrainer

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


def get_utility_key(client_attendance):
    key = 0
    for i in reversed(client_attendance):
        key = 2 * key + i
    return key


def utility_key_to_groups(key, world_size):
    client_attendance = [0] * world_size
    for i in range(world_size):
        flag = key % 2
        client_attendance[i] = flag
        key = key //2
    return client_attendance


def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print("device = {}".format(device))

    world_size = args.world_size
    rank = args.rank

    file_name = "{}/{}_{}".format(args.root, args.rank, args.world_size)
    print("read file {}".format(file_name))

    load_start = time.time()
    data, targets = load_dummy_partition_with_label(file_name)
    print("load data part cost {} s".format(time.time()-load_start))
    n_data = len(data)
    print("number of data = {}".format(n_data))
    n_class = args.n_classes
    print("number of class = {}".format(n_class))

    # shuffle the data to split train data and test data
    shuffle_ind = np.arange(n_data)
    np.random.shuffle(shuffle_ind)
    print("test data indices: {}".format(shuffle_ind[:args.n_test]))
    data = data[shuffle_ind]
    targets = targets[shuffle_ind]

    train_data = data[args.n_test:]
    train_targets = targets[args.n_test:]
    test_data = data[:args.n_test]
    test_targets = targets[:args.n_test]

    pred_targets = []
    pred_probs = []
    true_targets = []

    trainer = AllReduceTrainer(args, train_data, train_targets)

    test_start = time.time()

    # for each consortium:
    #   for each label:
    #       for each valid instance:
    #           the coordinator runs fagin, and the leader checks the label
    #           the leader decides whether to stop fagin when find k-nearest
    #           the server aggregates distance
    #           the leader calculates the number of this label and all samples
    #           calculate I for this label
    #   calculate average I for all labels

    client_mi_values = np.zeros(world_size)

    n_group_test = args.n_group_test

    # cal utility of all groups, group key = 1-(2^k-1)
    start_key = 1
    end_key = int(math.pow(2, args.world_size)) - 1
    group_key_ind = np.arange(end_key)
    np.random.shuffle(group_key_ind)
    sampled_group_key = group_key_ind[:min(n_group_test, end_key)]

    for group_key in sampled_group_key:
        group_start = time.time()
        group_flags = utility_key_to_groups(group_key, world_size)
        print("--- compute mutual info of group : {} ---".format(group_flags))
        for i in range(args.n_test):
            one_test_start = time.time()
            cur_test_data = test_data[i]
            cur_test_target = test_targets[i]
            true_targets.append(cur_test_target)

            mi_value = trainer.find_top_k(cur_test_data, cur_test_target, args.k, group_flags)
            client_mi_values += np.array(group_flags) * mi_value

        print("group {} finish: cost {} s, mutual info of clients = {}"
              .format(group_flags, time.time() - group_start, client_mi_values))

    mi_sort_ind = np.argsort(client_mi_values)[::-1]
    print("test {} data cost {} s, client mi ranking = {}, values = {}"
          .format(args.n_test, time.time() - test_start,
                  mi_sort_ind, client_mi_values[mi_sort_ind]))


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
    parser.add_argument('--n-features', type=int, default=N_FEATURES)
    parser.add_argument('--n-classes', type=int, default=N_CLASSES)
    parser.add_argument('--k', type=int, default=K)
    parser.add_argument('--n-test', type=int, default=N_TEST)
    parser.add_argument('--n-group-test', type=int, default=10)
    parser.add_argument('--seed', type=int, default=100)
    args = parser.parse_args()
    print(args)

    if args.world_size > 1:
        dist.init_process_group(backend=args.backend, init_method=args.init_method, world_size=args.world_size, rank=args.rank)

    run(args)


if __name__ == '__main__':
    main()
