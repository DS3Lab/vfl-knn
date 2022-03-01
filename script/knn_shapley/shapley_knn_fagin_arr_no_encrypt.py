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
from trainer.knn_shapley.shapley_fagin_arr_exact_trainer_no_encrypt import FaginArrTrainer


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

    world_size = args.world_size
    rank = args.rank
    print("rank = {}, world size = {}".format(rank, world_size))

    file_name = "{}/{}_{}".format(args.root, rank, world_size)
    print("read file {}".format(file_name))

    load_start = time.time()
    data, targets = load_dummy_partition_with_label(file_name)
    print("load data part cost {} s".format(time.time() - load_start))
    n_data = len(data)
    print("number of data = {}".format(n_data))

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

    # accuracy of a group of clients, key is binary encode of client attendance
    utility_value = dict()
    n_utility_round = 0

    trainer = FaginArrTrainer(args, train_data, train_targets)

    # cal utility of all groups, group key = 1-(2^k-1)
    start_key = 1
    end_key = int(math.pow(2, args.world_size)) - 1
    utility_start = time.time()
    for group_key in range(start_key, end_key + 1):
        group_flags = utility_key_to_groups(group_key, world_size)
        print("--- compute utility of group: {} ---".format(group_flags))

        pred_targets = []
        pred_probs = []
        true_targets = []

        test_start = time.time()

        for i in range(args.n_test):
            #print(">>>>>> test[{}] <<<<<<".format(i))
            one_test_start = time.time()
            cur_test_data = test_data[i]
            cur_test_target = test_targets[i]
            true_targets.append(cur_test_target)

            pred_target, pred_prob = trainer.find_top_k(cur_test_data, cur_test_target, args.k, group_flags)
            pred_targets.append(pred_target)
            pred_probs.append(pred_prob)

            one_test_time = time.time() - one_test_start

            #print("one test finish: target = {}, prediction = {}, cost {} s"
            #      .format(cur_test_target, pred_target, one_test_time))

        print("test {} data cost {} s".format(args.n_test, time.time() - test_start))
        # print("targets = {}".format(true_targets))
        # print("predictions = {}".format(pred_targets))
        # print("pred probs = {}".format(pred_probs))
        accuracy = accuracy_score(true_targets, pred_targets)
        # two-class
        # auc = roc_auc_score(true_targets, np.array(pred_probs)[:, 1])
        # multi-class
        auc = roc_auc_score(true_targets, np.array(pred_probs), multi_class="ovr")
        print("accuracy = {}, auc = {}".format(accuracy, auc))

        utility_value[group_key] = accuracy
        n_utility_round += 1

    print("calculate utility cost {:.2f} s, total round {}".format(time.time() - utility_start, n_utility_round))

    group_acc_sum = [0 for _ in range(args.world_size)]
    for group_key in range(start_key, end_key + 1):
        group_flags = utility_key_to_groups(group_key, world_size)
        n_participant = sum(group_flags)
        group_acc_sum[n_participant - 1] += utility_value[group_key]
        print("group {}, accuracy = {}".format(group_flags, utility_value[group_key]))
    print("accuracy sum of different size: {}".format(group_acc_sum))

    # cal factorial
    factor = [1] * args.world_size
    for i in range(1, args.world_size):
        factor[i] = factor[i - 1] * i

    # shapley value of all clients
    shapley_value = [0.0] * world_size
    n_shapley_round = 0

    # cal shapley value of each
    shapley_start = time.time()
    for i in range(world_size):
        score = 0.0
        # loop all possible groups including the current client
        start_key = 1
        end_key = int(math.pow(2, world_size)) - 1
        for group_key in range(start_key, end_key + 1):
            group_flags = utility_key_to_groups(group_key, world_size)
            group_size = sum(group_flags)
            # the current client is in the group
            if group_flags[i] == 1 and group_size > 1:
                u_with = utility_value[group_key]
                group_flags[i] = 0
                group_key = get_utility_key(group_flags)
                u_without = utility_value[group_key]
                score += factor[group_size - 1] / float(factor[world_size - 1]) * (u_with - u_without)
        score /= float(math.pow(2, world_size - 1))
        shapley_value[i] = score
        n_shapley_round += 1
    print("calculate shapley value cost {:.2f} s".format(time.time() - shapley_start))
    print("shapley value of {} clients: {}".format(len(shapley_value), shapley_value))

    shapley_ind = np.argsort(np.array(shapley_value))
    print("client ranking = {}".format(shapley_ind.tolist()[::-1]))


N_FEATURES = 28
N_CLASSES = 2
N_PARTS = 7
K = 10
N_TEST = 20


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
    parser.add_argument('--seed', type=int, default=100)
    args = parser.parse_args()
    print(args)

    if args.world_size > 1:
        dist.init_process_group(backend=args.backend, init_method=args.init_method, world_size=args.world_size, rank=args.rank)

    run(args)


if __name__ == '__main__':
    main()
