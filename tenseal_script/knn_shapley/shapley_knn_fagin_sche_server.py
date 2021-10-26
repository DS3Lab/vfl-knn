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
from tenseal_trainer.knn.shapley_fagin_schedule_trainer import ShapleyFaginScheduleTrainer
from utils.shapley_utils import generate_all_combinations


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
        key = key // 2
    return client_attendance


def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    world_size = args.world_size
    rank = args.rank
    print("rank = {}, world size = {}".format(args.rank, args.world_size))
    file_name = "{}/{}_{}".format(args.root, args.rank, args.world_size)
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

    trainer = ShapleyFaginScheduleTrainer(args, train_data, train_targets)

    # size:
    client_combinations = generate_all_combinations(args.world_size)
    # key -> index
    combination_key_dict = {client_combinations[i]: i for i in range(len(client_combinations))}
    true_targets = []
    shapley_accuracy = []

    shapley_start = time.time()
    print(">>> compute accuracy of all shapley groups")

    # loop over test data, and calculate all combinations of clients
    for i in range(args.n_test):
        print(">>> test round [{}]".format(i))
        one_test_start = time.time()
        cur_test_data = test_data[i]
        cur_test_target = test_targets[i]
        true_targets.append(cur_test_target)

        combination_accuracy = trainer.find_top_k(cur_test_data, cur_test_target, args.k)
        shapley_accuracy.append(combination_accuracy)

        one_test_time = time.time() - one_test_start

        print(">>> shapley value for one test finish: target = {}, accuracy of groups = {}, cost {} s"
              .format(cur_test_target, combination_accuracy, one_test_time))

    print(">>> shapley phase cost {:.2f} s".format(time.time() - shapley_start))

    # test accuracy (of test data) over all client combinations
    test_accuracies = np.mean(np.asarray(shapley_accuracy), axis=0)
    print("test accuracy of all client groups: {}".format(test_accuracies))

    print(">>> compute utility of all groups")
    # calculate utility of all groups, group key = 1~(2^k-1), key is binary encode of client attendance
    start_key = 1
    end_key = int(math.pow(2, args.world_size)) - 1

    # accuracy of each group of clients
    utility_value = dict()
    n_utility_round = 0

    utility_start = time.time()
    for group_key in range(start_key, end_key + 1):
        group_flags = utility_key_to_groups(group_key, world_size)
        client_indices = np.where(np.asarray(group_flags) == 1)[0].tolist()

        combination_key = "_".join(map(str, client_indices))
        #print("client_indices = {}, combination key = {}".format(client_indices, combination_key))
        combination_ind = combination_key_dict[combination_key]

        combination_accuracy = test_accuracies[combination_ind]
        #print("group {}, client combination key {}, client combination index {}, accuracy = {}"
        #      .format(group_flags, combination_key, combination_ind, combination_accuracy))
        print("group {}, accuracy = {}".format(group_flags, combination_accuracy))
        utility_value[group_key] = combination_accuracy
        n_utility_round += 1

    print(">>> utility phase cost {:.2f} s, total round {}".format(time.time() - utility_start, n_utility_round))

    # calculate factorial (n-1)!
    factor = [1] * world_size
    for i in range(1, world_size):
        factor[i] = factor[i - 1] * i

    # shapley value of all clients
    shapley_value = [0.0] * world_size
    n_shapley_round = 0

    # calculate shapley value of each client
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
