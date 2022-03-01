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
from trainer.lr_shapley.shapley_lr_trainer_no_encrypt import ShapleyLRTrainer


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

    file_name = "{}/{}_{}".format(args.root, rank, world_size)
    print("read file {}".format(file_name))

    load_start = time.time()
    data, targets = load_dummy_partition_with_label(file_name)
    print("load data part cost {} s".format(time.time()-load_start))
    n_data = len(data)
    print("number of data = {}".format(n_data))
    n_test = int(n_data * args.valid_ratio)
    n_train = n_data - n_test
    batch_size = args.batch_size
    n_batches = n_train // batch_size
    print("number of train data = {}, number of validation data = {}, batch size = {}, num batches = {}"
          .format(n_train, n_test, batch_size, n_batches))

    # shuffle the data to split train data and test data
    shuffle_ind = np.arange(n_data)
    np.random.shuffle(shuffle_ind)
    data = data[shuffle_ind]
    targets = targets[shuffle_ind]

    train_data = data[n_test:]
    train_targets = targets[n_test:]
    test_data = data[:n_test]
    test_targets = targets[:n_test]

    # accuracy of a group of clients, key is binary encode of client attendance
    utility_value = dict()
    n_utility_round = 0

    # cal utility of all groups, group key = 1-(2^k-1)
    start_key = 1
    end_key = int(math.pow(2, args.world_size)) - 1
    utility_start = time.time()
    n_utility_epochs = 0  # total used epochs
    for group_key in range(start_key, end_key + 1):
        group_flags = utility_key_to_groups(group_key, world_size)
        print("--- compute utility of group : {} ---".format(group_flags))

        group_start = time.time()

        trainer = ShapleyLRTrainer(args, group_flags)
        epoch_loss_lst = []
        loss_tol = 0.001
        epoch_tol = 3  # loss should decrease in ${epoch_tol} epochs
        accuracy, auc = 0.0, 0.0
        for epoch_idx in range(args.n_epochs):
            #print(">>> epoch [{}] start".format(epoch_idx))
            epoch_start = time.time()
            epoch_loss = 0.
            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = (batch_idx + 1) * batch_size if batch_idx < n_batches - 1 else n_train
                cur_train = train_data[start:end]
                cur_target = train_targets[start:end]
                batch_loss = trainer.one_iteration(epoch_idx, batch_idx, cur_train, cur_target)
                epoch_loss += batch_loss
            epoch_train_time = time.time() - epoch_start
            test_start = time.time()
            pred_targets, pred_probs = trainer.predict(test_data)

            accuracy = accuracy_score(test_targets, pred_targets)
            auc = roc_auc_score(test_targets, np.array(pred_probs))
            epoch_test_time = time.time() - test_start
            #print(">>> epoch[{}] finish, train loss {:.6f}, cost {:.2f} s, train cost {:.2f} s, test cost {:.2f} s, "
            #      "accuracy = {:.6f}, auc = {:.6f}"
            #      .format(epoch_idx, epoch_loss, time.time() - epoch_start, epoch_train_time, epoch_test_time, accuracy, auc))
            epoch_loss_lst.append(epoch_loss)
            if epoch_idx >= 9 and len(epoch_loss_lst) > epoch_tol \
                    and min(epoch_loss_lst[:-epoch_tol]) - min(epoch_loss_lst[-epoch_tol:]) < loss_tol:
                print("!!! train loss does not decrease > {} in {} epochs, early stop !!!"
                      .format(loss_tol, epoch_tol))
                break

        n_utility_epochs += epoch_idx + 1

        utility_value[group_key] = accuracy
        n_utility_round += 1
        print("compute utility of group {} cost {:.2f} s".format(group_flags, time.time() - group_start))

    print("calculate utility cost {:.2f} s, total round {}, total epochs {}"
          .format(time.time() - utility_start, n_utility_round, n_utility_epochs))

    group_acc_sum = [0 for _ in range(args.world_size)]
    for group_key in range(start_key, end_key + 1):
        group_flags = utility_key_to_groups(group_key, world_size)
        n_participant = sum(group_flags)
        group_acc_sum[n_participant - 1] += utility_value[group_key]
        print("group {}, accuracy = {}".format(group_flags, utility_value[group_key]))
    print("accuracy sum of different size: {}".format(group_acc_sum))

    # cal factorial
    factor = [1] * args.world_size
    for epoch_idx in range(1, args.world_size):
        factor[epoch_idx] = factor[epoch_idx - 1] * epoch_idx

    # shapley value of all clients
    shapley_value = [0.0] * world_size
    n_shapley_round = 0

    # cal shapley value of each
    shapley_start = time.time()
    for epoch_idx in range(world_size):
        score = 0.0
        # loop all possible groups including the current client
        start_key = 1
        end_key = int(math.pow(2, world_size)) - 1
        for group_key in range(start_key, end_key + 1):
            group_flags = utility_key_to_groups(group_key, world_size)
            group_size = sum(group_flags)
            # the current client is in the group
            if group_flags[epoch_idx] == 1 and group_size > 1:
                u_with = utility_value[group_key]
                group_flags[epoch_idx] = 0
                group_key = get_utility_key(group_flags)
                u_without = utility_value[group_key]
                score += factor[group_size - 1] / float(factor[world_size - 1]) * (u_with - u_without)
        score /= float(math.pow(2, world_size - 1))
        shapley_value[epoch_idx] = score
        n_shapley_round += 1
    print("calculate shapley value cost {:.2f} s".format(time.time() - shapley_start))
    print("shapley value of {} clients: {}".format(len(shapley_value), shapley_value))

    shapley_ind = np.argsort(np.array(shapley_value))
    print("client ranking = {}".format(shapley_ind.tolist()[::-1]))


N_FEATURES = 28
N_CLASSES = 2
N_PARTS = 7
N_EPOCHS = 1
BATCH_SIZE = 100
VALID_RATIO = 0.1
LEARNING_RATE=0.1
LAMBDA = 0.01


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
    parser.add_argument('--n-epochs', type=int, default=N_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--valid-ratio', type=float, default=VALID_RATIO)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--lam', type=float, default=LAMBDA)
    parser.add_argument('--seed', type=int, default=100)
    args = parser.parse_args()
    print(args)

    if args.world_size > 1:
        dist.init_process_group(backend=args.backend, init_method=args.init_method, world_size=args.world_size, rank=args.rank)

    run(args)


if __name__ == '__main__':
    main()
    # key = 7
    # world_size = 5
    # print(utility_key_to_groups(key, world_size))
