import time

import numpy as np
import argparse

import torch
import torch.distributed as dist
from sklearn.metrics import accuracy_score, roc_auc_score

import sys
sys.path.append("../../")
from data_loader.data_partition import load_dummy_partition_with_label
from tenseal_trainer.lr import LRTrainer


def dist_is_initialized():
    if dist.is_available():
        if dist.is_initialized():
            return True
    return False


def run(args):
    run_start = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # rank 0 is master
    print("rank = {}, world size = {}".format(args.rank, args.world_size))

    file_name = "{}/{}_{}".format(args.root, args.rank, args.world_size)
    print("read file {}".format(file_name))

    load_start = time.time()
    data, targets = load_dummy_partition_with_label(file_name)
    print("load data part cost {} s".format(time.time() - load_start))
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
    print("test data indices: {}".format((shuffle_ind[:n_test])[:10]))
    data = data[shuffle_ind]
    targets = targets[shuffle_ind]

    # normalize data
    data = data / data.max(axis=0)

    train_dataset = data[n_test:]
    train_targets = targets[n_test:]
    test_dataset = data[:n_test]
    test_targets = targets[:n_test]

    trainer = LRTrainer(args)

    train_start = time.time()
    epoch_loss_lst = []
    loss_tol = 0.01
    epoch_tol = 3   # loss should decrease in ${epoch_tol} epochs
    for epoch_idx in range(args.n_epochs):
        print(">>> epoch [{}] start".format(epoch_idx))
        epoch_start = time.time()
        epoch_loss = 0.
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = (batch_idx + 1) * batch_size if batch_idx < n_batches - 1 else n_train
            cur_train = train_dataset[start:end]
            cur_target = train_targets[start:end]
            batch_loss = trainer.one_iteration(epoch_idx, batch_idx, cur_train, cur_target)
            epoch_loss += batch_loss
        epoch_train_time = time.time() - epoch_start
        test_start = time.time()
        pred_targets, pred_probs = trainer.predict(test_dataset)
        # pred_targets = []
        # pred_probs = []
        # true_targets = []
        # for k in range(n_test):
        #     cur_test = test_dataset[k]
        #     test_target = test_targets[k]
        #     true_targets.append(test_target)
        #     pred_target, pred_prob = trainer.predict_one(cur_test)
        #     pred_targets.append(pred_target)
        #     pred_probs.append(pred_prob)
        # print("targets = {}".format(true_targets))
        # print("predictions = {}".format(pred_targets))
        # print("pred probs = {}".format(pred_probs))
        accuracy = accuracy_score(test_targets, pred_targets)
        auc = roc_auc_score(test_targets, np.array(pred_probs))
        epoch_test_time = time.time() - test_start
        print(">>> epoch[{}] finish, train loss {:.6f}, cost {:.2f} s, train cost {:.2f} s, test cost {:.2f} s, "
              "accuracy = {:.6f}, auc = {:.6f}"
              .format(epoch_idx, epoch_loss, time.time() - epoch_start, epoch_train_time, epoch_test_time,
                      accuracy, auc))
        epoch_loss_lst.append(epoch_loss)
        if epoch_idx >= 9 and len(epoch_loss_lst) > epoch_tol \
                and min(epoch_loss_lst[:-epoch_tol]) - min(epoch_loss_lst[-epoch_tol:]) < loss_tol:
            print("!!! train loss does not decrease > {} in {} epochs, early stop !!!"
                  .format(loss_tol, epoch_tol))
            break

    print(">>> task finish, cost {:.2f} s".format(time.time() - run_start))


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
    parser.add_argument('--config', type=str, default='ts_ckks.config')
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
