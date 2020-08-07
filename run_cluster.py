import time
import math

import numpy as np
import argparse

import torch
import torch.distributed as dist
from sklearn.metrics import accuracy_score

from utils.file_utils import load_part_with_label
from utils.distance import square_euclidean_np
from utils.comm_op import sum_all_reduce

N_FEATURES = 28
N_CLASSES = 2
N_PARTS = 7
K = 10
EarlyStopping_PATIENCE = 5
N_TEST = 20


def dist_is_initialized():
    if dist.is_available():
        if dist.is_initialized():
            return True
    return False


def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    torch.manual_seed(args.seed)

    file_name = "{}/{}_{}".format(args.root, args.rank, args.world_size)

    # average number of features in one part
    n_f = 0
    if args.max_features % args.world_size == 0:
        n_f = int(args.max_features / args.world_size)
    else:
        n_f = int((args.max_features - args.max_features % args.world_size) / (args.world_size - 1))

    # local feature index
    start_f = args.rank * n_f
    end_f = min(args.max_features, (args.rank + 1) * n_f)
    print("local features range = [{},{})".format(start_f, end_f))

    load_start = time.time()
    dataset, labels = load_part_with_label(file_name)
    print("load data part cost {} s".format(time.time()-load_start))

    n_data = len(dataset)
    print("number of data = {}".format(n_data))

    cluster_size = args.k
    n_clusters = math.ceil(1.0 * n_data / cluster_size)
    print("number of clusters = {}".format(n_clusters))

    np.random.seed(args.seed)

    test_inds = []
    pred_labels = []
    true_labels = []

    test_start = time.time()

    for i in range(args.n_test):
        print("======test[{}]======".format(i))
        new_test_start = time.time()
        test_ind = np.random.randint(low=0, high=n_data, size=1)[0]
        test_inds.append(test_ind)
        test_data = dataset[test_ind]
        print("test data index = {}, value = {}".format(test_ind, test_data))
        # local_test_data = test_data[start_f:end_f]

        local_dist_start = time.time()
        local_dist = square_euclidean_np(dataset, test_data)
        print("local distance: size = {}, values = {}".format(len(local_dist), local_dist[:10]))
        print("compute local distance cost {} s".format(time.time() - local_dist_start))

        sort_start = time.time()
        local_dist_ind = np.argsort(local_dist)
        print("local dist index = {}".format(local_dist_ind[:10]))
        print("local dist = {}".format(local_dist[local_dist_ind[:10]]))
        print("sort local distance cost {} s".format(time.time() - sort_start))

        step = 0
        cur_index = 0
        stable_step = 0
        cur_label = 0
        label_count = np.zeros(args.n_classes)

        comm_start = time.time()
        all_reduce_time = 0

        while stable_step < EarlyStopping_PATIENCE and step < n_clusters:
            batch_ind = local_dist_ind[cur_index:cur_index + cluster_size]

            cur_index += cluster_size
            step += 1

            batch_label = np.asarray(labels[batch_ind]).flatten()
            batch_label_count = np.bincount(batch_label)
            # in case there is only label 0
            if batch_label_count.size == 1:
                batch_label_count = np.concatenate((batch_label_count, [0]))
            print("batch label count = {}".format(batch_label_count))
            sum_start = time.time()
            sum_batch_count = sum_all_reduce(batch_label_count)
            all_reduce_time += time.time() - sum_start

            label_count += sum_batch_count
            print("global label count = {}".format(label_count))
            new_label = np.argmax(label_count)
            print("new label = {}".format(new_label))
            if new_label == cur_label:
                stable_step += 1
            else:
                cur_label = new_label
                stable_step = 0

        pred_labels.append(cur_label)
        true_labels.append(labels[test_ind])
        print("one test finish: test data index = {}, label = {}, prediction = {}, "
              "cost {} s, sync cost {} s, allreduce cost {} s"
              .format(test_ind, labels[test_ind], cur_label, time.time() - new_test_start,
                      time.time() - comm_start, all_reduce_time))

    print("test {} data cost {} s".format(args.n_test, time.time() - test_start))
    print("test indexes = {}".format(test_inds))
    print("predictions = {}".format(pred_labels))
    print("labels = {}".format(true_labels))
    print("accuracy = {}".format(accuracy_score(true_labels, pred_labels)))


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
    parser.add_argument('--max-features', type=int, default=N_FEATURES)
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
