import time

import numpy as np
import argparse

import torch
import torch.distributed as dist

from utils.file_utils import load_part
from utils.distance import square_euclidean_np
from utils.comm_op import sum_all_gather

N_FEATURES = 28
N_PARTS = 7
K = 10


def dist_is_initialized():
    if dist.is_available():
        if dist.is_initialized():
            return True
    return False


def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    torch.manual_seed(1234)

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
    dataset = load_part(file_name)
    print("load data part cost {} s".format(time.time()-load_start))

    n_data = len(dataset)
    print("number of data = {}".format(n_data))

    np.random.seed(args.seed)
    test_ind = np.random.randint(low=0, high=n_data, size=1)[0]
    test_data = dataset[test_ind]
    print("test data index = {}, value = {}".format(test_ind, test_data))
    #local_test_data = test_data[start_f:end_f]

    comp_start = time.time()
    local_dist = square_euclidean_np(dataset, test_data)
    print("local distance size = {}, values = {}".format(len(local_dist), local_dist[:10]))
    print("compute distance cost {} s".format(time.time()-comp_start))

    comm_start = time.time()
    global_dist = sum_all_gather(local_dist)
    print("global distance size = {}, values = {}".format(len(global_dist), global_dist[:10]))
    print("communication cost {} s".format(time.time() - comm_start))

    select_start = time.time()
    #ind_k = least_indices(global_dist, args.k)
    ind_k = np.argsort(global_dist)[:args.k]
    dist_k = global_dist[ind_k]
    print("indices of k near neighbor = {}".format(ind_k))
    print("distance of k near neighbor = {}".format(dist_k))
    print("select k minimal cost {} s".format(time.time() - select_start))

    print("total time cost = {}".format(time.time() - load_start))


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
    parser.add_argument('--k', type=int, default=K)
    parser.add_argument('--seed', type=int, default=100)
    args = parser.parse_args()
    print(args)

    if args.world_size > 1:
        dist.init_process_group(backend=args.backend, init_method=args.init_method, world_size=args.world_size, rank=args.rank)

    run(args)


if __name__ == '__main__':
    main()
