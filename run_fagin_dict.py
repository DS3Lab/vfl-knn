import time

import numpy as np
import argparse

import torch
import torch.distributed as dist

from utils.file_utils import load_part
from utils.distance import square_euclidean_np
from utils.comm_op import gather
from utils.fagin_utils import count_by_dict, suggest_size

N_FEATURES = 28
N_PARTS = 7
K = 10
RATIO_K = 100


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

    local_dist_ind = np.argsort(local_dist)
    print("local dist index = {}".format(local_dist_ind[:10]))
    print("local dist = {}".format(local_dist[local_dist_ind[:10]]))

    send_size = suggest_size(n_data, args.k, args.world_size)
    send_ind = 0

    comm_start = time.time()
    gather_time = 0
    bc_time = 0
    count_time = 0
    top_k = []
    counts = dict()
    cur_n_top = 0
    n_iter = 0
    while cur_n_top < args.k and send_ind+send_size <= n_data:
        rank = dist.get_rank()
        gather_start = time.time()
        new_lists = gather(local_dist_ind[send_ind:send_ind+send_size])
        gather_time += time.time() - gather_start
        send_ind += send_size
        if rank == 0:
            count_start = time.time()
            count_by_dict(new_lists, counts, top_k, args.k)
            #top_k.extend(new_top_k)
            count_time += time.time() - count_start
            bc_start = time.time()
            cur_n_top = len(top_k)
            dist.broadcast(torch.tensor(cur_n_top), 0)
            bc_time += time.time() - bc_start
            print("iter {}, scan {} rows, current top k = {}".format(n_iter, send_size, cur_n_top))
            n_iter += 1
            dist.barrier()
        else:
            bc_start = time.time()
            tmp_tensor = torch.tensor(0)
            dist.broadcast(tmp_tensor, 0)
            bc_time += time.time() - bc_start
            cur_n_top = tmp_tensor.item()
            print("iter {}, scan {} rows, current top k = {}".format(n_iter, send_size, cur_n_top))
            n_iter += 1
            dist.barrier()

    print("Fagin cost {} s <<< gather cost {} s, broadcast cost {} s, count cost {} s"
          .format(time.time() - comm_start, gather_time, bc_time, count_time))
    print("total number of current top k = {}".format(cur_n_top))
    print("indices of {} near neighbors = {}".format(args.k, top_k[:args.k]))
    print("local dist of {} near neighbors = {}".format(args.k, local_dist[top_k[:args.k]]))
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
