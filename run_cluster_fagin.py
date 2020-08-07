import time

import numpy as np
import argparse

import torch
import torch.distributed as dist
from sklearn.metrics import accuracy_score

from utils.file_utils import load_part_with_label
from utils.distance import square_euclidean_np
from utils.comm_op import gather
from utils.fagin_utils import count_by_arr

N_FEATURES = 28
N_CLASSES = 2
N_PARTS = 7
K = 10
EarlyStopping_PATIENCE = 5
N_TEST = 20
N_CLUSTER = 100


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

    cluster_size = int(n_data / args.n_cluster)
    if n_data % args.n_cluster != 0:
        cluster_size += 1
    print("number of clusters = {}, size of each clusters = {}".format(args.n_cluster, cluster_size))

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
        #print("test data index = {}, value = {}".format(test_ind, test_data))

        local_dist_start = time.time()
        local_dist = square_euclidean_np(dataset, test_data)
        #print("local distance: size = {}, values = {}".format(len(local_dist), local_dist[:10]))
        local_dist_time = time.time() - local_dist_start
        #print("compute local distance cost {} s".format(local_dist_time))

        sort_start = time.time()
        local_dist_ind = np.argsort(local_dist)
        #print("local dist index = {}".format(local_dist_ind[:10]))
        #print("local dist = {}".format(local_dist[local_dist_ind[:10]]))
        sort_time = time.time() - sort_start
        #print("sort local distance cost {} s".format(sort_time))

        cluster_start = time.time()
        # cluster_ind = [0 for _ in range(n_data)]
        # for c_ind in range(n_clusters):
        #     start_ind = c_ind * args.k
        #     end_ind = min(start_ind + args.k, n_data)
        #     for d_ind in range(start_ind, end_ind):
        #         cluster_ind[local_dist_ind[d_ind]] = c_ind
        #
        # local_dist_centroids = []
        # for c_ind in range(n_clusters):
        #     start_ind = c_ind * args.k
        #     end_ind = min(start_ind + args.k, n_data)
        #     mid_ind = int((start_ind + end_ind) / 2)
        #     local_dist_centroids.append(local_dist[local_dist_ind[mid_ind]])
        # #print("number of local dist centroids = {}".format(len(local_dist_centroids)))
        cluster_time = time.time() - cluster_start

        send_size = cluster_size
        send_ind = 0

        comm_start = time.time()
        gather_time = 0
        bc_time = 0
        count_time = 0
        top_k = []
        top_k_local_dist = []
        counts = [0 for _ in range(n_data)]
        cur_n_top = 0
        n_iter = 0
        rank = dist.get_rank()

        while cur_n_top < args.k and send_ind + send_size <= n_data:
            start_ind = send_ind
            end_ind = min(n_data, send_ind + send_size)
            mid_ind = int((start_ind + end_ind) / 2)
            gather_start = time.time()
            new_lists = gather(local_dist_ind[start_ind:end_ind])
            gather_time += time.time() - gather_start
            send_ind += send_size
            if rank == 0:
                count_start = time.time()
                count_by_arr(new_lists, counts, top_k, args.k)
                count_time += time.time() - count_start
                for _ in range(len(top_k) - cur_n_top):
                    top_k_local_dist.append(local_dist[local_dist_ind[mid_ind]])
                bc_start = time.time()
                cur_n_top = len(top_k)
                dist.broadcast(torch.tensor(cur_n_top), 0)
                bc_time += time.time() - bc_start
                # print("iter {}, scan {} rows, current top k = {}".format(n_iter, send_size, cur_n_top))
                n_iter += 1
                # dist.barrier()
            else:
                bc_start = time.time()
                tmp_tensor = torch.tensor(0)
                dist.broadcast(tmp_tensor, 0)
                bc_time += time.time() - bc_start
                new_n_top = tmp_tensor.item()
                for _ in range(new_n_top - cur_n_top):
                    top_k_local_dist.append(local_dist[local_dist_ind[mid_ind]])
                cur_n_top = tmp_tensor.item()
                #print("iter {}, scan {} rows, current top k = {}".format(n_iter, send_size, cur_n_top))
                n_iter += 1
                # dist.barrier()
            print("scan from {} to {}, number of found top {} = {}"
                  .format(start_ind, end_ind, args.k, cur_n_top))

        # broadcast top k
        # bc_top_k_start = time.time()
        # if rank == 0:
        #     tmp_tensor = torch.tensor(top_k, dtype=torch.int32)
        #     dist.broadcast(tmp_tensor, 0)
        # else:
        #     tmp_tensor = torch.zeros(args.k, dtype=torch.int32)
        #     dist.broadcast(tmp_tensor, 0)
        #     top_k = tmp_tensor.numpy().tolist()
        # bc_time += time.time() - bc_top_k_start

        # calculate distance
        sum_dist_start = time.time()
        # top_k_clusters = [cluster_ind[nid] for nid in top_k]
        # top_k_local_dist_np = np.asarray([local_dist_centroids[c_id] for c_id in top_k_clusters])
        # #print("top k local dist = {}".format(top_k_local_dist_np))
        # global_dist = sum_sqrt_all_reduce(top_k_local_dist_np)
        # #print("global distance of top k = {}".format(global_dist))
        sum_dist_time = time.time() - sum_dist_start
        # print("sum local distance cost {} s".format(sum_dist_time))

        # calculate label
        label_count = [0 for _ in range(args.n_classes)]
        for j in top_k:
            label_count[labels[j]] += 1
        pred_label = np.argmax(label_count)
        pred_labels.append(pred_label)
        true_labels.append(labels[test_ind])

        print("indices of {} near neighbors = {}".format(args.k, top_k[:args.k]))
        #print("local dist of {} near neighbors = {}".format(args.k, local_dist[top_k[:args.k]]))
        print("one test finish: test data index = {}, label = {}, prediction = {}, "
              "cost {} s, comp dist cost {} s, sort cost {} s, cluster cost {} s, "
              "sync cost {} s = gather cost {} s + broadcast cost {} s + "
              "count cost {} s + sum top-k distance cost {} s"
              .format(test_ind, labels[test_ind], pred_label, time.time() - new_test_start,
                      local_dist_time, sort_time, cluster_time, time.time() - comm_start,
                      gather_time, bc_time, count_time, sum_dist_time))

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
    parser.add_argument('--n-cluster', type=int, default=N_CLUSTER)
    parser.add_argument('--seed', type=int, default=100)
    args = parser.parse_args()
    print(args)

    if args.world_size > 1:
        dist.init_process_group(backend=args.backend, init_method=args.init_method, world_size=args.world_size, rank=args.rank)

    run(args)


if __name__ == '__main__':
    main()
