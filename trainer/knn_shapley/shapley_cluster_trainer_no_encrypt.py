import time
import sys
import math
import random

import numpy as np
import torch
import torch.distributed as dist

sys.path.append("../")
from utils.distance import square_euclidean_np, least_indices
from utils.comm_op import gather, sum_sqrt_all_reduce
from utils.fagin_utils import suggest_size, master_count_by_arr


class ClusterTrainer(object):

    def __init__(self, args, data, targets):
        self.args = args
        self.data = data
        self.n_data = len(data)
        self.targets = targets

        self.shuffle_seed = 42
        random.seed(self.shuffle_seed)

        shuffle_index_start = time.time()
        # index: original id -> value: shuffled id
        self.origin2shuffled = np.arange(self.n_data)
        np.random.shuffle(self.origin2shuffled)
        # index: shuffle id -> value: original id
        self.shuffled2original = np.zeros_like(self.origin2shuffled)
        for i in range(len(self.origin2shuffled)):
            original_id = i
            shuffled_id = self.origin2shuffled[i]
            self.shuffled2original[shuffled_id] = original_id
        shuffle_index_time = time.time() - shuffle_index_start
        print("shuffle index cost {} ms".format(shuffle_index_time))

    def find_top_k(self, test_data, test_target, k, n_cluster, group_flags):
        start_time = time.time()
        print(">>> start find top-{} <<<".format(k))

        # average number of features in one part
        n_f = int(self.args.n_features / self.args.world_size)
        if self.args.n_features % self.args.world_size != 0:
            n_f += 1

        # local feature range
        start_f = self.args.rank * n_f
        end_f = min(self.args.n_features, (self.args.rank + 1) * n_f)
        print("local features range = [{},{})".format(start_f, end_f))

        local_dist_start = time.time()
        local_dist = square_euclidean_np(self.data, test_data)
        local_dist_time = time.time() - local_dist_start

        sort_start = time.time()
        local_dist_ind = np.argsort(local_dist)
        sort_time = time.time() - sort_start

        shuffle_index_time = 0
        reindex_time = 0
        gather_time = 0
        bc_time = 0
        count_time = 0
        top_k = []

        counts = [0 for _ in range(self.n_data)]
        cur_n_top = 0
        n_iter = 0
        rank = dist.get_rank()

        # use pre-shuffled id
        shuffle_index_start = time.time()
        if rank != 0:
            shuffled_local_dist_ind = self.origin2shuffled[local_dist_ind]
        else:
            shuffled_local_dist_ind = local_dist_ind
        shuffle_index_time = time.time() - shuffle_index_start

        send_size = suggest_size(self.n_data, self.args.k, self.args.world_size)
        send_size = 100
        print("suggest batch size = {}".format(send_size))
        send_ind = 0

        # fagin step
        fagin_start = time.time()
        while cur_n_top < self.args.k and send_ind + send_size <= self.n_data:
            gather_start = time.time()
            start_ind = send_ind
            end_ind = min(self.n_data, send_ind + send_size)

            new_lists = gather(shuffled_local_dist_ind[start_ind:end_ind])
            gather_time += time.time() - gather_start
            send_ind += send_size
            if rank == 0:
                count_start = time.time()
                master_count_by_arr(new_lists, counts, top_k, self.args.k)
                count_time += time.time() - count_start
                bc_start = time.time()
                cur_n_top = len(top_k)
                dist.broadcast(torch.tensor(cur_n_top), 0)
                bc_time += time.time() - bc_start
                print("iter {}, scan {} rows, current top k = {}".format(n_iter, send_size, cur_n_top))
                n_iter += 1
                # dist.barrier()
            else:
                bc_start = time.time()
                tmp_tensor = torch.tensor(0)
                dist.broadcast(tmp_tensor, 0)
                bc_time += time.time() - bc_start
                cur_n_top = tmp_tensor.item()
                print("iter {}, scan {} rows, current top k = {}".format(n_iter, send_size, cur_n_top))
                n_iter += 1
                # dist.barrier()
        fagin_time = time.time() - fagin_start

        # sync candidates for top-k, i.e, the instances seen so far in fagin
        sync_candidate_start = time.time()
        n_candidate = 0
        candidate_ids = []
        if rank == 0:
            candidate_ids = [i for i, e in enumerate(counts) if e > 0]
            n_candidate = len(candidate_ids)
            # print("top-k candidates = {}".format(candidate_ind))
            print("num of candidates = {}".format(n_candidate))
            dist.broadcast(torch.tensor(n_candidate), 0)
            dist.broadcast(torch.tensor(candidate_ids, dtype=torch.int32), 0)
        else:
            tmp_tensor = torch.tensor(0)
            dist.broadcast(tmp_tensor, 0)
            n_candidate = tmp_tensor.item()
            tmp_tensor = torch.zeros([n_candidate], dtype=torch.int32)
            dist.broadcast(tmp_tensor, 0)
            candidate_ids = tmp_tensor.tolist()
            # print("top-k candidates = {}".format(candidate_ind))
            print("num of candidates = {}".format(n_candidate))
        sync_candidate_time = time.time() - sync_candidate_start
        print("candidate indices (shuffled) = {}".format(candidate_ids[:10]))

        cluster_size = len(candidate_ids) // n_cluster
        if len(candidate_ids) % n_cluster != 0:
            cluster_size = cluster_size + 1
        print("number of clusters = {}, size of each clusters = {}".format(n_cluster, cluster_size))

        # reindex candidates
        reindex_start = time.time()
        reindex_candidate_ind = self.shuffled2original[candidate_ids]
        print("{} candidate indices (original) = {}".format(len(candidate_ids), reindex_candidate_ind[:10]))
        reindex_time = time.time() - reindex_start

        # cluster candidates' local distance
        comp_centroid_start = time.time()
        candidate_local_dist = local_dist[reindex_candidate_ind]
        print("candidate local dist: {}".format(candidate_local_dist[:10]))

        # sort candidates' local distance
        # sorted index -> index of candidate_local_dist
        candidate_local_dist_ind = np.argsort(candidate_local_dist)

        centroids_dist = []
        candidate_centroid_dist = [0] * len(candidate_ids)

        # compute centroids for candidates
        for i in range(n_cluster):
            start_ind = i * cluster_size
            end_ind = min((i + 1) * cluster_size, len(candidate_ids))
            if end_ind <= start_ind:
                centroid = centroids_dist[-1]
                print("cur cluster range: {} to {}".format(start_ind, end_ind))
            else:
                cur_cluster_local_dist_ind = candidate_local_dist_ind[start_ind:end_ind]
                # print("distance in cluster {} = {}".format(i, (candidate_local_dist[cur_cluster_local_dist_ind])[:10]))
                centroid = np.mean(candidate_local_dist[cur_cluster_local_dist_ind])
            centroids_dist.append(centroid)
        print("{} centroids dist: {}".format(n_cluster, centroids_dist[:10]))
        for i in range(len(candidate_local_dist_ind)):
            candidate_ind = candidate_local_dist_ind[i]
            candidate_centroid_dist[candidate_ind] = centroids_dist[i // cluster_size]

        candidate_centroid_dist = np.array(candidate_centroid_dist)
        print("{} candidate centroid dist: {}".format(len(candidate_ids), candidate_centroid_dist[:10]))
        comp_centroid_time = time.time() - comp_centroid_start

        is_attend = group_flags[self.args.rank]
        if is_attend == 0:
            candidate_centroid_dist = np.zeros_like(candidate_centroid_dist)

        global_dist_start = time.time()
        global_candidate_dist = sum_sqrt_all_reduce(candidate_centroid_dist)
        print("{} global distance: {}".format(len(global_candidate_dist), global_candidate_dist[:10]))
        print("minimal global distance: {}".format(min(global_candidate_dist)))
        global_dist_time = time.time() - global_dist_start

        select_top_k_start = time.time()
        #top_k_ind = least_indices(global_candidate_dist, self.args.k)
        top_k_sort_ind = np.argsort(global_candidate_dist)[:self.args.k]
        top_k_shuffled_ids = np.array(candidate_ids)[top_k_sort_ind]
        print("top-k indices (shuffled) = {}".format(top_k_shuffled_ids))
        top_k_ids = self.shuffled2original[top_k_shuffled_ids]
        #top_k_ids = reindex_candidate_ind[top_k_sort_ind]
        top_k_dist = global_candidate_dist[top_k_sort_ind]
        select_top_k_time = time.time() - select_top_k_start
        print("indices of k near neighbor = {}".format(top_k_ids))
        print("distance of k near neighbor = {}".format(top_k_dist))

        # calculate label
        count_label_start = time.time()
        label_count = [0 for _ in range(self.args.n_classes)]
        for nid in top_k_ids:
            label_count[self.targets[nid]] += 1
        pred_target = np.argmax(label_count)
        pred_prob = [i / float(k) for i in label_count]
        print("label counts = {}".format(label_count))
        print("prob of labels = {}".format(pred_prob))
        count_label_time = time.time() - count_label_start

        print("find top-k finish: target = {}, prediction = {}, cost {:.2f} s, "
              "comp dist cost {:.2f} s, sort cost {:.2f} s, shuffle index cost {:.2f} s, "
              "fagin cost {:.2f} s = gather cost {:.2f} s + broadcast cost {:.2f} s + count cost {:.2f} s, "
              "sync candidate cost {:.2f} s, reindex cost {:.2f} s, comp centroids cost {:.2f} s, "
              "comp global dist cost {:.2f} s, select top-k cost {:.2f} s, count label cost {:.2f} s"
              .format(test_target, pred_target, time.time() - start_time,
                      local_dist_time, sort_time, shuffle_index_time,
                      fagin_time, gather_time, bc_time, count_time,
                      sync_candidate_time, reindex_time, comp_centroid_time,
                      global_dist_time, select_top_k_time, count_label_time))

        return pred_target, pred_prob
