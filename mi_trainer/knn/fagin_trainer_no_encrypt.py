import time
import math

import numpy as np
import torch
import torch.distributed as dist

from utils.distance import square_euclidean_np
from utils.comm_op import gather, sum_sqrt_all_reduce
from utils.fagin_utils import suggest_size, master_count_label


class FaginArrTrainer(object):

    def __init__(self, args, data, targets):
        self.args = args
        self.data = data
        self.n_data = len(data)
        self.targets = targets

        unique, counts = np.unique(self.targets, return_counts=True)
        self.label_counts = dict(zip(unique, counts))

        self.n_cur_fagin_cand = 0
        self.n_total_fagin_cand = 0

    @staticmethod
    def digamma(x):
        return math.log(x, math.e) - 0.5 / x

    # runs fagin, and the leader checks the label
    # the leader decides whether to stop fagin when find k-nearest
    # aggregates distance
    # the leader calculates the number of this label and all samples
    # calculate I for this label
    # calculate average I for all labels

    def find_top_k(self, test_data, test_target, k, group_flags):
        start_time = time.time()
        #print(">>> start find top-{} <<<".format(k))

        # average number of features in one part
        n_f = int(self.args.n_features / self.args.world_size)
        if self.args.n_features % self.args.world_size != 0:
            n_f += 1

        # local feature range
        start_f = self.args.rank * n_f
        end_f = min(self.args.n_features, (self.args.rank + 1) * n_f)
        #print("local features range = [{},{})".format(start_f, end_f))

        local_dist_start = time.time()
        is_attend = group_flags[self.args.rank]
        local_dist = square_euclidean_np(self.data, test_data)
        if is_attend == 0:
            local_dist = np.zeros_like(local_dist)
        local_dist_time = time.time() - local_dist_start

        sort_start = time.time()
        local_dist_ind = np.argsort(local_dist)
        sort_time = time.time() - sort_start

        send_size = suggest_size(self.n_data, self.args.k, self.args.world_size)
        send_size = 100
        #print("suggest batch size = {}".format(send_size))
        send_ind = 0

        fagin_start = time.time()
        gather_time = 0
        bc_time = 0
        count_time = 0
        top_k = []
        counts = [0 for _ in range(self.n_data)]
        cur_n_top = 0
        n_iter = 0
        rank = dist.get_rank()

        while cur_n_top < self.args.k and send_ind <= self.n_data:
            gather_start = time.time()
            send_ind_end = min(self.n_data, send_ind + send_size)
            new_lists = gather(local_dist_ind[send_ind:send_ind_end])
            gather_time += time.time() - gather_start
            if rank == 0:
                count_start = time.time()
                master_count_label(new_lists, counts, top_k, self.args.k,
                                   self.targets, test_target)
                count_time += time.time() - count_start
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
                cur_n_top = tmp_tensor.item()
                # print("iter {}, scan {} rows, current top k = {}".format(n_iter, send_size, cur_n_top))
                n_iter += 1
                # dist.barrier()
            send_ind += send_size

        fagin_time = time.time() - fagin_start

        # get candidates for top-k, i.e, the instances seen so far in fagin
        candidate_start = time.time()
        n_candidate = 0
        candidate_ind = []
        if rank == 0:
            candidate_ind = [i for i, e in enumerate(counts) if e > 0]
            n_candidate = len(candidate_ind)
            #print("number of candidates = {}".format(n_candidate))
            dist.broadcast(torch.tensor(n_candidate), 0)
            dist.broadcast(torch.tensor(candidate_ind, dtype=torch.int32), 0)
        else:
            tmp_tensor = torch.tensor(0)
            dist.broadcast(tmp_tensor, 0)
            n_candidate = tmp_tensor.item()
            #print("number of candidates = {}".format(n_candidate))
            tmp_tensor = torch.zeros([n_candidate], dtype=torch.int32)
            dist.broadcast(tmp_tensor, 0)
            candidate_ind = tmp_tensor.tolist()
            #print("top-k candidates = {}".format(candidate_ind))
            #print("number of candidates = {}".format(n_candidate))
        candidate_time = time.time() - candidate_start

        self.n_cur_fagin_cand += n_candidate
        self.n_total_fagin_cand += n_candidate

        # comm candidate distance
        candidate_dist_start = time.time()
        candidate_local_dist = local_dist[candidate_ind]
        is_attend = group_flags[self.args.rank]
        if is_attend == 0:
            candidate_local_dist = np.zeros_like(candidate_local_dist)
        candidate_dist = sum_sqrt_all_reduce(candidate_local_dist)
        candidate_dist_time = time.time() - candidate_dist_start

        # sort global distance
        select_top_start = time.time()
        sort_ids = np.argsort(candidate_dist)
        sort_dist = candidate_dist[sort_ids]
        sort_time = time.time() - sort_start

        # calculate k-nearest neighbors whose label = test data
        cal_mi_start = time.time()
        cur_label_top_k_ids = []
        cur_label_top_k_dist = []
        cur_label_count = 0
        all_label_count = 0
        for i in range(len(candidate_dist)):
            cur_id = sort_ids[i]
            candidate_id = candidate_ind[cur_id]
            all_label_count += 1
            if self.targets[candidate_id] == test_target:
                cur_label_top_k_ids.append(candidate_id)
                cur_label_top_k_dist.append(candidate_dist[cur_id])
                cur_label_count += 1
                if cur_label_count == k:
                    break

        distance_threshold = cur_label_top_k_dist[-1]

        N = len(self.data)
        N_i = self.label_counts[test_target]
        m_i = all_label_count
        #print("# samples = {}, # labels {} = {}, k = {}, # neighbors < distance {} = {}"
        #      .format(N, test_target, N_i, k, distance_threshold, m_i))
        cal_mi_time = time.time() - cal_mi_start

        mi_value = self.digamma(N) - self.digamma(N_i) + self.digamma(k) - self.digamma(m_i)

        return mi_value
