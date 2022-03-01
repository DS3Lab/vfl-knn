import time
import math

import numpy as np

from utils.distance import square_euclidean_np
from utils.comm_op import sum_sqrt_all_reduce


class AllReduceTrainer(object):

    def __init__(self, args, data, targets):
        self.args = args
        self.data = data
        self.targets = targets

        unique, counts = np.unique(self.targets, return_counts=True)
        self.label_counts = dict(zip(unique, counts))

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
        n_f = self.args.n_features // self.args.world_size
        if self.args.n_features % self.args.world_size != 0:
            n_f += 1

        # local feature range
        start_f = self.args.rank * n_f
        end_f = min(self.args.n_features, (self.args.rank + 1) * n_f)
        #print("local features range = [{},{})".format(start_f, end_f))

        local_dist_start = time.time()
        is_attend = group_flags[self.args.rank]
        local_dist = square_euclidean_np(self.data, test_data)
        #print("local distance shape: {}".format(local_dist.shape))
        if is_attend == 0:
            local_dist = np.zeros_like(local_dist)
        #print("{} local distance: {}".format(len(local_dist), local_dist[:10]))
        local_dist_time = time.time() - local_dist_start

        comm_start = time.time()
        global_dist = sum_sqrt_all_reduce(local_dist)
        #print("{} global distance: {}".format(len(global_dist), global_dist[:10]))
        comm_time = time.time() - comm_start

        sort_start = time.time()
        # ind_k = least_indices(global_dist, args.k)
        sort_ids = np.argsort(global_dist)
        sort_dist = global_dist[sort_ids]
        sort_time = time.time() - sort_start

        # calculate k-nearest neighbors whose label = test data
        cal_mi_start = time.time()
        cur_label_top_k_ids = []
        cur_label_top_k_dist = []
        cur_label_count = 0
        all_label_count = 0
        for i in range(len(global_dist)):
            cur_id = sort_ids[i]
            all_label_count += 1
            if self.targets[cur_id] == test_target:
                cur_label_top_k_ids.append(cur_id)
                cur_label_top_k_dist.append(global_dist[cur_id])
                cur_label_count += 1
                if cur_label_count == k:
                    break

        # print("indices of k near neighbor of label {} = {}"
        #       .format(cur_label_top_k_ids, test_target))
        # print("distance of k near neighbor of label {} = {}"
        #       .format(cur_label_top_k_dist, test_target))
        distance_threshold = cur_label_top_k_dist[-1]

        N = len(self.data)
        N_i = self.label_counts[test_target]
        m_i = all_label_count
        print("# samples = {}, # labels {} = {}, k = {}, # neighbors < distance {} = {}"
              .format(N, test_target, N_i, k, distance_threshold, m_i))
        cal_mi_time = time.time() - cal_mi_start

        mi_value = self.digamma(N) - self.digamma(N_i) + self.digamma(k) - self.digamma(m_i)

        # print("find top-k finish: cost {:.2f} s, "
        #       "compute local dist cost {:.2f} s, communication {:.2f} s, "
        #       "sort cost {:.2f} s, cal mutual info cost {:.2f} s"
        #       .format(time.time() - start_time,
        #               local_dist_time, comm_time, sort_time, cal_mi_time))

        return mi_value
