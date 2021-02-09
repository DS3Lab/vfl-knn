import time

import numpy as np

from utils.distance import square_euclidean_np
from utils.comm_op import sum_sqrt_all_reduce


class AllReduceTrainer(object):

    def __init__(self, args, data, targets):
        self.args = args
        self.data = data
        self.targets = targets

    def find_top_k(self, test_data, test_target, k):
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
        # print("local distance size = {}, values = {}".format(len(local_dist), local_dist[:10]))
        local_dist_time = time.time() - local_dist_start

        comm_start = time.time()
        global_dist = sum_sqrt_all_reduce(local_dist)
        # print("global distance size = {}, values = {}".format(len(global_dist), global_dist[:10]))
        comm_time = time.time() - comm_start

        select_top_start = time.time()
        # ind_k = least_indices(global_dist, args.k)
        ind_k = np.argsort(global_dist)[:self.args.k]
        dist_k = global_dist[ind_k]
        select_top_time = time.time() - select_top_start
        print("indices of k near neighbor = {}".format(ind_k))
        print("distance of k near neighbor = {}".format(dist_k))

        # calculate label
        label_count = [0 for _ in range(self.args.n_classes)]
        for nid in ind_k:
            label_count[self.targets[nid]] += 1
        pred_target = np.argmax(label_count)

        print("find top-k finish: target = {}, prediction = {}, "
              "cost {} s, compute local dist cost {} s, communication {} s, select top-k cost {} s"
              .format(test_target, pred_target, time.time() - start_time,
                      local_dist_time, comm_time, select_top_time))

        return pred_target
