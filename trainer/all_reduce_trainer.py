import time

import numpy as np

from utils.distance import square_euclidean_np
from utils.comm_op import sum_sqrt_all_reduce


PER_ENCRYPT_TIME = 0.01
FLOAT_BYTES = 4
HOMOMORPHIC_BYTES = 4096 * 4


class AllReduceTrainer(object):

    def __init__(self, args, data, targets):
        self.args = args
        self.data = data
        self.targets = targets
        self.n_data = len(data)

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

        encrypt_start = time.time()
        #time.sleep(PER_ENCRYPT_TIME * self.n_data)
        encrypt_time = PER_ENCRYPT_TIME * self.n_data

        dist_comm_start = time.time()
        global_dist = sum_sqrt_all_reduce(local_dist)
        # print("global distance size = {}, values = {}".format(len(global_dist), global_dist[:10]))
        float_dist_comm_time = time.time() - dist_comm_start

        homo_dist_comm_start = time.time()
        homo_size_factor = int(HOMOMORPHIC_BYTES / FLOAT_BYTES)
        print("do {} times homo communication".format(int(homo_size_factor / 4)))
        #for _ in range(int(homo_size_factor / 4)):
        homo_dist = [local_dist for _ in range(4)]
        homo_dist = sum_sqrt_all_reduce(np.asarray(homo_dist))
        one_homo_time = time.time() - homo_dist_comm_start
        print("send one homo tensor, size = {}, cost {} s".format(len(homo_dist), one_homo_time))
        homo_dist_comm_time = int(homo_size_factor / 4) * one_homo_time

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

        print("find top-k finish: target = {}, prediction = {}, cost {:.2f} s, "
              "compute local dist cost {:.2f} s, encryption cost {:.2f} s, "
              "float communication cost {:.2f} s, homo communication cost {:.2f} s, "
              "select top-k cost {:.2f} s"
              .format(test_target, pred_target,
                      time.time() - start_time + encrypt_time - float_dist_comm_time + homo_dist_comm_time,
                      local_dist_time, encrypt_time, float_dist_comm_time, homo_dist_comm_time, select_top_time))

        return pred_target
