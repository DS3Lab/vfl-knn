import time
import math
import random

import numpy as np
import torch
import torch.distributed as dist

from utils.distance import square_euclidean_np
from utils.comm_op import gather, sum_sqrt_all_reduce
from utils.fagin_utils import suggest_size, master_count_by_arr


class FaginArrTrainer(object):

    def __init__(self, args, data, targets):
        self.args = args
        self.data = data
        self.n_data = len(data)
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
        #print("local features range = [{},{})".format(start_f, end_f))

        local_dist_start = time.time()
        local_dist = square_euclidean_np(self.data, test_data)
        # print("local distance size = {}, values = {}".format(len(local_dist), local_dist[:10]))
        local_dist_time = time.time() - local_dist_start

        sort_start = time.time()
        local_dist_ind = np.argsort(local_dist)
        # print("local dist index = {}".format(local_dist_ind[:10]))
        # print("local dist = {}".format(local_dist[local_dist_ind[:10]]))
        sort_time = time.time() - sort_start

        send_size = suggest_size(self.n_data, self.args.k, self.args.world_size)
        print("suggest batch size = {}".format(send_size))
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
            new_lists = gather(local_dist_ind[send_ind:min(self.n_data, send_ind + send_size)])
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

        fagin_time = time.time() - fagin_start

        # get candidates for top-k, i.e, the instances seen so far in fagin
        candidate_start = time.time()
        n_candidate = 0
        candidate_ids = []
        if rank == 0:
            for i, e in enumerate(counts):
                is_sample = random.random() < e / float(self.args.world_size)
                if is_sample is True:
                    candidate_ids.append(i)
            n_candidate = len(candidate_ids)
            #print("top-k candidates = {}".format(candidate_ind))
            print("number of candidates = {}".format(n_candidate))
            dist.broadcast(torch.tensor(n_candidate), 0)
            dist.broadcast(torch.tensor(candidate_ids, dtype=torch.int32), 0)
        else:
            tmp_tensor = torch.tensor(0)
            dist.broadcast(tmp_tensor, 0)
            n_candidate = tmp_tensor.item()
            print("number of candidates = {}".format(n_candidate))
            tmp_tensor = torch.zeros([n_candidate], dtype=torch.int32)
            dist.broadcast(tmp_tensor, 0)
            candidate_ids = tmp_tensor.tolist()
            #print("top-k candidates = {}".format(candidate_ind))
            #print("number of candidates = {}".format(n_candidate))
        candidate_time = time.time() - candidate_start

        # comm candidate distance
        candidate_dist_start = time.time()
        candidate_local_dist = local_dist[candidate_ids]
        candidate_dist = sum_sqrt_all_reduce(candidate_local_dist)
        candidate_dist_time = time.time() - candidate_dist_start

        # select top-k
        select_top_start = time.time()
        ind_k = np.argsort(candidate_dist)[:self.args.k]
        #print("ind_k = {}".format(ind_k))
        top_k = np.asarray(candidate_ids)[ind_k]
        dist_k = candidate_dist[ind_k]
        select_top_time = time.time() - select_top_start
        print("indices of k near neighbor = {}".format(top_k))
        print("distance of k near neighbor = {}".format(dist_k))

        # calculate label
        count_label_start = time.time()
        label_count = [0 for _ in range(self.args.n_classes)]
        for j in top_k:
            label_count[self.targets[j]] += 1
        pred_target = np.argmax(label_count)
        pred_prob = [i / float(k) for i in label_count]
        #print("label counts = {}".format(label_count))
        #print("prob of labels = {}".format(pred_prob))
        count_label_time = time.time() - count_label_start

        print("find top-k finish: target = {}, prediction = {}, total cost {:.2f} s, "
              "comp dist cost {:.2f} s, sort cost {:.2f} s, "
              "fagin cost {:.2f} s = gather cost {:.2f} s + broadcast cost {:.2f} s + count cost {:.2f} s, "
              "get candidates cost {:.2f} s, comm candidate distance cost {:.2f} s, "
              "select top-k cost {:.2f} s, count label cost {:.2f} s"
              .format(test_target, pred_target, time.time() - start_time,
                      local_dist_time, sort_time, fagin_time, gather_time, bc_time, count_time,
                      candidate_time, candidate_dist_time, select_top_time, count_label_time))

        return pred_target, pred_prob
