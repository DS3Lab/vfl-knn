import time
import sys
import math

import numpy as np
import torch
import torch.distributed as dist

sys.path.append("../")
from utils.distance import square_euclidean_np
from utils.comm_op import gather, sum_sqrt_all_reduce
from utils.fagin_utils import suggest_size, count_by_arr


PER_ENCRYPT_TIME = 0.01
FLOAT_BYTES = 4
HOMOMORPHIC_BYTES = 4096 * 4


class ClusterTrainer(object):

    def __init__(self, args, data, targets):
        self.args = args
        self.data = data
        self.n_data = len(data)
        self.targets = targets

    def find_top_k(self, test_data, test_target, k, n_cluster):
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

        cluster_size = int(self.n_data / n_cluster)
        if self.n_data % n_cluster != 0:
            cluster_size += 1
        print("number of clusters = {}, size of each clusters = {}".format(n_cluster, cluster_size))

        local_dist_start = time.time()
        local_dist = square_euclidean_np(self.data, test_data)
        # print("local distance size = {}, values = {}".format(len(local_dist), local_dist[:10]))
        local_dist_time = time.time() - local_dist_start

        sort_start = time.time()
        local_dist_ind = np.argsort(local_dist)
        # print("local dist index = {}".format(local_dist_ind[:10]))
        # print("local dist = {}".format(local_dist[local_dist_ind[:10]]))
        sort_time = time.time() - sort_start

        send_size = cluster_size
        print("suggest batch size = {}".format(send_size))
        send_ind = 0

        comm_start = time.time()
        # each encryption cost 10ms
        gather_time = 0
        bc_time = 0
        count_time = 0
        top_k = []
        top_k_cluster_ind = []
        top_k_local_dist = []
        centroids_dist = []
        counts = [0 for _ in range(self.n_data)]
        cur_n_top = 0
        cluster_ind = 0
        rank = dist.get_rank()

        while cur_n_top < self.args.k and send_ind + send_size <= self.n_data:
            gather_start = time.time()
            start_ind = send_ind
            end_ind = min(self.n_data, send_ind + send_size)
            mid_ind = int((start_ind + end_ind) / 2)
            new_lists = gather(local_dist_ind[start_ind:end_ind])
            gather_time += time.time() - gather_start
            send_ind += send_size
            if rank == 0:
                count_start = time.time()
                count_by_arr(new_lists, counts, top_k, self.args.k)
                count_time += time.time() - count_start
                if len(top_k) - cur_n_top > 0:
                    centroids_dist.append(local_dist[local_dist_ind[mid_ind]])
                for _ in range(len(top_k) - cur_n_top):
                    top_k_cluster_ind.append(cluster_ind)
                    top_k_local_dist.append(local_dist[local_dist_ind[mid_ind]])
                bc_start = time.time()
                cur_n_top = len(top_k)
                dist.broadcast(torch.tensor(cur_n_top), 0)
                bc_time += time.time() - bc_start
                # print("iter {}, scan {} rows, current top k = {}".format(n_iter, send_size, cur_n_top))
                cluster_ind += 1
                # dist.barrier()
            else:
                bc_start = time.time()
                tmp_tensor = torch.tensor(0)
                dist.broadcast(tmp_tensor, 0)
                bc_time += time.time() - bc_start
                new_n_top = tmp_tensor.item()
                if new_n_top - cur_n_top > 0:
                    centroids_dist.append(local_dist[local_dist_ind[mid_ind]])
                for _ in range(new_n_top - cur_n_top):
                    top_k_cluster_ind.append(cluster_ind)
                    top_k_local_dist.append(local_dist[local_dist_ind[mid_ind]])
                cur_n_top = tmp_tensor.item()
                # print("iter {}, scan {} rows, current top k = {}".format(n_iter, send_size, cur_n_top))
                cluster_ind += 1
                # dist.barrier()
            #print("scan from {} to {}, number of found top-{} = {}".format(start_ind, end_ind, self.args.k, cur_n_top))

        # sync top-k indices
        if rank == 0:
            bc_start = time.time()
            cur_n_top = len(top_k)
            dist.broadcast(torch.tensor(cur_n_top), 0)
            top_k_tensor = torch.tensor(top_k, dtype=torch.int32)
            #print("top-k tensor: shape = {}, value = {}".format(top_k_tensor.type(), top_k_tensor.shape, top_k_tensor))
            dist.broadcast(top_k_tensor, 0)
            bc_time += time.time() - bc_start
        else:
            bc_start = time.time()
            tmp_tensor = torch.tensor(0)
            dist.broadcast(tmp_tensor, 0)
            cur_n_top = tmp_tensor.item()
            tmp_tensor = torch.zeros([cur_n_top], dtype=torch.int32)
            dist.broadcast(tmp_tensor, 0)
            #print("top-k tensor: shape = {}, value = {}".format(tmp_tensor.type(), tmp_tensor.shape, tmp_tensor))
            top_k = tmp_tensor.tolist()
            bc_time += time.time() - bc_start

        unique_cluster_ind = np.unique(top_k_cluster_ind)
        print("unique clusters: size = {}, indices= {}".format(len(unique_cluster_ind), unique_cluster_ind))
        # encrypt_start = time.time()
        # time.sleep(PER_ENCRYPT_TIME * send_size)
        encrypt_time = PER_ENCRYPT_TIME * 0

        dist_comm_start = time.time()
        print("total number of current top k = {}".format(cur_n_top))
        print("indices of {} near neighbors = {}".format(self.args.k, top_k[:self.args.k]))
        print("cluster index of top-{}: {}".format(self.args.k, top_k_cluster_ind))
        print("local dist of {} near neighbors = {}".format(self.args.k, top_k_local_dist[:self.args.k]))
        print("number of cluster centroids in top {} = {}".format(self.args.k, len(centroids_dist)))
        print("local dist of cluster centroids = {}".format(centroids_dist))
        global_centroids_dist = sum_sqrt_all_reduce(np.asarray(centroids_dist))
        print("global dist of cluster centroids = {}".format(global_centroids_dist))
        float_dist_comm_time = time.time() - dist_comm_start

        homo_dist_comm_start = time.time()
        homo_dist = np.array([centroids_dist for _ in range(int(HOMOMORPHIC_BYTES / FLOAT_BYTES))])
        print("homo tensor size = {}".format(homo_dist.shape))
        homo_dist = sum_sqrt_all_reduce(homo_dist)
        #homo_dist_comm_time = HOMOMORPHIC_BYTES / FLOAT_BYTES * float_dist_comm_time
        #time.sleep(homo_dist_comm_time - float_dist_comm_time)
        homo_dist_comm_time = time.time() - homo_dist_comm_start

        # calculate label
        target_count = [0 for _ in range(self.args.n_classes)]
        for j in top_k:
            target_count[self.targets[j]] += 1
        pred_target = np.argmax(target_count)

        print("find top-k finish: target = {}, prediction = {}, cost {:.2f} s, "
              "comp dist cost {:.2f} s, sort cost {:.2f} s, count cost {:.2f} s, encryption cost {:.2f} s, "
              "communication cost {:.2f} s = gather cost {:.2f} s + broadcast cost {:.2f} s "
              "+ float dist cost {:.2f} s + homo dist cost {:.2f} s"
              .format(test_target, pred_target,
                      time.time() - start_time + encrypt_time - float_dist_comm_time + homo_dist_comm_time,
                      local_dist_time, sort_time, count_time, encrypt_time,
                      gather_time + bc_time + float_dist_comm_time + homo_dist_comm_time,
                      gather_time, bc_time, float_dist_comm_time, homo_dist_comm_time))

        return pred_target
