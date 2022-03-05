import time
import math
import random
from multiprocessing import Process, Queue

import numpy as np
import torch
import torch.distributed as dist

from utils.distance import square_euclidean_np
from utils.comm_op import gather
from utils.fagin_utils import suggest_size, fagin_server_count, num_cur_label
from transmission.tenseal_mi.tenseal_mi_aggr_client import MIAggrClient


def get_utility_key(client_attendance):
    key = 0
    for i in reversed(client_attendance):
        key = 2 * key + i
    return key


def utility_key_to_groups(key, world_size):
    client_attendance = [0] * world_size
    for i in range(world_size):
        flag = key % 2
        client_attendance[i] = flag
        key = key // 2
    return client_attendance


class FaginScheTrainer(object):

    def __init__(self, args, data, targets):
        self.args = args
        self.data = data
        self.n_data = len(data)
        self.targets = targets

        # label counts of training data
        unique, counts = np.unique(self.targets, return_counts=True)
        self.label_counts = dict(zip(unique, counts))

        self.n_total_fagin_cand = 0

        self.max_msg_size = 900000000
        self.options = [('grpc.max_send_message_length', self.max_msg_size),
                        ('grpc.max_receive_message_length', self.max_msg_size)]
        self.request_data = []
        self.split_size = []
        self.response_data = []

        self.server_addrs = ["bach11.ethz.ch:8991"]
        self.n_threads = 4
        self.n_servers = len(self.server_addrs)

        self.clients = [MIAggrClient(address, args) for address in self.server_addrs]
        self.group_keys = []

        self.shuffle_seed = 42
        random.seed(self.shuffle_seed)

        if self.args.rank != 0:
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

    def split_data(self, data, n_splits):
        split_size = len(data) // n_splits
        print("len data: {}, num splits: {}, split size: {}".format(len(data), n_splits, split_size))
        for i in range(n_splits):
            if i < n_splits - 1:
                split_data = data[split_size * i:split_size * (i + 1)]
                self.request_data.append(split_data)
                self.split_size.append(len(self.request_data[i]))
            else:
                split_data = data[split_size * i:]
                self.request_data.append(split_data)
                self.split_size.append(len(self.request_data[i]))

    def transmit(self, q, i):
        print("send to aggr server {}: address: {}, split size:{}".format(i, self.server_addrs[i], self.split_size[i]))
        tmp = self.clients[i].transmit(self.request_data[i], self.group_keys)
        # add server index for check
        tmp.append(i)
        q.put(tmp)
        return

    def multi_thread_trans(self):
        q = Queue()
        processes = []
        rets = []

        for i in range(self.n_servers):
            t = Process(target=self.transmit, args=(q, i))
            processes.append(t)
        for p in processes:
            p.start()
        for i in range(len(processes)):
            ret = q.get()
            rets.append(ret)
        for p in processes:
            p.join()

        for server_idx in range(self.n_servers):
            for elem in rets:
                if elem[-1] == server_idx:
                    self.response_data.extend((elem[:-1]))

        print("aggr server return sorted ids, size {}".format(len(self.response_data)))
        return self.response_data

    def clear_cache(self):
        self.request_data = []
        self.split_size = []
        self.response_data = []

    @staticmethod
    def digamma(x):
        return math.log(x, math.e) - 0.5 / x

    def find_top_k(self, test_data, test_target, k, group_keys):
        start_time = time.time()
        self.group_keys = group_keys

        n_f = self.args.n_features // (self.args.world_size - 1)  # rank 0 is Fagin aggregation server
        if self.args.n_features % (self.args.world_size - 1) != 0:
            n_f += 1

        # local feature range
        if self.args.rank > 0:
            print("dist rank = {}, rank = {}".format(dist.get_rank(), self.args.rank))
            start_f = self.args.rank * (n_f - 1)
            end_f = min(self.args.n_features, self.args.rank * n_f)
            print("local features range = [{},{})".format(start_f, end_f))
        else:
            start_f = 0
            end_f = min(self.args.n_features, n_f)
            print("local features range = [{},{})".format(start_f, end_f))

        local_dist = square_euclidean_np(self.data, test_data)
        local_dist_ind = np.argsort(local_dist)
        cal_dist_time = time.time() - start_time

        fagin_start = time.time()
        send_size = suggest_size(self.n_data, self.args.k, self.args.world_size - 1)
        send_size = 100
        print("suggest size by fagin = {}".format(send_size))
        send_ind = 0

        cur_top_k = []
        cur_label_top_k = []
        counts = [0 for _ in range(self.n_data)]
        cur_n_top = 0
        cur_label_n_top = 0
        n_iter = 0
        rank = dist.get_rank()

        # use pre-shuffled id
        if rank != 0:
            shuffled_local_dist_ind = self.origin2shuffled[local_dist_ind]
        else:
            shuffled_local_dist_ind = local_dist_ind

        while cur_label_n_top < self.args.k and send_ind <= self.n_data:
            new_lists = gather(shuffled_local_dist_ind[send_ind:min(self.n_data, send_ind + send_size)])
            send_ind += send_size

            if rank == 0:   # the fagin server
                fagin_server_count(new_lists, counts, cur_top_k, self.args.k)
                cur_n_top = len(cur_top_k)
                dist.broadcast(torch.tensor(cur_n_top), 0) # send num of top-k
                dist.broadcast(torch.tensor(cur_top_k), 0) # send top-k ids to workers
                tmp_tensor = torch.tensor(0)
                dist.broadcast(tmp_tensor, 1)   # receive num of top-k (same label) from leader
                cur_label_n_top = tmp_tensor.item()
                n_iter += 1
                # dist.barrier()
            else:
                tmp_tensor = torch.tensor(0)
                dist.broadcast(tmp_tensor, 0)   # receive num of top-k
                cur_n_top = tmp_tensor.item()
                top_k_tensor = torch.tensor([1, cur_n_top])
                dist.broadcast(top_k_tensor, 0)     # receive top-k ids
                if rank == 1:    # the leader worker
                    reindex_top_k_ids = self.shuffled2original[top_k_tensor.tolist()]
                    cur_label_top_k = num_cur_label(reindex_top_k_ids, self.targets, test_target)
                    cur_label_n_top = len(cur_label_top_k)
                    dist.broadcast(torch.tensor(cur_label_n_top), 1)
                else:
                    dist.broadcast(tmp_tensor, 1)  # receive num of top-k (same label) from leader
                    cur_label_n_top = tmp_tensor.item()
                n_iter += 1
                # dist.barrier()
            print("iter {}, scan {} rows, num top-k = {}, num cur top-k"
                  .format(n_iter, send_size, cur_n_top, cur_label_n_top))

        # sync candidates for top-k, i.e, the instances seen so far in fagin
        n_candidate = 0
        candidate_ids = []
        if rank == 0:
            candidate_ids = [i for i, e in enumerate(counts) if e > 0]
            n_candidate = len(candidate_ids)
            dist.broadcast(torch.tensor(n_candidate), 0)
            dist.broadcast(torch.tensor(candidate_ids, dtype=torch.int32), 0)
        else:
            tmp_tensor = torch.tensor(0)
            dist.broadcast(tmp_tensor, 0)
            n_candidate = tmp_tensor.item()
            tmp_tensor = torch.zeros([n_candidate], dtype=torch.int32)
            dist.broadcast(tmp_tensor, 0)
            candidate_ids = tmp_tensor.tolist()
        print("num of candidates = {}".format(n_candidate))
        self.n_total_fagin_cand += n_candidate
        fagin_time = time.time() - fagin_start

        if rank > 0:
            homo_start = time.time()
            # reindex candidates
            reindex_candidate_ids = self.shuffled2original[candidate_ids]
            # comm candidate distance
            local_candidate_dist = local_dist[reindex_candidate_ids]

            # split data
            self.split_data(local_candidate_dist, self.n_servers)
            # send to server
            self.multi_thread_trans()
            homo_time = time.time() - homo_start

            cal_mi_start = time.time()
            # calculate label
            n_groups = len(self.group_keys)
            group_rankings = np.asarray(self.response_data).reshape(n_groups, n_candidate)

            # calculate k-nearest neighbors whose label = test data
            client_mi_values = np.zeros(self.args.world_size - 1)
            group_accuracy = []
            for group_ind in range(len(self.group_keys)):
                group_flags = utility_key_to_groups(self.group_keys[group_ind])

                reindex_top_k_ids = reindex_candidate_ids[group_rankings[group_ind]]

                # cal prediction
                target_count = [0 for _ in range(self.args.n_classes)]
                for j in reindex_top_k_ids[:k]:
                    target_count[self.targets[j]] += 1
                pred = np.argmax(target_count)
                group_accuracy.append(float(pred == test_target))

                cur_label_top_k_ids = []
                cur_label_count = 0
                all_label_count = 0
                for i in range(n_candidate):
                    candidate_id = reindex_top_k_ids[i]
                    all_label_count += 1
                    if self.targets[candidate_id] == test_target:
                        cur_label_top_k_ids.append(candidate_id)
                        cur_label_count += 1
                        if cur_label_count == k:
                            break

                N = len(self.data)
                N_i = self.label_counts[test_target]
                m_i = all_label_count

                mi_value = self.digamma(N) - self.digamma(N_i) + self.digamma(k) - self.digamma(m_i)
                client_mi_values += np.array(group_flags) * mi_value
            cal_mi_time = time.time() - cal_mi_start

            print("cost {} s, cal dist cost {} s, fagin cost {} s, homo cost {} s, cal mi cost {} s"
                  .format(time.time() - start_time, cal_dist_time, fagin_time, homo_time, cal_mi_time))

            return np.asarray(group_accuracy), client_mi_values
        else:
            return np.zeros(len(self.group_keys)), np.zeros(self.args.world_size - 1)
