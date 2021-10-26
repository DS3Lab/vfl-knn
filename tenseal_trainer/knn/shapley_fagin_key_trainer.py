import time
import sys
from multiprocessing import Process, Queue

import random
import numpy as np
import torch
import torch.distributed as dist

sys.path.append("../../")
from utils.distance import square_euclidean_np
from utils.comm_op import gather
from utils.fagin_utils import suggest_size, master_count_by_arr
from utils.shapley_utils import generate_all_combinations
from transmission.tenseal.tenseal_shapley_aggr_key_client import ShapleyAggrKeyClient


class ShapleyFaginKeyTrainer(object):

    def __init__(self, args, data, targets):
        self.args = args
        self.data = data
        self.n_data = len(data)
        self.targets = targets

        self.max_msg_size = 900000000
        self.options = [('grpc.max_send_message_length', self.max_msg_size),
                        ('grpc.max_receive_message_length', self.max_msg_size)]
        self.request_data = []
        self.split_size = []
        self.response_data = []

        self.server_addrs = ["bach11.ethz.ch:8991"]
        self.n_threads = 4
        self.n_servers = len(self.server_addrs)

        self.clients = [ShapleyAggrKeyClient(address, args) for address in self.server_addrs]

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
        print("send to server {}: address: {}, split size:{}".format(i, self.server_addrs[i], self.split_size[i]))
        tmp = self.clients[i].transmit(self.request_data[i])
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

        print("server return shapley top-k, size {}".format(len(self.response_data)))
        return self.response_data

    def clear_cache(self):
        self.request_data = []
        self.split_size = []
        self.response_data = []

    def find_top_k(self, test_data, test_target, k):
        start_time = time.time()
        print(">>> start find top-{}".format(k))

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

        # rank 0 is coordinator and does no training
        send_size = suggest_size(self.n_data, self.args.k, self.args.world_size)
        send_size = 100
        print("suggest size by fagin = {}".format(send_size))
        send_ind = 0

        shuffle_index_time = 0
        gather_time = 0
        bc_time = 0
        count_time = 0
        reindex_time = 0
        split_time = 0
        trans_time = 0

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

        fagin_start = time.time()
        while cur_n_top < self.args.k and send_ind <= self.n_data:
            gather_start = time.time()
            new_lists = gather(shuffled_local_dist_ind[send_ind:min(self.n_data, send_ind + send_size)])
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
            #print("top-k candidates = {}".format(candidate_ind))
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
            #print("top-k candidates = {}".format(candidate_ind))
            print("num of candidates = {}".format(n_candidate))
        sync_candidate_time = time.time() - sync_candidate_start

        # reindex candidates
        reindex_start = time.time()
        reindex_candidate_ids = self.shuffled2original[candidate_ids]
        reindex_time = time.time() - reindex_start

        # select local distances of k
        local_candidate_dist = local_dist[reindex_candidate_ids]

        # split data
        split_start = time.time()
        self.split_data(local_candidate_dist, self.n_servers)
        split_time = time.time() - split_start

        # send to server
        trans_start = time.time()
        self.multi_thread_trans()
        trans_time = time.time() - trans_start

        # calculate label
        count_label_start = time.time()
        client_combinations = generate_all_combinations(self.args.world_size)
        n_combinations = len(client_combinations)
        assert n_combinations == 2**self.args.world_size - 1
        top_k_rankings = np.asarray(self.response_data).reshape(n_combinations, self.args.k)

        client_combination_accuracy = []
        for i in range(n_combinations):
            target_count = [0 for _ in range(self.args.n_classes)]
            #print("client combination {}, top k = {}".format(client_combinations[i], top_k_rankings[i]))
            reindex_top_k_ids = reindex_candidate_ids[top_k_rankings[i]]
            for j in reindex_top_k_ids:
                target_count[self.targets[j]] += 1
            pred = np.argmax(target_count)
            client_combination_accuracy.append(float(pred == test_target))
        count_label_time = time.time() - count_label_start
        #print("client combination accuracies = {}".format(client_combination_accuracy))

        homo_time = reindex_time + split_time + trans_time

        print("fagin for one test data finish, cost {:.2f} s, "
              "comp dist cost {:.2f} s, sort cost {:.2f} s, shuffling index time {:.2f} s, "
              "fagin cost {:.2f} s = gather cost {:.2f} s + broadcast cost {:.2f} s + count cost {:.2f} s, "
              "sync candidates cost {:.2f} s, homomorphic cost {:.2f} s = "
              "reindex to real {:.2f} s + split {:.2f} s + transmission {:.2f} s, count labels {:.2f} s"
              .format(time.time() - start_time,
                      local_dist_time, sort_time, shuffle_index_time,
                      fagin_time, gather_time, bc_time, count_time,
                      sync_candidate_time, homo_time,
                      reindex_time, split_time, trans_time, count_label_time))

        self.clear_cache()
        return client_combination_accuracy


if __name__ == '__main__':
    print(int(True))
