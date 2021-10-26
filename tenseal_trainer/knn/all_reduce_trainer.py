import time
from multiprocessing import Queue

import numpy as np
from phe import paillier
from utils.distance import square_euclidean_np
from multiprocessing import Process

import sys
sys.path.append("../../")
from transmission.tenseal.tenseal_client import Client


class Trainer(object):

    def __init__(self, args, data, targets):
        self.args = args
        self.data = data
        self.targets = targets
        self.n_data = len(data)

        self.max_msg_size = 900000000
        self.options = [('grpc.max_send_message_length', self.max_msg_size), ('grpc.max_receive_message_length', self.max_msg_size)]
        self.request_data = []
        self.split_size = []
        self.response_data = []

        self.server_addrs = ["bach11.ethz.ch:8991"]
        self.n_threads = 4
        self.n_servers = len(self.server_addrs)

        self.clients = [Client(address, args.rank, args.config) for address in self.server_addrs]

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

        print("server return summed distance, size {}".format(len(self.response_data)))
        return self.response_data

    def clear_cache(self):
        self.request_data = []
        self.split_size = []
        self.response_data = []

    def find_top_k(self, test_data, test_target, k):
        start_time = time.time()
        print(">>> start find top-{}".format(k))

        # average number of features in one part
        n_f = self.args.n_features // self.args.world_size
        if self.args.n_features % self.args.world_size != 0:
            n_f += 1

        # local feature range
        start_f = self.args.rank * n_f
        end_f = min(self.args.n_features, (self.args.rank + 1) * n_f)
        print("local features range = [{},{})".format(start_f, end_f))

        local_dist_start = time.time()
        local_dist = square_euclidean_np(self.data, test_data)
        local_dist_time = time.time() - local_dist_start

        split_start = time.time()
        self.split_data(local_dist, self.n_servers)
        split_time = time.time() - split_start

        trans_start = time.time()
        self.multi_thread_trans()
        trans_time = time.time() - trans_start

        global_dist = np.sqrt(self.response_data)

        select_top_start = time.time()
        top_k_ids = np.argsort(global_dist)[:self.args.k]
        top_k_dist = global_dist[top_k_ids]
        select_top_k_time = time.time() - select_top_start
        print("indices of k near neighbor = {}".format(top_k_ids))
        print("distance of k near neighbor = {}".format(top_k_dist))

        # calculate label
        label_count = [0 for _ in range(self.args.n_classes)]
        for nid in top_k_ids:
            label_count[self.targets[nid]] += 1
        pred_target = np.argmax(label_count)
        pred_prob = [i / float(k) for i in label_count]
        print("label counts = {}".format(label_count))
        print("prob of labels = {}".format(pred_prob))

        print("find top-k finish: target = {}, prediction = {}, total cost {:.2f} s, compute local dist cost {:.2f} s, "
              "split cost {:.2f} s, trans cost {:.2f} s, select top-k cost {:.2f} s"
              .format(test_target, pred_target, time.time() - start_time,
                      local_dist_time, split_time, trans_time, select_top_k_time))

        self.clear_cache()

        return pred_target, pred_prob
