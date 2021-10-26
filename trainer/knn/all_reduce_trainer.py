#TODO clean , same in cluster DONE
import time
from multiprocessing import Queue

import numpy as np
from phe import paillier
from utils.distance import square_euclidean_np
from multiprocessing import Process
from transmission.encryption import listEncryptor

from transmission.pallier.client import Client


class AllReduceTrainer(object):

    def __init__(self, args, data, targets):
        self.args = args
        self.data = data
        self.targets = targets
        self.n_data = len(data)

        #n, p and q define public and private key for clients
        self.n = 17800808502650258601875550958844515471709553355942844427678646632260820003199045314388404911994371628035329514822432294991493010990668660709752669595319311588926604853146533480048467772777445948194921231741018143470691643994153991869049452806017700509483577279858687223679450266912590905234105035466324132288032205654668616576048089785291612076679452672446088686439614576640913512366651332202016430324832498990806535377063538455898743073536813616879452457083048955093965336592637835994478510756622055811364251581599289127031126978248323101428581989623586684860664966965029305250191840964067180439265318568513663821573
        self.p = 116242551141358396967757172433944279550272901992454036802654774494190746770968998083776136351121804910854116390314290022188662376461418450157499142709114012730639393559056023381721752977836405940319438814897846391098115623373672289466369739591592235520707348103430597208678014448138590557989073256027414539843
        self.q = 153135046743798093992187848724462999193161952087785766943569652651932006292000925023754406052364365346195900521786925310092032955755059761206246049824001051042530418086950249290052967287714797696459181867415751414708390688671561087914604963522538318647976596983244525277965449248412149963925303550396761593111
        self.public_key = paillier.PaillierPublicKey(n=self.n)
        self.private_key = paillier.PaillierPrivateKey(self.public_key, self.p, self.q)
        self.max_msg_size = 900000000
        self.options = [('grpc.max_send_message_length', self.max_msg_size), ('grpc.max_receive_message_length', self.max_msg_size)]
        self.trans_data = []
        self.split_sizes = []
        self.global_data = []

        self.addresses = ["bach03.ethz.ch:8991", "bach04.ethz.ch:8991", "bach06.ethz.ch:8991",
                          "bach07.ethz.ch:8991", "bach08.ethz.ch:8991"]
        self.n_threads = 4
        self.n_servers = len(self.addresses)

    def transmit(self, q, i):
        addr = self.addresses[i]
        split_size = self.split_sizes[i]
        print("server {}, address: {}, splitSize:{}".format(i, addr, split_size))
        client = Client(self.addresses[i], self.public_key, self.private_key)
        tmp = client.transmit(self.trans_data[i], self.split_sizes[i], self.options)
        tmp.append(i)
        q.put(tmp)

    def encrypt_thread(self, data, public_key, q, i):
        encrypted = listEncryptor(data, public_key)
        encrypted.append(i)
        q.put(encrypted)

    def split_data(self, data, n_splits):
        size = len(data) // n_splits + (0 if len(data) % n_splits == 0 else 1)
        print("len data: {}, nSplits: {}, split size: {}".format(len(data), n_splits, size))
        for i in range(n_splits):
            if i < n_splits - 1:
                encr_data = data[size*i:size*(i+1)]
                self.trans_data.append(encr_data)
                self.split_sizes.append(len(self.trans_data[i]))
            else:
                encr_data = data[size * i:]
                self.trans_data.append(encr_data)
                self.split_sizes.append(len(self.trans_data[i]))

    def encrypt_data(self, data, n_threads):
        threads = []
        q = Queue()
        rets = []
        encrypted_data = []

        size = len(data) // n_threads
        for i in range(n_threads):
            # Creation of threads to execute
            if i < n_threads - 1:
                split = data[size*i:size*(i+1)]
                t = Process(target=self.encrypt_thread, args=(split, self.public_key, q, i))
                threads.append(t)
            else:
                split = data[size * i:]
                t = Process(target=self.encrypt_thread, args=(split, self.public_key, q, i))
                threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            ret = q.get()
            rets.append(ret)
        for t in threads:
            t.join()

        for i in range(n_threads):
            for elem in rets:
                if elem[-1] == i:
                    #print("last elem {}".format(elem[-1]))
                    encrypted_data.extend(elem[:-1])
        return encrypted_data

    def multi_thread_trans(self, n_servers):
        q = Queue()
        processes = []
        rets = []

        for i in range(n_servers):
            t = Process(target=self.transmit, args=(q, i))
            processes.append(t)
        for p in processes:
            p.start()
        for p in processes:
            ret = q.get()
            rets.append(ret)
        for p in processes:
            p.join()

        for i in range(n_servers):
            for elem in rets:
                if elem[-1] == i:
                    self.global_data.extend((elem[:-1]))
        # for elem in rets:
        #     print(elem[:5])

    def find_top_k(self, test_data, test_target, k):
        start_time = time.time()
        print(">>> start find top-{} <<<".format(k))

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

        homo_start = time.time()
        no_split = True
        if no_split:
            encr_start = time.time()
            encrypted_data = self.encrypt_data(local_dist, self.n_threads)
            encr_time = time.time() - encr_start
            split_start = time.time()
            self.split_data(encrypted_data, self.n_servers)
            split_time = time.time() - split_start
            trans_start = time.time()
            self.multi_thread_trans(self.n_servers)
            trans_time = time.time() - trans_start
        else:
            cuts_number = 10
            data_size = len(local_dist)
            cut = data_size // cuts_number
            for i in range(cuts_number):
                if i < cuts_number - 1:
                    data_split = local_dist[cut * i:cut * (i + 1)]
                    print("current cut size: {}".format(len(data_split)))
                    encr_split = self.encrypt_data(data_split, self.n_threads)
                    self.split_data(encr_split, self.n_servers)
                    print("split encrypted")
                    self.multi_thread_trans(self.n_servers)
                    print("split {} done".format(i))
                    self.trans_data = []
                    self.split_sizes = []
                    print("current global data size : {}".format(len(self.global_data)))
                else:
                    data_split = local_dist[cut * i:]
                    print("current cut size: {}".format(len(data_split)))
                    encr_split = self.encrypt_data(data_split, self.n_threads)
                    print("split encrypted")
                    self.split_data(encr_split, self.n_servers)
                    self.multi_thread_trans(self.n_servers)
                    print("split {} done".format(i))
                    self.trans_data = []
                    self.split_sizes = []
                    print("current global data size : {}".format(len(self.global_data)))
            print("end loop, global data size = {}".format(len(self.global_data)))

        global_dist = np.sqrt(self.global_data)
        homo_time = time.time() - homo_start

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

        print("find top-k finish: target = {}, prediction = {}, total cost {:.2f} s, "
              "compute local dist cost {:.2f} s, "
              "homo communication cost {:.2f} s = encrypt cost {:.2f} s, split cost {:.2f} s, trans cost {:.2f} s, "
              "select top-k cost {:.2f} s"
              .format(test_target, pred_target, time.time() - start_time,
                      local_dist_time, homo_time, encr_time, split_time, trans_time, select_top_k_time))

        self.trans_data = []
        self.split_sizes = []
        self.global_data = []

        return pred_target, pred_prob
