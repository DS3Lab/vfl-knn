#TODO  clean + same in CLUSTER DONE
import time
import sys

import random
import numpy as np
import torch
import torch.distributed as dist
from phe import paillier
from multiprocessing import Queue
from multiprocessing import Process

sys.path.append("../")
from utils.distance import square_euclidean_np
from utils.comm_op import gather
from utils.fagin_utils import suggest_size, coordinator_count_by_arr
from transmission.pallier.client import Client
from transmission.encryption import encryptor


class FaginArrTrainer(object):

    def __init__(self, args, data, targets):
        self.args = args
        self.data = data
        self.n_data = len(data)
        self.targets = targets

        # n, p and q define public and private key for clients
        self.n = 17800808502650258601875550958844515471709553355942844427678646632260820003199045314388404911994371628035329514822432294991493010990668660709752669595319311588926604853146533480048467772777445948194921231741018143470691643994153991869049452806017700509483577279858687223679450266912590905234105035466324132288032205654668616576048089785291612076679452672446088686439614576640913512366651332202016430324832498990806535377063538455898743073536813616879452457083048955093965336592637835994478510756622055811364251581599289127031126978248323101428581989623586684860664966965029305250191840964067180439265318568513663821573
        self.p = 116242551141358396967757172433944279550272901992454036802654774494190746770968998083776136351121804910854116390314290022188662376461418450157499142709114012730639393559056023381721752977836405940319438814897846391098115623373672289466369739591592235520707348103430597208678014448138590557989073256027414539843
        self.q = 153135046743798093992187848724462999193161952087785766943569652651932006292000925023754406052364365346195900521786925310092032955755059761206246049824001051042530418086950249290052967287714797696459181867415751414708390688671561087914604963522538318647976596983244525277965449248412149963925303550396761593111
        self.public_key = paillier.PaillierPublicKey(n=self.n)
        self.private_key = paillier.PaillierPrivateKey(self.public_key, self.p, self.q)
        self.max_msg_size = 800000000
        self.options = [('grpc.max_send_message_length', self.max_msg_size), ('grpc.max_receive_message_length', self.max_msg_size)]
        self.trans_data = []
        self.split_sizes = []
        self.global_data = []

        self.addresses = ["bach03.ethz.ch:8991", "bach04.ethz.ch:8991", "bach06.ethz.ch:8991", "bach07.ethz.ch:8991", "bach08.ethz.ch:8991"]
        self.n_threads = 4
        self.n_servers = len(self.addresses)

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

    def encryptor_thread(self, data, public_key, q, i):
        encrypted = encryptor(data, public_key)
        encrypted.append(i)
        q.put(encrypted)

    def encrypt_data(self, data, n_threads):
        threads = []
        q = Queue()
        rets = []
        encrypted_data = []

        size = len(data) // n_threads
        for i in range(n_threads):
            #Creation of threads to execute
            if i < n_threads - 1:
                split = data[size*i:size*(i+1)]
                t = Process(target=self.encryptor_thread, args=(split, self.public_key, q, i))
                threads.append(t)
            else:
                split = data[size * i:]
                t = Process(target=self.encryptor_thread, args=(split, self.public_key, q, i))
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
                    encrypted_data.extend(elem[:-1])
        return encrypted_data

    def split_data(self, data, nSplits):
        size = len(data) // nSplits + (0 if len(data) % nSplits == 0 else 1)
        print("len data: {}, nSplits: {}, split size: {}".format(len(data), nSplits, size))
        for i in range(nSplits):
            if i < nSplits - 1:
                encrData = data[size*i : size*(i+1)]
                self.trans_data.append(encrData)
                self.split_sizes.append(len(self.trans_data[i]))
            else:
                encrData = data[size * i:]
                self.trans_data.append(encrData)
                self.split_sizes.append(len(self.trans_data[i]))

    def transmit(self, q, i):
        addr = self.addresses[i]
        spliSize = self.split_sizes[i]
        print("server {}, addr: {}, splitSize:{}". format(i, addr, spliSize))
        client = Client(self.addresses[i], self.public_key, self.private_key)
        tmp = client.transmit(self.trans_data[i], self.split_sizes[i], self.options)
        tmp.append(i)
        q.put(tmp)

    def multi_thread_trans(self, n_servers):
        q = Queue()
        processes = []
        rets = []
        global_data = []

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
                    global_data.extend((elem[:-1]))

        # for elem in rets:
        #     print(elem[:5])

        print("data returned by servers has size {}".format(len(global_data)))
        return global_data

    def find_top_k(self, test_data, test_target, k):
        start_time = time.time()
        print(">>> start find top-{} <<<".format(k))

        n_f = self.args.n_features // (self.args.world_size - 1)  # We add -1, in order to pass 11 as world_size
        if self.args.n_features % (self.args.world_size - 1) != 0:
            n_f += 1

        # local feature range
        # Each machine takes the range of data according to their rank
        if self.args.rank < self.args.world_size - 1:
            print("rank dist= {}, rank = {}".format(dist.get_rank(), self.args.rank))
            start_f = self.args.rank * n_f
            end_f = min(self.args.n_features, (self.args.rank + 1) * n_f)
            print("local features range = [{},{})".format(start_f, end_f))
        else:
            start_f = 0
            end_f = min(self.args.n_features, n_f)
            print("local features range = [{},{})".format(start_f, end_f))

        local_dist_start = time.time()
        local_dist = square_euclidean_np(self.data, test_data)
        local_dist_time = time.time() - local_dist_start

        sort_start = time.time()
        local_dist_ind = np.argsort(local_dist)
        sort_time = time.time() - sort_start

        # add -1 since rank 0 is master
        send_size = suggest_size(self.n_data, self.args.k, self.args.world_size - 1)
        # send_size = 1000
        print("suggest size by fagin = {}".format(send_size))
        send_ind = 0

        shuffle_index_time = 0
        gather_time = 0
        bc_time = 0
        count_time = 0
        reindex_time = 0
        trans_time = 0
        encr_time = 0
        split_time = 0

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
                coordinator_count_by_arr(new_lists, counts, top_k, self.args.k)
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
            for i, e in enumerate(counts):
                is_sample = random.random() < e / float(self.args.world_size - 1)
                if is_sample is True:
                    candidate_ids.append(i)
            n_candidate = len(candidate_ids)
            #print("top-k candidates = {}".format(candidate_ind))
            print("num of candidates = {}".format(n_candidate))
            dist.broadcast(torch.tensor(n_candidate), 0)
            dist.broadcast(torch.tensor(candidate_ids, dtype=torch.int32), 0)
        else:
            tmp_tensor = torch.tensor(0)
            dist.broadcast(tmp_tensor, 0)
            n_candidate = tmp_tensor.item()
            print("num of candidates = {}".format(n_candidate))
            tmp_tensor = torch.zeros([n_candidate], dtype=torch.int32)
            dist.broadcast(tmp_tensor, 0)
            candidate_ids = tmp_tensor.tolist()
            #print("top-k candidates = {}".format(candidate_ind))
            print("number of candidates = {}".format(n_candidate))
        sync_candidate_time = time.time() - sync_candidate_start

        max_msg_size = 80000000
        options = [('grpc.max_send_message_length', max_msg_size), ('grpc.max_receive_message_length', max_msg_size)]
        if rank != 0:
            # reindex candidates
            reindex_start = time.time()
            reindex_candidate_ids = self.shuffled2original[candidate_ids]
            reindex_time = time.time() - reindex_start

            # select local distances of k
            local_candidate_dist = local_dist[reindex_candidate_ids]

            n = 17800808502650258601875550958844515471709553355942844427678646632260820003199045314388404911994371628035329514822432294991493010990668660709752669595319311588926604853146533480048467772777445948194921231741018143470691643994153991869049452806017700509483577279858687223679450266912590905234105035466324132288032205654668616576048089785291612076679452672446088686439614576640913512366651332202016430324832498990806535377063538455898743073536813616879452457083048955093965336592637835994478510756622055811364251581599289127031126978248323101428581989623586684860664966965029305250191840964067180439265318568513663821573
            p = 116242551141358396967757172433944279550272901992454036802654774494190746770968998083776136351121804910854116390314290022188662376461418450157499142709114012730639393559056023381721752977836405940319438814897846391098115623373672289466369739591592235520707348103430597208678014448138590557989073256027414539843
            q = 153135046743798093992187848724462999193161952087785766943569652651932006292000925023754406052364365346195900521786925310092032955755059761206246049824001051042530418086950249290052967287714797696459181867415751414708390688671561087914604963522538318647976596983244525277965449248412149963925303550396761593111
            public_key = paillier.PaillierPublicKey(n=n)
            private_key = paillier.PaillierPrivateKey(public_key, p, q)

            encr_start = time.time()
            encr_local_top_k_dist = self.encrypt_data(local_candidate_dist, self.n_threads)
            encr_time = time.time() - encr_start

            split_start = time.time()
            self.split_data(encr_local_top_k_dist, self.n_servers)
            split_time = time.time() - split_start

            print("multi-thread transmission start")
            trans_start = time.time()
            tmp = self.multi_thread_trans(self.n_servers)
            trans_time = time.time() - trans_start
            print("multi-thread transmission end")

            global_candidate_dist = np.sqrt(tmp)
        else:
            global_candidate_dist = [10 for _ in range(cur_n_top)]

        if rank != 0:
            global_candidate_sort_ind = np.argsort(global_candidate_dist)[:k]
            global_top_k_ids = reindex_candidate_ids[global_candidate_sort_ind]
            print("indices of top-{} near neighbors = {}".format(k, global_top_k_ids))
            print("distance of top-{} near neighbors = {}".format(k, global_candidate_dist[global_candidate_sort_ind]))

            # calculate label
            count_label_start = time.time()
            target_count = [0 for _ in range(self.args.n_classes)]
            for j in global_top_k_ids:
                target_count[self.targets[j]] += 1
            count_label_time = time.time() - count_label_start
            pred_target = np.argmax(target_count)
            pred_prob = [i / float(k) for i in target_count]
            print("count of labels = {}".format(target_count))
            print("prob of labels = {}".format(pred_prob))

            homo_time = reindex_time + encr_time + split_time + trans_time

            print("find top-k finish: target = {}, prediction = {}, cost {:.2f} s,"
                  "comp dist cost {:.2f} s, sort cost {:.2f} s, shuffling index time {:.2f} s, "
                  "fagin cost {:.2f} s = gather cost {:.2f} s + broadcast cost {:.2f} s + count cost {:.2f} s, "
                  "sync candidates cost {:.2f} s, "
                  "homomorphic encryption cost {:.2f} s = reindex to real cost {:.2f} s + encryption cost {:.2f} s + "
                  "splitting cost {:.2f} s + transmission cost {:.2f} s, count labels cost {:.2f} s"
                  .format(test_target, pred_target, time.time() - start_time,
                          local_dist_time, sort_time, shuffle_index_time,
                          fagin_time, gather_time, bc_time, count_time,
                          sync_candidate_time, homo_time,
                          reindex_time, encr_time, split_time, trans_time, count_label_time))

            self.trans_data = []
            self.split_sizes = []
            self.global_data = []

            return pred_target, pred_prob
        else:
            return 0, [[0]] * k

