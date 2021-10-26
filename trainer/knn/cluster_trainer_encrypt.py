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

from transmission.pallier_cluster.cluster_client import Client
from transmission.encryption import encryptor


class ClusterTrainer(object):

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
        self.options = [('grpc.max_send_message_length', self.max_msg_size),
                        ('grpc.max_receive_message_length', self.max_msg_size)]
        self.trans_data_ids = []
        self.trans_data_cids = []
        self.trans_cids = []
        self.trans_data = []
        self.split_sizes = []
        self.ret_data = []

        self.addresses = ["bach11.ethz.ch:8991"]
        self.nThreads = 4
        self.nServers = len(self.addresses)
        assert self.nServers == 1

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

    def encryptorThread(self, data, public_key, q, i):
        encrypted = encryptor(data, public_key)
        encrypted.append(i)
        q.put(encrypted)

    def encryptData(self, data, nThreads):
        threads = []
        q=Queue()
        rets = []
        encryptedData= []

        size = len(data) // nThreads
        for i in range(nThreads):
            #Creation of threads to execute
            if i < nThreads - 1:
                split = data[size * i:size * (i + 1)]
                t = Process(target=self.encryptorThread, args=(split, self.public_key, q, i))
                threads.append(t)
            else:
                split = data[size * i:]
                t = Process(target=self.encryptorThread, args=(split, self.public_key, q, i))
                threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            ret = q.get()
            rets.append(ret)
        for t in threads:
            t.join()

        for i in range(nThreads):
            for elem in rets:
                if elem[-1] == i:
                    encryptedData.extend(elem[:-1])
        return encryptedData

    def transmit(self):
        addr = self.addresses[0]
        print("server addr: {}".format(addr))
        client = Client(addr, self.public_key, self.private_key)
        ret = client.transmit(self.trans_data_ids, self.trans_data_cids,
                              self.trans_cids, self.trans_data,  self.options)
        print("data returned by servers has size {}".format(len(ret)))
        self.reset_trans_data()
        return ret

    def reset_trans_data(self):
        self.trans_data_ids = []
        self.trans_data_cids = []
        self.trans_cids = []
        self.trans_data = []

    def find_top_k(self, test_data, test_target, k, n_cluster):
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

        shuffle_index_time = 0
        reindex_time = 0
        gather_time = 0
        bc_time = 0
        count_time = 0
        top_k = []

        counts = [0 for _ in range(self.n_data)]
        cur_n_top = 0
        cluster_ind = 0
        n_iter = 0
        rank = dist.get_rank()

        # use pre-shuffled id
        shuffle_index_start = time.time()
        if rank != 0:
            shuffled_local_dist_ind = self.origin2shuffled[local_dist_ind]
        else:
            shuffled_local_dist_ind = local_dist_ind
        shuffle_index_time = time.time() - shuffle_index_start

        send_size = suggest_size(self.n_data, self.args.k, self.args.world_size - 1)  # add -1, the first is master
        send_size = 100
        print("suggest batch size = {}".format(send_size))
        send_ind = 0

        # fagin step
        fagin_start = time.time()
        while cur_n_top < self.args.k and send_ind + send_size <= self.n_data:
            gather_start = time.time()
            start_ind = send_ind
            end_ind = min(self.n_data, send_ind + send_size)
            new_lists = gather(shuffled_local_dist_ind[start_ind:end_ind])
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
                cluster_ind += 1
                # dist.barrier()
            else:
                bc_start = time.time()
                tmp_tensor = torch.tensor(0)
                dist.broadcast(tmp_tensor, 0)
                bc_time += time.time() - bc_start
                cur_n_top = tmp_tensor.item()
                print("iter {}, scan {} rows, current top k = {}".format(n_iter, send_size, cur_n_top))
                cluster_ind += 1
                # dist.barrier()
        fagin_time = time.time() - fagin_start

        # sync candidates for top-k, i.e, the instances seen so far in fagin
        sync_candidate_start = time.time()
        n_candidate = 0
        candidate_ids = []
        if rank == 0:
            candidate_ids = [i for i, e in enumerate(counts) if e > 0]
            n_candidate = len(candidate_ids)
            # print("top-k candidates = {}".format(candidate_ind))
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
            # print("top-k candidates = {}".format(candidate_ind))
            print("num of candidates = {}".format(n_candidate))
        sync_candidate_time = time.time() - sync_candidate_start
        print("{} candidate indices (shuffled) = {}".format(len(candidate_ids), candidate_ids[:10]))

        max_msg_size = 80000000
        options = [('grpc.max_send_message_length', max_msg_size), ('grpc.max_receive_message_length', max_msg_size)]
        if rank != 0:
            # reindex candidates
            reindex_start = time.time()
            reindex_candidate_ind = self.shuffled2original[candidate_ids]
            print("{} candidate indices (original) = {}".format(len(reindex_candidate_ind), reindex_candidate_ind[:10]))
            reindex_time = time.time() - reindex_start

            # cluster candidates' local distance
            comp_centroid_start = time.time()
            candidate_local_dist = local_dist[reindex_candidate_ind]
            print("candidate local dist: {}".format(candidate_local_dist[:10]))

            # sort candidates' local distance
            # sorted index -> index of candidate_local_dist
            candidate_local_dist_ind = np.argsort(candidate_local_dist)
            print("max candidate local dist: {}".format(candidate_local_dist[candidate_local_dist_ind[-10:]]))

            centroids_dist = []
            candidate_cent_id = [0] * len(candidate_ids)
            cluster_size = len(candidate_ids) // n_cluster
            cluster_sizes = [cluster_size] * len(candidate_ids)
            if len(candidate_ids) % n_cluster != 0:
                cluster_size += 1
            print("number of clusters = {}, size of each clusters = {}".format(n_cluster, cluster_size))

            # compute centroids for candidates
            for i in range(n_cluster):
                start_ind = i * cluster_size
                end_ind = min((i + 1) * cluster_size, len(candidate_ids))
                if end_ind <= start_ind:
                    centroid = centroids_dist[-1]
                    print("cur cluster range: {} to {}".format(start_ind, end_ind))
                else:
                    cur_cluster_local_dist_ind = candidate_local_dist_ind[start_ind:end_ind]
                    # print("distance in cluster {} = {}".format(i, (candidate_local_dist[cur_cluster_local_dist_ind])[:10]))
                    centroid = np.mean(candidate_local_dist[cur_cluster_local_dist_ind])
                centroids_dist.append(centroid)
            for i in range(len(candidate_local_dist_ind)):
                candidate_ind = candidate_local_dist_ind[i]
                candidate_cent_id[candidate_ind] = i // cluster_size
            centroids_dist = np.array(centroids_dist)
            candidate_cent_id = np.array(candidate_cent_id)
            comp_centroid_time = time.time() - comp_centroid_start

            print("{} centroids: {}".format(len(centroids_dist), centroids_dist))
            print("candidates' centroid id: {}".format(candidate_cent_id[:10]))

            # shuffled candidate id
            self.trans_data_ids = candidate_ids

            # candidates' centroid ids
            # centroid id: rank+local_index
            self.trans_data_cids = ["{}_{}".format(self.args.rank, i) for i in candidate_cent_id]

            # centroids' ids
            self.trans_cids = ["{}_{}".format(self.args.rank, i) for i in np.arange(len(centroids_dist))]

            n = 17800808502650258601875550958844515471709553355942844427678646632260820003199045314388404911994371628035329514822432294991493010990668660709752669595319311588926604853146533480048467772777445948194921231741018143470691643994153991869049452806017700509483577279858687223679450266912590905234105035466324132288032205654668616576048089785291612076679452672446088686439614576640913512366651332202016430324832498990806535377063538455898743073536813616879452457083048955093965336592637835994478510756622055811364251581599289127031126978248323101428581989623586684860664966965029305250191840964067180439265318568513663821573
            p = 116242551141358396967757172433944279550272901992454036802654774494190746770968998083776136351121804910854116390314290022188662376461418450157499142709114012730639393559056023381721752977836405940319438814897846391098115623373672289466369739591592235520707348103430597208678014448138590557989073256027414539843
            q = 153135046743798093992187848724462999193161952087785766943569652651932006292000925023754406052364365346195900521786925310092032955755059761206246049824001051042530418086950249290052967287714797696459181867415751414708390688671561087914604963522538318647976596983244525277965449248412149963925303550396761593111
            public_key = paillier.PaillierPublicKey(n=n)
            private_key = paillier.PaillierPrivateKey(public_key, p, q)

            encr_start = time.time()
            self.trans_data = self.encryptData(centroids_dist, self.nThreads)
            encr_time = time.time() - encr_start  # Time taken to encrypt data

            #print("communication with server start")
            comm_server_start = time.time()
            tmp = self.transmit()
            comm_server_time = time.time() - comm_server_start
            #print("communication with server end")
            #print("global distance: shape {}, {}".format(len(tmp), tmp))

            global_candidate_dist = np.sqrt(np.asarray(tmp, dtype=np.float32))
            print("minimal global distance: {}".format(min(global_candidate_dist)))

            select_top_k_start = time.time()
            # this is shuffled id
            # top_k_ind = least_indices(global_candidate_dist, self.args.k)
            top_k_sort_ind = np.argsort(global_candidate_dist)[:self.args.k]
            top_k_shuffled_ids = np.array(candidate_ids)[top_k_sort_ind]
            print("top-k indices (shuffled) = {}".format(top_k_shuffled_ids))
            top_k_ids = self.shuffled2original[top_k_shuffled_ids]
            print("top-k indices (original) = {}".format(top_k_ids))
            top_k_dist = global_candidate_dist[top_k_sort_ind]
            select_top_k_time = time.time() - select_top_k_start
            print("indices of k near neighbors = {}".format(top_k_ids))
            print("distance of k near neighbors = {}".format(top_k_dist))

            # calculate label
            count_label_start = time.time()
            label_count = [0 for _ in range(self.args.n_classes)]
            for nid in top_k_ids:
                label_count[self.targets[nid]] += 1
            pred_target = np.argmax(label_count)
            pred_prob = [i / float(k) for i in label_count]
            #print("label counts = {}".format(label_count))
            #print("prob of labels = {}".format(pred_prob))
            count_label_time = time.time() - count_label_start

            if rank != 0:
                print("find top-k finish: target = {}, prediction = {}, cost {:.2f} s, "
                      "comp dist cost {:.2f} s, sort cost {:.2f} s, shuffle index cost {:.2f} s, "
                      "fagin cost {:.2f} s = gather cost {:.2f} s + broadcast cost {:.2f} s + count cost {:.2f} s, "
                      "sync candidate cost {:.2f} s, reindex cost {:.2f} s, comp centroids cost {:.2f} s, "
                      "encrypt cost {:.2f} s, comm server cost {:.2f} s, "
                      "select top-k cost {:.2f} s, count label cost {:.2f} s"
                      .format(test_target, pred_target, time.time() - start_time,
                              local_dist_time, sort_time, shuffle_index_time,
                              fagin_time, gather_time, bc_time, count_time,
                              sync_candidate_time, reindex_time, comp_centroid_time,
                              encr_time, comm_server_time, select_top_k_time, count_label_time))

            return pred_target, pred_prob
        else:
            print("master (rank 0) is waiting for other")
            time.sleep(self.args.k)
            return 0, [[0]] * k
