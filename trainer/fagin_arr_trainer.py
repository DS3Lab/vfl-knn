#TODO  clean + same in CLUSTER DONE
import time
import sys

import numpy as np
import torch
import torch.distributed as dist
from phe import paillier
sys.path.append("../")
from utils.distance import square_euclidean_np
from utils.comm_op import gather
from utils.fagin_utils import suggest_size, count_by_arr, createLookUpTable, get_real_ind, get_shuffled_ind
from transmission.client import Client
from multiprocessing import Queue
from multiprocessing import Process
from transmission.encryption import encryptor

class FaginArrTrainer(object):

    def __init__(self, args, data, targets):
        self.args = args
        self.data = data
        self.n_data = len(data)
        self.targets = targets
        #n, p and q define public and private key for clients
        self.n = 17800808502650258601875550958844515471709553355942844427678646632260820003199045314388404911994371628035329514822432294991493010990668660709752669595319311588926604853146533480048467772777445948194921231741018143470691643994153991869049452806017700509483577279858687223679450266912590905234105035466324132288032205654668616576048089785291612076679452672446088686439614576640913512366651332202016430324832498990806535377063538455898743073536813616879452457083048955093965336592637835994478510756622055811364251581599289127031126978248323101428581989623586684860664966965029305250191840964067180439265318568513663821573
        self.p = 116242551141358396967757172433944279550272901992454036802654774494190746770968998083776136351121804910854116390314290022188662376461418450157499142709114012730639393559056023381721752977836405940319438814897846391098115623373672289466369739591592235520707348103430597208678014448138590557989073256027414539843
        self.q = 153135046743798093992187848724462999193161952087785766943569652651932006292000925023754406052364365346195900521786925310092032955755059761206246049824001051042530418086950249290052967287714797696459181867415751414708390688671561087914604963522538318647976596983244525277965449248412149963925303550396761593111
        self.public_key = paillier.PaillierPublicKey(n=self.n)
        self.private_key = paillier.PaillierPrivateKey(self.public_key, self.p, self.q)
        self.max_msg_size = 800000000
        self.options = [('grpc.max_send_message_length', self.max_msg_size), ('grpc.max_receive_message_length', self.max_msg_size)]
        self.transData = []
        self.splitSizes = []
        self.global_data = []
        self.addresses = ["bach13.ethz.ch:8991","bach14.ethz.ch:8991","bach15.ethz.ch:8991","bach16.ethz.ch:8991","bach13.ethz.ch:8992","bach14.ethz.ch:8992","bach15.ethz.ch:8992","bach16.ethz.ch:8992","bach13.ethz.ch:8993","bach14.ethz.ch:8993","bach15.ethz.ch:8993","bach16.ethz.ch:8993","bach13.ethz.ch:8994","bach14.ethz.ch:8994","bach15.ethz.ch:8994","bach16.ethz.ch:8994"]

    def encryptorThread(self, data, public_key, q, i):
        encrypted = encryptor(data, public_key)
        encrypted.append(i)
        q.put(encrypted)

    def encryptData(self, data, nThreads):
        threads = []
        q=Queue()
        rets = []
        encryptedData= []

        size = len(data)//nThreads
        for i in range(nThreads):
            #Creation of threads to execute
            if i < nThreads - 1:
                split = data[size*i:size*(i+1)]
                t=Process(target=self.encryptorThread, args=(split, self.public_key, q, i))
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

    def splitData(self, data, nSplits):

        size = len(data)//nSplits
        for i in range(nSplits):

            if i < nSplits - 1:
                encrData = data[size*i:size*(i+1)]
                self.transData.append(encrData)
                self.splitSizes.append(len(self.transData[i]))
            else:
                encrData = data[size * i:]
                self.transData.append(encrData)
                self.splitSizes.append(len(self.transData[i]))


    def transmit(self,q, i):
        addr=self.addresses[i]
        spliSize = self.splitSizes[i]
        print("server {}, addr: {}, splitSize:{}". format(i,addr, spliSize))
        client = Client(self.addresses[i], self.public_key, self.private_key)
        tmp = client.transmit(self.transData[i], self.splitSizes[i], self.options)
        tmp.append(i)
        q.put(tmp)


    def multiThreadTransm(self, nServers):
        q = Queue()
        processes = []
        rets = []
        global_data = []

        for i in range(nServers):
            t = Process(target=self.transmit, args=(q, i))
            processes.append(t)
        for p in processes:
            p.start()
        for p in processes:
            ret = q.get()
            rets.append(ret)
        for p in processes:
            p.join()


        for i in range(nServers):
            for elem in rets:
                if elem[-1] == i:
                    global_data.extend((elem[:-1]))
        for elem in rets:
            print(elem[:5])

        print(len(global_data))
        print("data returned by servers has size {}".format(len(global_data)))
        return global_data

    def find_top_k(self, test_data, test_target, k):
        start_time = time.time()
        print(">>> start find top-{} <<<".format(k))


        n_f = int(self.args.n_features / self.args.world_size-1) # We add -1, in order to pass 11 as world_size
        if self.args.n_features % self.args.world_size-1 != 0:
            n_f += 1

        # local feature range
        # Each machine takes the range of data according to their rank
        if self.args.rank < self.args.world_size-1:
            print("rank dist= {}, rank = {}".format(dist.get_rank(), self.args.rank))
            start_f = self.args.rank * n_f
            end_f = min(self.args.n_features, (self.args.rank + 1) * n_f)
            print("local features range = [{},{})".format(start_f, end_f))
        else:
            start_f = 0 * n_f
            end_f = min(self.args.n_features, (0 + 1) * n_f)
            print("local features range = [{},{})".format(start_f, end_f))

        local_dist_start = time.time()
        local_dist = square_euclidean_np(self.data, test_data)
        local_dist_time = time.time() - local_dist_start

        sort_start = time.time()
        local_dist_ind = np.argsort(local_dist)
        sort_time = time.time() - sort_start

        send_size = suggest_size(self.n_data, self.args.k, self.args.world_size-1)# We add -1, in order to pass 11 as world_size
        send_ind = 0

        shuffle_time = 0
        gather_time = 0
        bc_time = 0
        count_time = 0
        map_time = 0
        transm_end = 0
        encr_end=0
        split_end=0
        top_k = []
        counts = [0 for _ in range(self.n_data)]
        cur_n_top = 0
        n_iter = 0
        rank = dist.get_rank()

        if rank != 0:
            shuffle_start = time.time()
            #Provide seed to create lookup table
            seed = 42
            lookUpTable = createLookUpTable(self.n_data, seed)
            shuffled_local_dist_ind = get_shuffled_ind(local_dist_ind, lookUpTable)
            shuffle_time = time.time() - shuffle_start
        else:
            shuffled_local_dist_ind = local_dist_ind



        while cur_n_top < self.args.k and send_ind <= self.n_data:
            gather_start = time.time()
            new_lists = gather(shuffled_local_dist_ind[send_ind:min(self.n_data, send_ind + send_size)])
            gather_time += time.time() - gather_start
            send_ind += send_size

            if rank == 0:
                count_start = time.time()
                count_by_arr(new_lists, counts, top_k, self.args.k)
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

        # sync top-k indices
        if rank == 0:
            bc_start = time.time()
            cur_n_top = len(top_k)
            dist.broadcast(torch.tensor(cur_n_top), 0)
            top_k_tensor = torch.tensor(top_k, dtype=torch.int32)
            dist.broadcast(top_k_tensor, 0)
            bc_time += time.time() - bc_start
        else:
            bc_start = time.time()
            tmp_tensor = torch.tensor(0)
            dist.broadcast(tmp_tensor, 0)
            cur_n_top = tmp_tensor.item()
            tmp_tensor = torch.zeros([cur_n_top], dtype=torch.int32)
            dist.broadcast(tmp_tensor, 0)
            top_k = tmp_tensor.tolist()
            bc_time += time.time() - bc_start




        if rank != 0:
            map_start = time.time()
            top_k = get_real_ind(top_k, lookUpTable)
            map_time = time.time() - map_start


        # we select local distances of k
        local_top_k_dist = local_dist[top_k[:self.args.k]]


        max_msg_size = 80000000
        options = [('grpc.max_send_message_length', max_msg_size), ('grpc.max_receive_message_length', max_msg_size)]
        addr = "bach13.ethz.ch:8991"
        if rank != 0:
            n = 17800808502650258601875550958844515471709553355942844427678646632260820003199045314388404911994371628035329514822432294991493010990668660709752669595319311588926604853146533480048467772777445948194921231741018143470691643994153991869049452806017700509483577279858687223679450266912590905234105035466324132288032205654668616576048089785291612076679452672446088686439614576640913512366651332202016430324832498990806535377063538455898743073536813616879452457083048955093965336592637835994478510756622055811364251581599289127031126978248323101428581989623586684860664966965029305250191840964067180439265318568513663821573
            p = 116242551141358396967757172433944279550272901992454036802654774494190746770968998083776136351121804910854116390314290022188662376461418450157499142709114012730639393559056023381721752977836405940319438814897846391098115623373672289466369739591592235520707348103430597208678014448138590557989073256027414539843
            q = 153135046743798093992187848724462999193161952087785766943569652651932006292000925023754406052364365346195900521786925310092032955755059761206246049824001051042530418086950249290052967287714797696459181867415751414708390688671561087914604963522538318647976596983244525277965449248412149963925303550396761593111
            public_key = paillier.PaillierPublicKey(n=n)
            private_key = paillier.PaillierPrivateKey(public_key, p, q)

            encr_start = time.time()
            print("Multithreaded Trasmission started")
            #Define number of threads to use for encryption
            nThreads = 16
            encr_local_top_k_dist = self.encryptData(local_top_k_dist,nThreads)
            encr_end = time.time() - encr_start #Time taken to encrypt data
            split_start = time.time()
            #Define number of server to use for transmission
            nServers = 1
            self.splitData(encr_local_top_k_dist, nServers)
            split_end = time.time() - split_start
            transm_start = time.time()
            tmp = self.multiThreadTransm(nServers)
            transm_end = time.time() - transm_start
            print("Multithreaded Transmission ended ")
            global_top_k_dist = np.sqrt(tmp)
        else:
            global_top_k_dist = [10 for _ in range(cur_n_top)]
            #We need to make wait the fagins coordinator if we have a big k
            print("waiting for other")
            if self.args.k > 500:
                time.sleep(self.args.k * 3)
            print("started again at time = {}".format(time.time()-start_time))


        print("distance of {} top near neighbor = {}".format(k,global_top_k_dist[:10]))
        # calculate label
        count_start = time.time()
        target_count = [0 for _ in range(self.args.n_classes)]
        for j in top_k:
            target_count[self.targets[j]] += 1
        count_time = time.time()-count_start
        pred_target = np.argmax(target_count)

        print("Count of labels")
        print(target_count)

        encrypt_time = shuffle_time + map_time + encr_end + split_end + transm_end

        print("find top-k finish: target = {}, prediction = {}, cost {:.2f} s,"
              "comp dist cost {:.2f} s, sort cost {:.2f} s, broadcast cost {:.2f}, gather cost {:.2f}, "
              "map to real cost {:.2f}, encryption cost {:.2f}, splitting cost {:.2f}, transmission cost {:.2f}"
              "count labels cost {:.2f}, shuffling time {:.2f}, homomorphic encr {:.2f}"
              .format(test_target, pred_target, time.time() - start_time,
                      local_dist_time, sort_time, bc_time, gather_time,
                      map_time, encr_end, split_end, transm_end,
                      count_time, shuffle_time, encrypt_time))

        return pred_target