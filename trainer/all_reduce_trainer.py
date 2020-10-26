import time

import numpy as np
import math
from phe import paillier
from utils.distance import square_euclidean_np
from utils.comm_op import sum_sqrt_all_reduce

from transmission.client import Client
from transmission.server import broadcast


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

        # local_dist is the array we have to encrypt and send to server
        # We consider rank=0 -> server
        num_dist = len(local_dist)
        size = num_dist
        split_size = num_dist  # if ==num_dist -> no splitting of data
        max_msg_size = 80000000
        options = [('grpc.max_send_message_length', max_msg_size), ('grpc.max_receive_message_length', max_msg_size)]
        addr = self.args.init_method
        addr = addr.replace("tcp://", "")
        if self.args.rank == 20:
            num_clients = self.args.world_size - 1  # since server is counted in world size
            n = 17800808502650258601875550958844515471709553355942844427678646632260820003199045314388404911994371628035329514822432294991493010990668660709752669595319311588926604853146533480048467772777445948194921231741018143470691643994153991869049452806017700509483577279858687223679450266912590905234105035466324132288032205654668616576048089785291612076679452672446088686439614576640913512366651332202016430324832498990806535377063538455898743073536813616879452457083048955093965336592637835994478510756622055811364251581599289127031126978248323101428581989623586684860664966965029305250191840964067180439265318568513663821573
            public_key = paillier.PaillierPublicKey(n=n)
            expExch = math.ceil(size / split_size)
            print("Server parameters are:")
            print("addr: {}, numClients: {}, public_key: {}, local_dist: {}, split_size: {}, expExch: {}".format(addr, num_clients, public_key, local_dist[:5], split_size, expExch))
            print("Start broadcasting")
            broadcast(addr,num_clients,public_key,local_dist, split_size, expExch)
            global_dist = [1,2,3,4]
        else:
            n = 17800808502650258601875550958844515471709553355942844427678646632260820003199045314388404911994371628035329514822432294991493010990668660709752669595319311588926604853146533480048467772777445948194921231741018143470691643994153991869049452806017700509483577279858687223679450266912590905234105035466324132288032205654668616576048089785291612076679452672446088686439614576640913512366651332202016430324832498990806535377063538455898743073536813616879452457083048955093965336592637835994478510756622055811364251581599289127031126978248323101428581989623586684860664966965029305250191840964067180439265318568513663821573
            p = 116242551141358396967757172433944279550272901992454036802654774494190746770968998083776136351121804910854116390314290022188662376461418450157499142709114012730639393559056023381721752977836405940319438814897846391098115623373672289466369739591592235520707348103430597208678014448138590557989073256027414539843
            q = 153135046743798093992187848724462999193161952087785766943569652651932006292000925023754406052364365346195900521786925310092032955755059761206246049824001051042530418086950249290052967287714797696459181867415751414708390688671561087914604963522538318647976596983244525277965449248412149963925303550396761593111
            public_key = paillier.PaillierPublicKey(n=n)
            private_key = paillier.PaillierPrivateKey(public_key, p, q)
            print("Client parameters are:")
            print("addr: {}, public_key: {}, private_key: {}, local_dist: {}, split_size: {}".format(addr, public_key, private_key, local_dist[:5], split_size))
            client = Client(addr,public_key,private_key)
            tmp = client.transmit(local_dist,split_size,options)
            global_dist = np.sqrt(tmp)




        # print("local distance size = {}, values = {}".format(len(local_dist), local_dist[:10]))
        local_dist_time = time.time() - local_dist_start

        encrypt_start = time.time()
        # time.sleep(PER_ENCRYPT_TIME * self.n_data)
        encrypt_time = PER_ENCRYPT_TIME * self.n_data

        dist_comm_start = time.time()
        # global_dist = sum_sqrt_all_reduce(local_dist)
        # print("global distance size = {}, values = {}".format(len(global_dist), global_dist[:10]))
        float_dist_comm_time = time.time() - dist_comm_start

        """
        homo_dist_comm_start = time.time()
        homo_size_factor = int(HOMOMORPHIC_BYTES / FLOAT_BYTES)
        print("do {} times homo communication".format(int(homo_size_factor / 4)))
        #for _ in range(int(homo_size_factor / 4)):
        homo_dist = [local_dist for _ in range(4)]
        homo_dist = sum_sqrt_all_reduce(np.asarray(homo_dist))
        one_homo_time = time.time() - homo_dist_comm_start
        print("send one homo tensor, size = {}, cost {} s".format(len(homo_dist), one_homo_time))
        homo_dist_comm_time = int(homo_size_factor / 4) * one_homo_time
        """

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
              "float communication cost {:.2f} s, "
              "select top-k cost {:.2f} s"
              .format(test_target, pred_target,
                      time.time() - start_time + encrypt_time - float_dist_comm_time,
                      local_dist_time, encrypt_time, float_dist_comm_time, select_top_time))

        return pred_target
