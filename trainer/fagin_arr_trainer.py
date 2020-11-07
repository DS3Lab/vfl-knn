import time
import sys
import math

import numpy as np
import torch
import torch.distributed as dist
from phe import paillier
sys.path.append("../")
from utils.distance import square_euclidean_np
from utils.comm_op import gather, sum_sqrt_all_reduce
from utils.fagin_utils import suggest_size, count_by_arr, createLookUpTable, get_real_ind, get_shuffled_ind
from transmission.client import Client

PER_ENCRYPT_TIME = 0.01
FLOAT_BYTES = 4
HOMOMORPHIC_BYTES = 4096 * 4


class FaginArrTrainer(object):

    def __init__(self, args, data, targets):
        self.args = args
        self.data = data
        self.n_data = len(data)
        self.targets = targets

    def find_top_k(self, test_data, test_target, k):
        start_time = time.time()
        print(">>> start find top-{} <<<".format(k))

        # ATTENTION we use world_size to cut the data

        # average number of features in one part
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
        # print("local distance size = {}, values = {}".format(len(local_dist), local_dist[:10]))
        local_dist_time = time.time() - local_dist_start

        sort_start = time.time()
        local_dist_ind = np.argsort(local_dist)
        # print("local dist index = {}".format(local_dist_ind[:10]))
        # print("local dist = {}".format(local_dist[local_dist_ind[:10]]))
        sort_time = time.time() - sort_start

        send_size = suggest_size(self.n_data, self.args.k, self.args.world_size-1)# We add -1, in order to pass 11 as world_size
        #print("suggest batch size = {}".format(send_size))
        send_ind = 0

        comm_start = time.time()
        # each encryption cost 10ms
        gather_time = 0
        bc_time = 0
        count_time = 0
        top_k = []
        counts = [0 for _ in range(self.n_data)]
        cur_n_top = 0
        n_iter = 0
        rank = dist.get_rank()

        if rank != 0: # if client
            lookUpTable = createLookUpTable([i for i in range(self.n_data)])
            shuffled_local_dist_ind = get_shuffled_ind(local_dist_ind, lookUpTable)
        else:
            shuffled_local_dist_ind = local_dist_ind



        while cur_n_top < self.args.k and send_ind <= self.n_data:
            gather_start = time.time()
            new_lists = gather(shuffled_local_dist_ind[send_ind:min(self.n_data, send_ind + send_size)])
            gather_time += time.time() - gather_start
            send_ind += send_size

            if rank == 0:
                # we go voer all list of ind sent by others and count
                count_start = time.time()
                count_by_arr(new_lists, counts, top_k, self.args.k)
                count_time += time.time() - count_start
                bc_start = time.time()
                cur_n_top = len(top_k)
                # we send the number of top elements we have
                # once cur_n_top > top_k ---->exit while
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
            # we send number of k best
            # and their ids
            dist.broadcast(torch.tensor(cur_n_top), 0)
            top_k_tensor = torch.tensor(top_k, dtype=torch.int32)
            #print("top-k tensor: shape = {}, value = {}".format(top_k_tensor.type(), top_k_tensor.shape, top_k_tensor))
            dist.broadcast(top_k_tensor, 0)
            bc_time += time.time() - bc_start
        else:
            bc_start = time.time()
            tmp_tensor = torch.tensor(0)
            dist.broadcast(tmp_tensor, 0)
            # we receive top k number
            # and their indicators - Here no problem since we will be able to decrypt
            cur_n_top = tmp_tensor.item()
            tmp_tensor = torch.zeros([cur_n_top], dtype=torch.int32)
            dist.broadcast(tmp_tensor, 0)
            #print("top-k tensor: shape = {}, value = {}".format(tmp_tensor.type(), tmp_tensor.shape, tmp_tensor))
            top_k = tmp_tensor.tolist()
            bc_time += time.time() - bc_start

        encrypt_start = time.time()
        # time.sleep(PER_ENCRYPT_TIME * send_size)
        encrypt_time = PER_ENCRYPT_TIME * self.args.k


        if rank != 0:
            print("shuffled top values: {}".format(top_k))
            top_k = get_real_ind(top_k, lookUpTable)
            print("real top values: {}".format(top_k))

        dist_comm_start = time.time()
        #print("total number of current top k = {}".format(cur_n_top))
        print("indices of {} near neighbors = {}".format(self.args.k, top_k[:self.args.k]))
        # we select local distances of k
        local_top_k_dist = local_dist[top_k[:self.args.k]]
        print("local dist of {} near neighbors = {}".format(self.args.k, local_top_k_dist))
        # same method as allReduce so no problem to implements

        # TRANSMISSION---------------------------------------------
        max_msg_size = 80000000
        options = [('grpc.max_send_message_length', max_msg_size), ('grpc.max_receive_message_length', max_msg_size)]
        addr = self.args.init_method
        addr = addr.replace("tcp://", "")
        addr = addr.replace("8888", "8999")
        if rank != 0:
            n = 17800808502650258601875550958844515471709553355942844427678646632260820003199045314388404911994371628035329514822432294991493010990668660709752669595319311588926604853146533480048467772777445948194921231741018143470691643994153991869049452806017700509483577279858687223679450266912590905234105035466324132288032205654668616576048089785291612076679452672446088686439614576640913512366651332202016430324832498990806535377063538455898743073536813616879452457083048955093965336592637835994478510756622055811364251581599289127031126978248323101428581989623586684860664966965029305250191840964067180439265318568513663821573
            p = 116242551141358396967757172433944279550272901992454036802654774494190746770968998083776136351121804910854116390314290022188662376461418450157499142709114012730639393559056023381721752977836405940319438814897846391098115623373672289466369739591592235520707348103430597208678014448138590557989073256027414539843
            q = 153135046743798093992187848724462999193161952087785766943569652651932006292000925023754406052364365346195900521786925310092032955755059761206246049824001051042530418086950249290052967287714797696459181867415751414708390688671561087914604963522538318647976596983244525277965449248412149963925303550396761593111
            public_key = paillier.PaillierPublicKey(n=n)
            private_key = paillier.PaillierPrivateKey(public_key, p, q)
            print("Client parameters are:")
            print("addr: {}, public_key: {}, private_key: {}, local_dist: {}, split_size: {}".format(addr, public_key,
                                                                                                     private_key,
                                                                                                     local_top_k_dist[:5],
                                                                                                     cur_n_top))
            client = Client(addr, public_key, private_key)
            tmp = client.transmit(local_top_k_dist, cur_n_top, options)
            global_top_k_dist = np.sqrt(tmp)
        else:
            global_top_k_dist = [10 for _ in range(cur_n_top)]
        #global_top_k_dist = sum_sqrt_all_reduce(local_top_k_dist)
        # END TRANSMISSION----------------------------------------------------------
        print("distance of {} near neighbor = {}".format(self.args.k, global_top_k_dist))
        float_dist_comm_time = time.time() - dist_comm_start

        homo_dist_comm_start = time.time()
        homo_dist = np.array([local_top_k_dist for _ in range(int(HOMOMORPHIC_BYTES / FLOAT_BYTES))])
        print("homo tensor size = {}".format(homo_dist.shape))
        homo_dist = sum_sqrt_all_reduce(homo_dist)
        # homo_dist_comm_time = HOMOMORPHIC_BYTES / FLOAT_BYTES * float_dist_comm_time
        # time.sleep(homo_dist_comm_time - float_dist_comm_time)
        homo_dist_comm_time = time.time() - homo_dist_comm_start

        # calculate label
        target_count = [0 for _ in range(self.args.n_classes)]
        for j in top_k:
            target_count[self.targets[j]] += 1
        pred_target = np.argmax(target_count)

        print("find top-k finish: target = {}, prediction = {}, cost {:.2f} s, "
              "comp dist cost {:.2f} s, sort cost {:.2f} s, encryption cost {:.2f} s, count cost {:.2f} s, "
              "communication cost {:.2f} s = gather cost {:.2f} s + broadcast cost {:.2f} s "
              "+ float dist cost {:.2f} s + homo dist cost {:.2f} s"
              .format(test_target, pred_target,
                      time.time() - start_time + encrypt_time,
                      local_dist_time, sort_time, encrypt_time, count_time,
                      gather_time + bc_time + float_dist_comm_time + homo_dist_comm_time,
                      gather_time, bc_time, float_dist_comm_time, homo_dist_comm_time))

        return pred_target
