#TODO clean , same in cluster DONE
import time
from multiprocessing import Queue

import numpy as np
from phe import paillier
from utils.distance import square_euclidean_np
from multiprocessing import Process
from transmission.encryption import encryptor
from transmission.encryption import listEncryptor

from transmission.client import Client

#daredevil1205!

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
        self.transData = []
        self.splitSizes = []
        self.globalData = []
        #List of addresses to connect - Servers
        self.addresses = ["bach13.ethz.ch:8991","bach14.ethz.ch:8991","bach15.ethz.ch:8991","bach16.ethz.ch:8991","bach13.ethz.ch:8992","bach14.ethz.ch:8992","bach15.ethz.ch:8992","bach16.ethz.ch:8992","bach13.ethz.ch:8993","bach14.ethz.ch:8993","bach15.ethz.ch:8993","bach16.ethz.ch:8993","bach13.ethz.ch:8994","bach14.ethz.ch:8994","bach15.ethz.ch:8994","bach16.ethz.ch:8994"]


    def transmit(self,q, i):
        addr=self.addresses[i]
        spliSize = self.splitSizes[i]
        #print("server {}, addr: {}, splitSize:{}". format(i,addr, spliSize))
        client = Client(self.addresses[i], self.public_key, self.private_key)
        tmp = client.transmit(self.transData[i], self.splitSizes[i], self.options)
        tmp.append(i)
        q.put(tmp)

    def encryptorThread(self,data, public_key, q, i):
        #print("Thread {} is encrypting")
        encrypted = listEncryptor(data, public_key)
        encrypted.append(i)
        q.put(encrypted)




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
                    print("last elem {}".format(elem[-1]))
                    encryptedData.extend(elem[:-1])
        return encryptedData

    def multiThreadTransm(self, nServers):
        q = Queue()
        processes = []
        rets = []

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
                    self.globalData.extend((elem[:-1]))
        for elem in rets:
            print(elem[:5])



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
        local_dist_time = time.time() - local_dist_start

        encrypt_start = time.time()
        #Define number of Servers to connect to
        nServers = 16


        encrypt_data_time = 0
        noSplit = True
        if noSplit:
            encrDataS = time.time()
            #Define number of threads you want to create to encrypt data
            nThreads = 16
            encryptedData = self.encryptData(local_dist, nThreads)
            self.splitData(encryptedData, nServers)
            self.multiThreadTransm(nServers)
            encrypt_data_time = time.time() - encrDataS
        else:
            cuts_number = 10
            data_size = len(local_dist)
            cut = data_size // cuts_number
            encrDataS = time.time()
            for i in range(cuts_number):
                if i < cuts_number - 1:
                    data_split = local_dist[cut * i:cut * (i + 1)]
                    print("current cut size: {}".format(len(data_split)))
                    encr_split = self.encryptData(data_split, 16)
                    self.splitData(encr_split, nServers)
                    print("Split Encrypted")
                    self.multiThreadTransm(nServers)
                    print("Split {} done".format(i))
                    self.transData = []
                    self.splitSizes = []
                    print("Current global data size : {}".format(len(self.globalData)))
                    encrypt_data_time += time.time() - encrDataS
                else:
                    data_split = local_dist[cut * i:]
                    print("current cut size: {}".format(len(data_split)))
                    # Define number of threads you want to create to encrypt data
                    nThreads = 16
                    encr_split = self.encryptData(data_split, nThreads)
                    print("Split Encrypted")
                    self.splitData(encr_split, nServers)
                    self.multiThreadTransm(nServers)
                    print("Split {} done".format(i))
                    self.transData = []
                    self.splitSizes = []
                    print("Current global data size : {}".format(len(self.globalData)))
                    encrypt_data_time += time.time() - encrDataS
            print("End loop for, globalData size = {}".format(len(self.globalData)))



        encrypt_time = time.time() - encrypt_start
        sqrt_comp_start = time.time()

        global_dist = np.sqrt(self.globalData)



        sqrt_comp_time = time.time() - sqrt_comp_start


        select_top_start = time.time()
        ind_sorted = np.argsort(global_dist)
        ind_k = ind_sorted[:self.args.k]



        dist_k = global_dist[ind_k]
        select_top_time = time.time() - select_top_start
        print("indices of k near neighbor = {}".format(ind_k))
        print("distance of k near neighbor = {}".format(dist_k))

        # calculate label
        label_count = [0 for _ in range(self.args.n_classes)]
        for nid in ind_k:
            label_count[self.targets[nid]] += 1
        pred_target = np.argmax(label_count)
        print("label counts are")
        print(label_count)





        print("find top-k finish: target = {}, prediction = {}, total cost {:.2f} s, "
              "compute local dist cost {:.2f} s, encrypted communication cost {:.2f} s, "
              "sqrt computation cost {:.2f} s, "
              "select top-k cost {:.2f} s, encrypting data cost {}"
              .format(test_target, pred_target, time.time() - start_time,
                      local_dist_time, encrypt_time,
                      sqrt_comp_time,
                      select_top_time, encrypt_data_time))
        self.globalData = []
        self.transData = []

        return pred_target
