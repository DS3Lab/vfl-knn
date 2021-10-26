import time
import math
from multiprocessing import Queue

import numpy as np
from phe import paillier
from multiprocessing import Process
from transmission.encryption import listEncryptor

from transmission.pallier.client import Client


class LRTrainer(object):

    def __init__(self, args):
        self.args = args

        # average number of features in one part
        self.n_f = self.args.n_features // self.args.world_size
        if self.args.n_features % self.args.world_size != 0:
            self.n_f += 1

        # local feature range
        self.start_f = self.args.rank * self.n_f
        self.end_f = min(self.args.n_features, (self.args.rank + 1) * self.n_f)
        print("local features range = [{},{})".format(self.start_f, self.end_f))

        # init local weight
        self.W = np.random.rand(self.n_f)

        # for Adam
        self.n_step = 0
        self.adam_lr = np.zeros_like(self.W)
        self.adam_lr.fill(self.args.lr)
        self.m = np.zeros_like(self.W)
        self.v = np.zeros_like(self.W)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8

        # n, p and q define public and private key for clients
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

        self.addresses = ["bach11.ethz.ch:8991"]
        self.n_threads = 1
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
        #print("len data: {}, nSplits: {}, split size: {}".format(len(data), n_splits, size))
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

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def update_adam(self, grad):
        self.n_step += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.power(grad, 2)
        v_hat = self.v / (1 - math.pow(self.beta2, self.n_step))
        self.adam_lr = self.args.lr / (np.sqrt(v_hat) + self.epsilon)
        #print("step {}, adam learning rate: {}".format(self.n_step, self.adam_lr))

    def reset_data(self):
        self.trans_data = []
        self.split_sizes = []
        self.global_data = []

    def one_iteration(self, n_epoch, n_batch, X, Y):
        start_time = time.time()
        np.set_printoptions(precision=4)
        np.set_printoptions(threshold=2)

        partial_Z = X @ self.W
        # print("partial dot = {}".format(partial_dot))

        # homomorphic transmission
        homo_start = time.time()

        # encryption
        encr_start = time.time()
        encrypted_Z = self.encrypt_data(partial_Z, self.n_threads)
        encr_time = time.time() - encr_start

        # split data
        split_start = time.time()
        self.split_data(encrypted_Z, self.n_servers)
        split_time = time.time() - split_start

        # send to server
        trans_start = time.time()
        self.multi_thread_trans(self.n_servers)
        trans_time = time.time() - trans_start
        homo_time = time.time() - homo_start

        h = self.sigmoid(np.asarray(self.global_data))
        grads = (X.T * (h - Y)).T
        grad = np.mean(grads, axis=0) + self.args.lam * self.W

        # Vanilla SGD
        # cur_lr = self.args.lr * math.pow(0.9, n_epoch)
        # self.weight = self.weight - cur_lr * grad

        # Adam
        self.update_adam(grad)
        m_hat = self.m / (1 - math.pow(self.beta1, self.n_step))
        self.W = self.W - self.adam_lr * m_hat

        loss = np.average(- Y * np.log(h) - (1 - Y) * np.log(1 - h))
        self.reset_data()

        # sum regularization
        partial_reg = np.linalg.norm(self.W, 2)
        # print("partial reg = {}".format(partial_reg))

        # homomorphic transmission
        homo_start = time.time()
        # encryption
        encr_start = time.time()
        encrypted_reg = self.encrypt_data([partial_reg], self.n_threads)
        encr_time += time.time() - encr_start

        # split data
        split_start = time.time()
        self.split_data(encrypted_reg, self.n_servers)
        split_time += time.time() - split_start

        # send to server
        trans_start = time.time()
        self.multi_thread_trans(self.n_servers)
        trans_time += time.time() - trans_start
        homo_time += time.time() - homo_start

        loss += self.args.lam / 2. * self.global_data[0]
        self.reset_data()

        iter_time = time.time() - start_time

        print("iteration[{}] finish, cost {:.2f} s, encryption cost {:.2f} s, server comm cost {:.2f} s, "
              "grad = {}, loss = {:.6f}, weight = {}"
              .format(n_batch, iter_time, encr_time, trans_time, grad, loss, self.W))

        return loss

    def predict_one(self, X):
        start_time = time.time()

        partial_Z = X @ self.W

        # homomorphic transmission
        homo_start = time.time()

        # encryption
        encr_start = time.time()
        encrypted_Z = self.encrypt_data([partial_Z], self.n_threads)
        encr_time = time.time() - encr_start

        # split data
        split_start = time.time()
        self.split_data(encrypted_Z, self.n_servers)
        split_time = time.time() - split_start

        # send to server
        trans_start = time.time()
        self.multi_thread_trans(self.n_servers)
        trans_time = time.time() - trans_start
        homo_time = time.time() - homo_start

        Z = self.global_data[0]
        pos_prob = self.sigmoid(Z)
        pred = 1 if pos_prob > 0.5 else 0

        self.reset_data()

        return pred, pos_prob

    def predict(self, X):
        start_time = time.time()

        partial_Z = X @ self.W

        # homomorphic transmission
        homo_start = time.time()

        # encryption
        encr_start = time.time()
        encrypted_Z = self.encrypt_data(partial_Z, self.n_threads)
        encr_time = time.time() - encr_start

        # split data
        split_start = time.time()
        self.split_data(encrypted_Z, self.n_servers)
        split_time = time.time() - split_start

        # send to server
        trans_start = time.time()
        self.multi_thread_trans(self.n_servers)
        trans_time = time.time() - trans_start
        homo_time = time.time() - homo_start

        Z = np.asarray(self.global_data)
        pos_prob = self.sigmoid(Z)
        pred = (pos_prob > 0.5).astype(int)

        self.reset_data()

        return pred, pos_prob


if __name__ == '__main__':
    train_data = np.asarray([[1, 2], [3, 4]])
    train_target = np.asarray([0, 1])
    weight = np.asarray([0.1, 0.1])
    dot_product = train_data @ weight
    print(dot_product)
    h = 1. / (1. + np.exp(dot_product))
    print(h)
    grad = (train_data.T * (train_target - h)).T
    print(grad)
    print(np.mean(grad, axis=0))
    print(np.log(h))
    loss = - train_target * np.log(h) - (1 - train_target) * np.log(1 - h)
    print(np.sum(loss))
