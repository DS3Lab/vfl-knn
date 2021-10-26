import time
import math
from multiprocessing import Queue, Process

import numpy as np
from scipy.special import softmax
from phe import paillier

from transmission.encryption import listEncryptor
from transmission.tenseal import Client


class MLRTrainer(object):

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
        self.W = np.zeros((self.n_f, self.args.n_classes), dtype=np.float32)

        # for Adam
        self.n_step = 0
        self.adam_lr = np.zeros_like(self.W)
        self.adam_lr.fill(self.args.lr)
        self.m = np.zeros_like(self.W)
        self.v = np.zeros_like(self.W)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8

        self.max_msg_size = 900000000
        self.options = [('grpc.max_send_message_length', self.max_msg_size),
                        ('grpc.max_receive_message_length', self.max_msg_size)]
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

    def update_adam(self, grad):
        self.n_step += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.power(grad, 2)
        v_hat = self.v / (1 - math.pow(self.beta2, self.n_step))
        self.adam_lr = self.args.lr / (np.sqrt(v_hat) + self.epsilon)
        #print("step {}, adam learning rate: {}".format(self.n_step, self.adam_lr))

    def one_iteration(self, n_epoch, n_iter, X, Y):
        start_time = time.time()
        np.set_printoptions(precision=4)
        np.set_printoptions(threshold=2)

        partial_Z = - X @ self.W
        partial_Z_flat = partial_Z.flatten()

        # homomorphic transmission
        homo_start = time.time()
        # split data
        split_start = time.time()
        self.split_data(partial_Z_flat, self.n_servers)
        split_time = time.time() - split_start

        # send to server
        trans_start = time.time()
        self.multi_thread_trans()
        trans_time = time.time() - trans_start
        homo_time = time.time() - homo_start

        Z = np.asarray(self.response_data).reshape(partial_Z.shape)

        P = softmax(Z, axis=1)
        # print("P shape: {}, {}".format(P.shape, P))
        N = X.shape[0]
        grad = 1. / N * (X.T @ (Y - P)) + self.args.lam * self.W
        # print("grad shape: {}, {}".format(grad.shape, grad))

        # Vanilla SGD
        # cur_lr = self.args.lr * math.pow(0.9, n_epoch)
        # self.W = self.W - cur_lr * grad

        # Adam
        self.update_adam(grad)
        m_hat = self.m / (1 - math.pow(self.beta1, self.n_step))
        self.W = self.W - self.adam_lr * m_hat

        loss = 1 / N * (np.trace(X @ self.W @ Y.T) + np.sum(np.log(np.sum(np.exp(Z), axis=1))))

        self.clear_cache()

        # sum regularization
        partial_reg = np.linalg.norm(self.W, 2)

        # homomorphic transmission
        homo_start = time.time()
        # split data
        split_start = time.time()
        self.split_data([partial_reg], self.n_servers)
        split_time += time.time() - split_start

        # send to server
        trans_start = time.time()
        reg = self.multi_thread_trans()[0]
        trans_time += time.time() - trans_start
        homo_time += time.time() - homo_start

        loss += self.args.lam / 2. * reg

        self.clear_cache()

        iter_time = time.time() - start_time

        print("iteration[{}] finish, cost {:.2f} s, server comm cost {:.2f} s, "
              "loss = {:.6f}, weight = {}"
              .format(n_iter, iter_time, trans_time, loss, self.W))

        return loss

    def predict_one(self, X):
        start_time = time.time()

        partial_Z = - X @ self.W
        partial_Z_flat = partial_Z.flatten()

        # homomorphic transmission
        homo_start = time.time()
        # split data
        split_start = time.time()
        self.split_data([partial_Z_flat], self.n_servers)
        split_time = time.time() - split_start

        # send to server
        trans_start = time.time()
        self.multi_thread_trans()
        trans_time = time.time() - trans_start
        homo_time = time.time() - homo_start

        Z = np.asarray(self.response_data).reshape(partial_Z.shape)

        P = softmax(Z, axis=1)
        pred = np.argmax(P, axis=1)

        self.clear_cache()

        return pred, P.tolist()

    def predict(self, X):
        start_time = time.time()

        partial_Z = - X @ self.W
        partial_Z_flat = partial_Z.flatten()

        # homomorphic transmission
        homo_start = time.time()
        # split data
        split_start = time.time()
        self.split_data(partial_Z_flat, self.n_servers)
        split_time = time.time() - split_start

        # send to server
        trans_start = time.time()
        self.multi_thread_trans()
        trans_time = time.time() - trans_start
        homo_time = time.time() - homo_start

        Z = np.asarray(self.response_data).reshape(partial_Z.shape)
        P = softmax(Z, axis=1)
        pred = np.argmax(P, axis=1)

        self.clear_cache()

        return pred, P
