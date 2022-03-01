import time
import math
from multiprocessing import Queue

import numpy as np

from utils.comm_op import sum_all_reduce


class LassoTrainer(object):

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
        self.W = np.random.rand(self.end_f - self.start_f)

        # for Adam
        self.n_step = 0
        self.adam_lr = np.zeros_like(self.W)
        self.adam_lr.fill(self.args.lr)
        self.m = np.zeros_like(self.W)
        self.v = np.zeros_like(self.W)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8

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

    def one_iteration(self, n_epoch, n_iter, X, Y):
        start_time = time.time()
        np.set_printoptions(precision=4)
        np.set_printoptions(threshold=2)

        partial_Z = X @ self.W
        # print("partial dot = {}".format(partial_dot))
        comm_dot_start = time.time()
        Z = sum_all_reduce(partial_Z)
        comm_time = time.time() - comm_dot_start

        h = self.sigmoid(Z)
        grads = (X.T * (h - Y)).T
        grad = np.mean(grads, axis=0) + self.args.lam * np.sign(self.W)

        # Vanilla SGD
        cur_lr = self.args.lr * math.pow(0.9, n_epoch)
        self.W = self.W - self.args.lr * grad

        # Adam
        # self.update_adam(grad)
        # m_hat = self.m / (1 - math.pow(self.beta1, self.n_step))
        # self.W = self.W - self.adam_lr * m_hat

        loss = np.average(- Y * np.log(h) - (1 - Y) * np.log(1 - h))

        # sum regularization
        partial_reg = np.linalg.norm(self.W, 1)
        comm_reg_start = time.time()
        global_reg = sum_all_reduce(np.asarray(partial_reg))
        comm_time += time.time() - comm_reg_start

        loss += self.args.lam / 2. * global_reg
        iter_time = time.time() - start_time

        print("iteration[{}] finish, cost {:.2f} s, comm cost {:.2f} s, "
              "grad = {}, loss = {:.6f}, weight = {}"
              .format(n_iter, iter_time, comm_time, grad, loss, self.W))

        return loss

    def predict_one(self, X):
        start_time = time.time()

        partial_Z = X @ self.W
        comm_dot_start = time.time()
        Z = sum_all_reduce(np.asarray(partial_Z))
        comm_time = time.time() - comm_dot_start

        pos_prob = self.sigmoid(Z)
        pred = 1 if pos_prob > 0.5 else 0

        return pred, pos_prob

    def predict(self, X):
        start_time = time.time()

        partial_Z = X @ self.W
        comm_dot_start = time.time()
        Z = sum_all_reduce(partial_Z)
        comm_time = time.time() - comm_dot_start

        pos_prob = self.sigmoid(Z)
        pred = (pos_prob > 0.5).astype(int)

        return pred, pos_prob


if __name__ == '__main__':
    a = np.asarray([1, 2])
    b = np.asarray([3, 4])
    print(a * b)
