import time
import math
from multiprocessing import Queue

import numpy as np
from scipy.special import softmax
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

from utils.comm_op import sum_all_reduce


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

    def update_adam(self, grad):
        self.n_step += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.power(grad, 2)
        v_hat = self.v / (1 - math.pow(self.beta2, self.n_step))
        self.adam_lr = self.args.lr / (np.sqrt(v_hat) + self.epsilon)
        print("step {}, adam learning rate: {}".format(self.n_step, self.adam_lr))

    def one_iteration(self, n_epoch, n_iter, X, Y):
        start_time = time.time()
        np.set_printoptions(precision=4)
        np.set_printoptions(threshold=2)

        partial_Z = - X @ self.W
        # print("partial dot = {}".format(partial_dot))
        comm_dot_start = time.time()
        Z = sum_all_reduce(partial_Z)
        #print("Z shape: {}, {}".format(Z.shape, Z))
        comm_time = time.time() - comm_dot_start

        P = softmax(Z, axis=1)
        #print("P shape: {}, {}".format(P.shape, P))
        N = X.shape[0]
        grad = 1. / N * (X.T @ (Y - P)) + self.args.lam * self.W
        #print("grad shape: {}, {}".format(grad.shape, grad))

        # Vanilla SGD
        #cur_lr = self.args.lr * math.pow(0.9, n_epoch)
        #self.W = self.W - cur_lr * grad

        # Adam
        self.update_adam(grad)
        m_hat = self.m / (1 - math.pow(self.beta1, self.n_step))
        self.W = self.W - self.adam_lr * m_hat
        # print("weight = {}, grad = {}".format(self.weight, grad))

        loss = 1 / N * (np.trace(X @ self.W @ Y.T) + np.sum(np.log(np.sum(np.exp(Z), axis=1))))

        # sum regularization
        partial_reg = np.linalg.norm(self.W, 2)
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

        partial_Z = - X @ self.W
        comm_dot_start = time.time()
        Z = sum_all_reduce(np.asarray([partial_Z]))
        comm_time = time.time() - comm_dot_start

        P = softmax(Z, axis=1)
        pred = np.argmax(P, axis=1)

        return pred, P.tolist()

    def predict(self, X):
        start_time = time.time()

        partial_Z = - X @ self.W
        comm_dot_start = time.time()
        Z = sum_all_reduce(partial_Z)
        comm_time = time.time() - comm_dot_start

        P = softmax(Z, axis=1)
        pred = np.argmax(P, axis=1)

        return pred, P


if __name__ == '__main__':
    X = np.asarray([[1, 2], [3, 4], [5, 6], [7, 8]])
    Y = np.asarray([0, 1, 2, 1])
    onehot_encoder = OneHotEncoder(sparse=False)
    Y_onehot = onehot_encoder.fit_transform(Y.reshape(-1, 1))
    print("Y: {}".format(Y_onehot))
    W = np.ones((X.shape[1], Y_onehot.shape[1]))
    print("W: {}".format(W))

    for i in range(10):
        Z = - X @ W
        print("Z: {}".format(Z))
        P = softmax(Z, axis=1)
        print("P: {}".format(P))
        N = X.shape[0]
        gd = 1 / N * (X.T @ (Y_onehot - P)) + 0.01 * W
        print("Gradient: {}".format(gd))

        W -= 0.01 * gd

        loss = 1 / N * (np.trace(X @ W @ Y_onehot.T) + np.sum(np.log(np.sum(np.exp(Z), axis=1))))
        print("loss: {}".format(loss))




