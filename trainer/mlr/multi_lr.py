import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, roc_auc_score

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

onehot_encoder = OneHotEncoder(sparse=False)


def loss(X, Y, W):
    """
    Y: onehot encoded
    """
    Z = - X @ W
    N = X.shape[0]
    loss = 1/N * (np.trace(X @ W @ Y.T) + np.sum(np.log(np.sum(np.exp(Z), axis=1))))
    return loss


def gradient(X, Y, W, mu):
    """
    Y: onehot encoded
    """
    Z = - X @ W
    P = softmax(Z, axis=1)
    N = X.shape[0]
    gd = 1/N * (X.T @ (Y - P)) + 2 * mu * W
    return gd


def gradient_descent(X, Y, max_iter=1000, eta=0.1, mu=0.01):
    """
    Very basic gradient descent algorithm with fixed eta and mu
    """
    Y_onehot = onehot_encoder.fit_transform(Y.reshape(-1 ,1))
    W = np.zeros((X.shape[1], Y_onehot.shape[1]))
    step = 0
    step_lst = []
    loss_lst = []
    W_lst = []

    while step < max_iter:
        step += 1
        W -= eta * gradient(X, Y_onehot, W, mu)
        step_lst.append(step)
        W_lst.append(W)
        loss_lst.append(loss(X, Y_onehot, W))

    print("loss = {}".format(loss_lst))

    # plt.plot(step_lst, loss_lst)
    # plt.xlabel("step")
    # plt.ylabel("loss")
    # plt.show()

    df = pd.DataFrame({
        'step': step_lst,
        'loss': loss_lst
    })
    return df, W


class Multiclass:

    def __init__(self):
        return

    def fit(self, X, Y):
        self.loss_steps, self.W = gradient_descent(X, Y, 10)

    def loss_plot(self):
        return self.loss_steps.plot(
            x='step',
            y='loss',
            xlabel='step',
            ylabel='loss'
        )

    def predict(self, H):
        Z = - H @ self.W
        P = softmax(Z, axis=1)
        return np.argmax(P, axis=1)


if __name__ == '__main__':
    X = load_iris().data
    Y = load_iris().target
    model = Multiclass()
    model.fit(X, Y)
    model.loss_plot()
    pred = model.predict(X)
    print("accuracy = {}".format(accuracy_score(Y, pred)))
