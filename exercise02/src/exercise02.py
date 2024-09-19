#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

random_state = 123
rng = np.random.RandomState(1234)
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

x_train = np.load(os.path.join(parent_dir, "data", "x_train.npy"))
t_train = np.load(os.path.join(parent_dir, "data", "y_train.npy"))
x_test = np.load(os.path.join(parent_dir, "data", "x_test.npy"))

x_train, x_test = x_train / 255.0, x_test / 255.0
x_train, x_test = x_train.reshape(x_train.shape[0], -1), x_test.reshape(
    x_test.shape[0], -1
)
t_train = np.eye(N=10)[t_train.astype("int32").flatten()]
x_train, x_val, t_train, t_val = train_test_split(x_train, t_train, test_size=10000)


def np_log(x):
    return np.log(np.clip(x, 1e-10, 1e10))


def create_batch(data, batch_size):
    """
    :param data: np.ndarray, 入力データ
    :param batch_size: int, バッチサイズ
    """
    num_batches, mod = divmod(data.shape[0], batch_size)
    batched_data = np.split(data[: batch_size * num_batches], num_batches)
    if mod:
        batched_data.append(data[batch_size * num_batches :])
    return batched_data


def relu(x):
    return np.maximum(x, 0)


def deriv_relu(x):
    return (x > 0).astype(x.dtype)


def softmax(x):
    x -= x.max(axis=1, keepdims=True)  # オーバーフローを避ける
    x_exp = np.exp(x)
    return x_exp / np.sum(x_exp, axis=1, keepdims=True)


def deriv_softmax(x):
    return softmax(x) * (1 - softmax(x))


def crossentropy_loss(t, y):
    return (-t * np_log(y)).sum(axis=1).mean()


class Dense:

    def __init__(self, in_dim, out_dim, function, deriv_function):
        self.W = np.random.uniform(low=-0.08, high=0.08, size=(in_dim, out_dim)).astype(
            "float64"
        )
        self.b = np.zeros(out_dim).astype("float64")
        self.function = function
        self.deriv_function = deriv_function
        self.x = None
        self.u = None
        self.dW = None
        self.db = None
        self.params_idxs = np.cumsum([self.W.size, self.b.size])

    def __call__(self, x):
        """
        順伝播処理を行うメソッド．
        x: shape=(batch_size, in_dim_{j})
        h: shape=(batch_size, out_dim_{j})
        """
        self.x = x
        self.u = np.matmul(self.x, self.W) + self.b
        h = self.function(self.u)
        return h

    def b_prop(self, delta, W):
        """
        誤差逆伝播を行うメソッド．
        delta (=delta_{j+1}): shape=(batch_size, out_dim_{j+1})
        W (=W_{j+1}): shape=(out_dim_{j}, out_dim_{j+1})
        self.delta (=delta_{j}: shape=(batch_size, out_dim_{j})
        """
        self.delta = self.deriv_function(self.u) * np.matmul(delta, W.T)
        return self.delta

    def compute_grad(self):
        """
        勾配を計算するメソッド．
        self.x: shape=(batch_size, in_dim_{j})
        self.delta: shape=(batch_size, out_dim_{j})
        self.dW: shape=(in_dim_{j}, out_dim_{j})
        self.db: shape=(out_dim_{j})
        """
        batch_size = self.delta.shape[0]
        self.dW = np.matmul(self.x.T, self.delta) / batch_size
        self.db = np.matmul(np.ones(batch_size), self.delta) / batch_size

    def get_params(self):
        return np.concatenate([self.W.ravel(), self.b], axis=0)

    def set_params(self, params):
        """
        params: List[np.ndarray, np.ndarray]
            1つ目の要素が重みW(shape=(in_dim, out_dim), 2つ目の要素がバイアス(shape=(out_dim))
        """
        _W, _b = np.split(params, self.params_idxs)[:-1]
        self.W = _W.reshape(self.W.shape)
        self.b = _b

    def get_grads(self):
        return np.concatenate([self.dW.ravel(), self.db], axis=0)


class Model:
    def __init__(self, hidden_dims, activation_functions, deriv_functions):
        """
        :param hiden_dims: List[int]，各層のノード数を格納したリスト．
        :params activation_functions: List, 各層で用いる活性化関数を格納したリスト．
        :params derive_functions: List, 各層で用いる活性化関数の導関数を格納したリスト.
        """
        self.layers = []
        for i in range(len(hidden_dims) - 2):
            self.layers.append(
                Dense(
                    hidden_dims[i],
                    hidden_dims[i + 1],
                    activation_functions[i],
                    deriv_functions[i],
                )
            )
        self.layers.append(
            Dense(
                hidden_dims[-2],
                hidden_dims[-1],
                activation_functions[-1],
                deriv_functions[-1],
            )
        )

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, delta):
        for i, layer in enumerate(self.layers[::-1]):
            if i == 0:  # 出力層の場合
                layer.delta = delta  # y - t
                layer.compute_grad()
            else:  # 出力層以外の場合
                delta = layer.b_prop(delta, W)
                layer.compute_grad()
            W = layer.W

    def update(self, eps=0.01):
        for layer in self.layers:
            layer.W -= eps * layer.dW
            layer.b -= eps * layer.db


lr = 0.2
n_epochs = 25
batch_size = 256
mlp = Model(
    hidden_dims=[784, 512, 256, 128, 64, 10],
    activation_functions=[relu, relu, relu, relu, softmax],
    deriv_functions=[deriv_relu, deriv_relu, deriv_relu, deriv_relu, deriv_softmax],
)


def train_model(mlp, x_train, t_train, x_val, t_val, n_epochs=10):
    for epoch in range(n_epochs):
        losses_train = []
        losses_valid = []
        train_num = 0
        train_true_num = 0
        valid_num = 0
        valid_true_num = 0
        x_train, t_train = shuffle(x_train, t_train)
        x_train_batches, t_train_batches = create_batch(
            x_train, batch_size
        ), create_batch(t_train, batch_size)
        x_val, t_val = shuffle(x_val, t_val)
        x_val_batches, t_val_batches = create_batch(x_val, batch_size), create_batch(
            t_val, batch_size
        )
        for x, t in zip(x_train_batches, t_train_batches):
            y = mlp(x)
            loss = crossentropy_loss(t, y)
            losses_train.append(loss.tolist())
            delta = y - t
            mlp.backward(delta)
            if loss < 0.35:
                _lr = 0.1
            elif loss < 0.25:
                _lr = 5e-3
            else:
                _lr = lr
            mlp.update(_lr)
            acc = accuracy_score(t.argmax(axis=1), y.argmax(axis=1), normalize=False)
            train_num += x.shape[0]
            train_true_num += acc
        for x, t in zip(x_val_batches, t_val_batches):
            y = mlp(x)
            loss = crossentropy_loss(t, y)
            losses_valid.append(loss.tolist())
            acc = accuracy_score(t.argmax(axis=1), y.argmax(axis=1), normalize=False)
            valid_num += x.shape[0]
            valid_true_num += acc
        print(
            "EPOCH: {}, Train [Loss: {:.3f}, Accuracy: {:.3f}], Valid [Loss: {:.3f}, Accuracy: {:.3f}]".format(
                epoch,
                np.mean(losses_train),
                train_true_num / train_num,
                np.mean(losses_valid),
                valid_true_num / valid_num,
            )
        )


train_model(mlp, x_train, t_train, x_val, t_val, n_epochs)

t_pred = []
for x in x_test:
    x = x[np.newaxis, :]
    y = mlp(x)
    pred = y.argmax(1).tolist()
    t_pred.extend(pred)

submission = pd.Series(t_pred, name="label")
submission.to_csv(
    os.path.join(parent_dir, "data", "submission_pred.csv"),
    header=True,
    index_label="id",
)
