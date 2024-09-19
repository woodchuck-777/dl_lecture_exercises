#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

np.random.seed(34)
sys.modules["tensorflow"] = None
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_fashionmnist():
    x_train = np.load(os.path.join(parent_dir, "data", "x_train.npy"))
    y_train = np.load(os.path.join(parent_dir, "data", "y_train.npy"))
    x_test = np.load(os.path.join(parent_dir, "data", "x_test.npy"))
    x_train = x_train.reshape(-1, 784).astype("float32") / 255
    y_train = np.eye(10)[y_train.astype("int32")]
    x_test = x_test.reshape(-1, 784).astype("float32") / 255
    return x_train, y_train, x_test


def np_log(x):
    return np.log(np.clip(a=x, a_min=1e-10, a_max=1e10))


def softmax(x, axis=1):
    x -= x.max(axis, keepdims=True)
    x_exp = np.exp(x)
    return x_exp / x_exp.sum(axis, keepdims=True)


def train(x, t, eps=0.2):
    """
    :param x: np.ndarray, 入力データ, shape=(batch_size, 入力の次元数)
    :param t: np.ndarray, 教師ラベル, shape=(batch_size, 出力の次元数)
    :param eps: float, 学習率
    """
    global W, b
    batch_size = x.shape[0]
    y_hat = softmax(np.matmul(x, W) + b)  # shape: (batch_size, 出力の次元数)
    cost = (-t * np_log(y_hat) - (1 - t) * np_log(1 - y_hat)).mean()
    delta = y_hat - t  # shape: (batch_size, 出力の次元数)
    dW = np.matmul(x.T, delta) / batch_size  # shape: (入力の次元数, 出力の次元数)
    db = (
        np.matmul(np.ones(shape=(batch_size,)), delta) / batch_size
    )  # shape: (出力の次元数,)
    W -= eps * dW
    b -= eps * db
    return cost


def valid(x, t):
    y_hat = softmax(np.matmul(x, W) + b)
    cost = (-t * np_log(y_hat) - (1 - t) * np_log(1 - y_hat)).mean()
    return cost, y_hat


x_train, y_train, x_test = load_fashionmnist()
W = np.random.uniform(low=-0.08, high=0.08, size=(784, 10)).astype("float32")
b = np.zeros(shape=(10,)).astype("float32")
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1)
_accuracy_score, _accuracy_score_previous = 0.0, 0.0

for epoch in range(1000):
    x_train, y_train = shuffle(x_train, y_train)
    cost = train(x_train, y_train)
    cost, y_pred = valid(x_valid, y_valid)
    if epoch % 10 == 9 or epoch == 0:
        _accuracy_score = accuracy_score(y_valid.argmax(axis=1), y_pred.argmax(axis=1))
        print(
            "EPOCH: {}, Valid Cost: {:.3f}, Valid Accuracy: {:.3f}".format(
                epoch + 1, cost, _accuracy_score
            )
        )
    if _accuracy_score > _accuracy_score_previous:
        y_pred = np.argmax(softmax(np.matmul(x_test, W) + b), axis=1)
        submission = pd.Series(y_pred, name="label")
        submission.to_csv(
            os.path.join(parent_dir, "data", "submission_pred.csv"),
            header=True,
            index_label="id",
        )
        print("saved")
    _accuracy_score_previous = _accuracy_score
