#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

x_train = np.load(os.path.join(parent_dir, "data", "x_train.npy"))
t_train = np.load(os.path.join(parent_dir, "data", "t_train.npy"))
x_test = np.load(os.path.join(parent_dir, "data", "x_test.npy"))


class train_dataset(torch.utils.data.Dataset):

    def __init__(self, x_train, t_train):
        data = x_train.astype("float32")
        self.x_train = []
        for i in range(data.shape[0]):
            self.x_train.append(Image.fromarray(np.uint8(data[i])))
        self.t_train = t_train
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        return self.transform(self.x_train[idx]), torch.tensor(
            t_train[idx], dtype=torch.long
        )


class test_dataset(torch.utils.data.Dataset):

    def __init__(self, x_test):
        data = x_test.astype("float32")
        self.x_test = []
        for i in range(data.shape[0]):
            self.x_test.append(Image.fromarray(np.uint8(data[i])))
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.x_test)

    def __getitem__(self, idx):
        return self.transform(self.x_test[idx])


def fix_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


fix_seed(seed=42)
trainval_data = train_dataset(x_train, t_train)
test_data = test_dataset(x_test)


class gcn:

    def __init__(self):
        pass

    def __call__(self, x):
        mean = torch.mean(x)
        std = torch.std(x)
        return (x - mean) / (std + 10 ** (-6))  # 0除算を防ぐ


class ZCAWhitening:

    def __init__(self, epsilon=1e-4, device="cuda"):  # 計算が重いのでGPUを用いる
        self.epsilon = epsilon
        self.device = device

    def fit(self, images):  # 変換行列と平均をデータから計算
        x = images[0][0].reshape(1, -1)
        self.mean = torch.zeros([1, x.size()[1]]).to(self.device)
        con_matrix = torch.zeros([x.size()[1], x.size()[1]]).to(self.device)
        for i in range(len(images)):  # 各データについての平均を取る
            x = images[i][0].reshape(1, -1).to(self.device)
            self.mean += x / len(images)
            con_matrix += torch.mm(x.t(), x) / len(images)
            if i % 1000 == 0:
                print("{0}/{1}".format(i, len(images)))
        self.E, self.V = torch.linalg.eigh(con_matrix)  # 固有値分解
        self.E = torch.max(
            self.E, torch.zeros_like(self.E)
        )  # 誤差の影響で負になるのを防ぐ
        self.ZCA_matrix = torch.mm(
            torch.mm(self.V, torch.diag((self.E.squeeze() + self.epsilon) ** (-0.5))),
            self.V.t(),
        )
        print("completed!")

    def __call__(self, x):
        size = x.size()
        x = x.reshape(1, -1).to(self.device)
        x -= self.mean
        x = torch.mm(x, self.ZCA_matrix.t())
        x = x.reshape(tuple(size))
        x = x.to("cpu")
        return x


zca = ZCAWhitening()
zca.fit(trainval_data)


# with open("arr.pkl", "wb") as f:
#     pickle.dump(zca, f) #保存
# with open("arr.pkl", "rb") as f:
#     zca = pickle.load(f)

batch_size = 64
val_size = 3000
train_data, val_data = torch.utils.data.random_split(
    trainval_data, [len(trainval_data) - val_size, val_size]
)
transform_train = transforms.Compose([transforms.ToTensor(), zca])
transform = transforms.Compose([transforms.ToTensor(), zca])
trainval_data.transform = transform_train
test_data.transform = transform
dataloader_train = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True
)
dataloader_valid = torch.utils.data.DataLoader(
    val_data, batch_size=batch_size, shuffle=True
)
dataloader_test = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size, shuffle=False
)


rng = np.random.RandomState(1234)
random_state = 42
device = torch.device("cuda" if torch.mps.is_available() else "cpu")

conv_net = nn.Sequential(
    nn.Conv2d(3, 32, 3),  # 32x32x3 -> 30x30x32
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.AvgPool2d(2),  # 30x30x32 -> 15x15x32
    nn.Conv2d(32, 64, 3),  # 15x15x32 -> 13x13x64
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.AvgPool2d(2),  # 13x13x64 -> 6x6x64
    nn.Conv2d(64, 128, 3),  # 6x6x64 -> 4x4x128
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.AvgPool2d(2),  # 4x4x128 -> 2x2x128
    nn.Flatten(),
    nn.Linear(2 * 2 * 128, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 10),
)


def init_weights(m):  # Heの初期化
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.0)


conv_net.apply(init_weights)

with open("model.pkl", "rb") as f:
    conv_net = pickle.load(f)

n_epochs = 30
lr = 0.001
# device = 'mps'
device = "mps"
conv_net.to(device)
optimizer = optim.Adam(conv_net.parameters(), lr=lr)
loss_function = nn.CrossEntropyLoss()

max_accuracy = 0.0
for epoch in range(n_epochs):
    losses_train, losses_valid = [], []
    n_train, acc_train = 0, 0
    conv_net.train()
    for x, t in dataloader_train:
        n_train += t.size()[0]
        conv_net.zero_grad()
        x = x.to(device)
        t = t.to(device)
        y = conv_net.forward(x)
        loss = loss_function(y, t)
        loss.backward()
        optimizer.step()
        pred = y.argmax(1)
        acc_train += (pred == t).float().sum().item()
        losses_train.append(loss.tolist())
    conv_net.eval()
    n_val, acc_val = 0, 0
    conv_net.eval()
    for x, t in dataloader_valid:
        n_val += t.size()[0]
        x = x.to(device)
        t = t.to(device)
        y = conv_net.forward(x)
        loss = loss_function(y, t)
        pred = y.argmax(1)
        acc_val += (pred == t).float().sum().item()
        losses_valid.append(loss.tolist())
    accuracy = acc_val / n_val
    print(
        "EPOCH: {}, Train [Loss: {:.3f}, Accuracy: {:.3f}], Valid [Loss: {:.3f}, Accuracy: {:.3f}]".format(
            epoch,
            np.mean(losses_train),
            acc_train / n_train,
            np.mean(losses_valid),
            accuracy,
        )
    )
    if accuracy > max_accuracy:
        max_accuracy = accuracy
        t_pred = []
        for x in dataloader_test:
            x = x.to(device)
            y = conv_net.forward(x)
            pred = y.argmax(1).tolist()
            t_pred.extend(pred)
        submission = pd.Series(t_pred, name="label")
        submission.to_csv(
            os.path.join(parent_dir, "data", "submission_pred.csv"),
            header=True,
            index_label="id",
        )
        with open("model.pkl", "wb") as f:
            pickle.dump(conv_net, f)

t_pred = []
for x in dataloader_test:
    x = x.to(device)
    y = conv_net.forward(x)
    pred = y.argmax(1).tolist()
    t_pred.extend(pred)
submission = pd.Series(t_pred, name="label")
submission.to_csv(
    os.path.join(parent_dir, "data", "submission_pred.csv"),
    header=True,
    index_label="id",
)
