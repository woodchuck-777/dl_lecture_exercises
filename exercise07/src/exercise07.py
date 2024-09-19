#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

seed = 1234
device = "cuda"
torch.manual_seed(seed)
np.random.seed(seed)
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

x_train = np.load(os.path.join(parent_dir, "data", "x_train.npy"))
x_test = np.load(os.path.join(parent_dir, "data", "x_test.npy"))


class dataset(torch.utils.data.Dataset):

    def __init__(self, x_test):
        self.x_test = x_test.reshape(-1, 784).astype("float32") / 255

    def __len__(self):
        return self.x_test.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.x_test[idx], dtype=torch.float)


trainval_data = dataset(x_train)
test_data = dataset(x_test)

batch_size = 128
val_size = 10000
train_size = len(trainval_data) - val_size
train_data, val_data = torch.utils.data.random_split(
    trainval_data, [train_size, val_size]
)
dataloader_train = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True
)
dataloader_valid = torch.utils.data.DataLoader(
    val_data, batch_size=batch_size, shuffle=True
)
dataloader_test = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size, shuffle=False
)


# torch.log(0)によるnanを防ぐ
def torch_log(x):
    return torch.log(torch.clamp(x, min=1e-10))


class VAE(nn.Module):

    def __init__(self, z_dim: int) -> None:
        super().__init__()
        self.dense_enc1 = nn.Conv2d(
            in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.dense_enc2 = nn.Conv2d(
            in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.dense_encmean = nn.Linear(196, z_dim)
        self.dense_encvar = nn.Linear(196, z_dim)
        self.dense_dec1 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.dense_dec2 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.lr = nn.Linear(128, 784)
        self.norm1 = nn.BatchNorm2d(128)
        self.norm2 = nn.BatchNorm2d(64)
        self.norm3 = nn.BatchNorm2d(128)
        self.norm4 = nn.BatchNorm2d(256)
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)
        self.norm_e = nn.LayerNorm(z_dim)
        self.norm_v = nn.LayerNorm(z_dim)
        self.relu1 = nn.LeakyReLU()
        self.relu2 = nn.LeakyReLU()
        self.relu3 = nn.LeakyReLU()
        self.relu4 = nn.LeakyReLU()
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def _encoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.unsqueeze(0).unsqueeze(0)
        x = self.dense_enc1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dense_enc2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        m = self.dense_encmean(x)
        mean = self.norm_e(m)
        v = self.dense_encvar(x)
        v = self.norm_v(v)
        std = self.softplus(v)
        return mean, std

    def _sample_z(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        epsilon = torch.randn(mean.shape).to(device)
        return mean + std * epsilon

    def _decoder(self, z: torch.Tensor) -> torch.Tensor:
        z = self.dense_dec1(z)
        z = self.norm3(z)
        z = self.relu3(z)
        z = self.pool3(z)
        z = self.dense_dec2(z)
        z = self.norm4(z)
        z = self.relu4(z)
        z = self.pool4(z)
        z = self.sigmoid(z)
        return z

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, std = self._encoder(x)
        z = self._sample_z(mean, std)
        x = self._decoder(z)
        return x, z

    def loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, std = self._encoder(x)
        KL = -0.5 * torch.mean(
            torch.sum(1 + torch_log(std**2) - mean**2 - std**2, dim=1)
        )
        z = self._sample_z(mean, std)
        y = self._decoder(z)
        y = y.view(-1, 128)
        y = self.lr(y)
        reconstruction = torch.mean(
            torch.sum(x * torch_log(y) + (1 - x) * torch_log(1 - y), dim=1)
        )
        return KL, -reconstruction


z_dim = 32
n_epochs = 50
model = VAE(z_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
temp_loss, min_loss = 0.0, 1000.0
for epoch in range(n_epochs):
    losses = []
    KL_losses = []
    reconstruction_losses = []
    model.train()
    for x in dataloader_train:
        if x.size(0) != 128:
            continue
        x = x.to(device)
        model.zero_grad()
        KL_loss, reconstruction_loss = model.loss(x)
        loss = 0.05 * KL_loss + reconstruction_loss
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().detach().numpy())
        KL_losses.append(KL_loss.cpu().detach().numpy())
        reconstruction_losses.append(reconstruction_loss.cpu().detach().numpy())
    losses_val = []
    model.eval()
    for x in dataloader_valid:
        x = x.to(device)
        KL_loss, reconstruction_loss = model.loss(x)
        loss = KL_loss + reconstruction_loss
        losses_val.append(loss.cpu().detach().numpy())
    temp_loss = np.average(losses_val)
    if temp_loss < 0:
        break
    print(
        "EPOCH:%d, Train Lower Bound:%lf, (%lf, %lf), Valid Lower Bound:%lf"
        % (
            epoch + 1,
            np.average(losses),
            np.average(KL_losses),
            np.average(reconstruction_losses),
            temp_loss,
        )
    )
    if temp_loss > 0 and min_loss > temp_loss:
        sample_x = []
        answer = []
        model.eval()
        for x in dataloader_test:
            x = x.to(device)
            y, _ = model(x)
            y = y.tolist()
            sample_x.extend(y)
        with open(
            os.path.join(parent_dir, "data", "/submission_pred.csv"), "w"
        ) as file:
            writer = csv.writer(file, lineterminator="\n")
            writer.writerows(sample_x)
        file.close()
        min_loss = temp_loss
