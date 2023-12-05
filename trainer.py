import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

import torch

import torchvision.models as models

import time


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.Flatten(),
            nn.Linear(32 * 64 * 64, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )
        
    def forward(self, x):
        return self.net(x)


class Trainer:
    def init(self, device):
        self.log_interval = 10

        self.model = ConvNet()

        self.model.train(True)
        self.model = self.model.to(device)

        self.device = device

        self.criterion = nn.CrossEntropyLoss()

        self.st = {}

    def init_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def load_model(self, path):
        if path is not None:
            self.model.load_state_dict(torch.load(path))

    def save_model(self, model, path):
        torch.save(model.state_dict(), path)

    def state(self) -> dict:
        return self.st

    def train_one_epoch(
        self,
        epoch_no,
        train_loader,
    ):
        start_time = time.time()
        backward_time = 0
        forward_time = 0
        optimizer_step_time = 0

        self.st['epoch'] = epoch_no

        self.model.train()
        num_batches = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            num_batches += 1

            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            forward_start_time = time.time()
            output = self.model(data)
            forward_time += (time.time() - forward_start_time)

            loss = self.criterion(output, target)

            backward_start_time = time.time()
            loss.backward()
            backward_time += (time.time() - backward_start_time)

            optimizer_step_time_start = time.time()
            self.optimizer.step()
            optimizer_step_time += (time.time() - optimizer_step_time_start)

            if batch_idx % self.log_interval == 0:
                print(f'batch loss: {loss.item()}')

        total_time = time.time() - start_time

        self.st['num_batches'] = num_batches
        self.st['total_time'] = total_time
        self.st['backward_time'] = backward_time
        self.st['forward_time'] = forward_time
        self.st['optimizer_step_time'] = optimizer_step_time
                

    def test(self, test_loader):
        return None
