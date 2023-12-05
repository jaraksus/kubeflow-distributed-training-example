import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
import torch.utils.data as data_utils

import json

import os
import time


DATASET_FILE = 'dataset.py'
TRAINER_FILE = 'trainer.py'

WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))
LOCAL_RANK = int(os.environ.get("RANK", 0))

TOTAL_EPOCH = 1


def load_dataset():
    from dataset import Dataset
    d = Dataset()
    return d

def load_trainer():
    from trainer import Trainer
    t = Trainer()
    return t


def should_distribute():
    return dist.is_available() and WORLD_SIZE > 1

def is_distributed():
    return dist.is_available() and dist.is_initialized()


def main():
    print(f'torch.num_threads: {torch.get_num_threads()}')
    print(f'torch.num_num_interop_threads: {torch.get_num_interop_threads()}')

    device = 'cpu'

    print(f'world size: {WORLD_SIZE}')
    print(f'local rank: {LOCAL_RANK}')

    d = load_dataset()
    t = load_trainer()

    d.init()
    t.init(device=device)

    assert should_distribute()

    backend = dist.Backend.GLOO
    print('Using distributed PyTorch with {} backend'.format(backend))
    dist.init_process_group(backend=backend)

    assert d.train_dataset() is not None

    assert is_distributed()

    train_set = d.train_dataset()
    train_sampler = DistributedSampler(train_set)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        sampler=train_sampler,
        batch_size=10,
    )

    print('loader initialised...')

    Distributor = nn.parallel.DistributedDataParallel
    t.model = Distributor(t.model)

    t.init_optimizer()

    time_elapsed = 0

    def one_step(epoch):
        start_time = time.time()
        t.train_one_epoch(epoch, train_loader)
        cur_time = time.time() - start_time

        t.save_model(t.model.module, '/model.weights')

        return cur_time

    for epoch in range(1, TOTAL_EPOCH + 1):
        time_elapsed += one_step(epoch)

    if LOCAL_RANK == 0:
        print(f'time elapsed: {time_elapsed}')

if __name__ == "__main__":
    main()
