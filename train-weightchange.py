#!/usr/bin/env python3
import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple
from scipy.ndimage.measurements import maximum

import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torch.optim import optimizer
import torchvision.datasets
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt

import argparse
from pathlib import Path

import pickle as pkl

from dataset import Salicon

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description="Saliency Map reduction",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
default_dataset_dir = Path.home() / ".cache" / "torch" / "datasets"
parser.add_argument("--dataset-root", default=default_dataset_dir)
parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument(
    "--batch-size",
    default=128,
    type=int,
    help="Number of images within each mini-batch",
)
parser.add_argument(
    "--epochs",
    default=1000,
    type=int,
    help="Number of epochs (passes through the entire dataset) to train for",
)
parser.add_argument(
    "--val-frequency",
    default=10,
    type=int,
    help="How frequently to test the model on the validation set in number of epochs",
)
parser.add_argument(
    "--log-frequency",
    default=50,
    type=int,
    help="How frequently to save logs to tensorboard in number of steps",
)
parser.add_argument(
    "-j",
    "--worker-count",
    default=cpu_count(),
    type=int,
    help="Number of worker processes used to load data.",
)


class ImageShape(NamedTuple):
    height: int
    width: int
    channels: int


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def main(args):
    transform = transforms.ToTensor()
    args.dataset_root.mkdir(parents=True, exist_ok=True)
    train_dataset = Salicon('/mnt/storage/scratch/wp13824/adl-2020/train.pkl')
    test_dataset = Salicon('/mnt/storage/scratch/wp13824/adl-2020/val.pkl')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.worker_count,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.worker_count,
        pin_memory=True,
    )

    model = CNN(height=96, width=96, channels=3, batch_size=args.batch_size)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=0.0005, nesterov=True)

    log_dir = get_summary_writer_log_dir(args)
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )
    trainer = Trainer(
        model, train_loader, test_loader, criterion, optimizer, summary_writer, DEVICE
    )

    trainer.train(
        args.epochs,
        args.val_frequency,
        args.log_frequency,
    )
    trainer.evaluate()
    # trainer.get_first_layer()
    summary_writer.close()
    

class CNN(nn.Module):
    def __init__(self, height: int, width: int, channels: int, batch_size: int):
        super().__init__()
        self.input_shape = ImageShape(height=height, width=width, channels=channels)
        self.batch_size = batch_size

        self.conv1 = nn.Conv2d(
            in_channels=self.input_shape.channels,
            out_channels=32,
            kernel_size=(5, 5),
            padding=(2, 2),
        )
        self.initialise_layer(self.conv1)

        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            padding=(1, 1)
        )
        self.initialise_layer(self.conv2)

        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            padding=(1, 1)
        )
        self.initialise_layer(self.conv3)

        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.fc1 = nn.Linear(15488, 4608)
        self.initialise_layer(self.fc1)

        self.fc2 = nn.Linear(2304, 2304)
        self.initialise_layer(self.fc2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.conv1(images)
        x = F.relu(x)

        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = self.pool3(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)

        chunks = x.reshape(x.shape[0], 2304, 2)
        x = torch.max(chunks, 2)[0]
        
        x = self.fc2(x)

        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.constant_(layer.bias, 0.1)
        if hasattr(layer, "weight"):
            nn.init.uniform_(layer.weight, -0.5, 0.5)
            #initialize with random?


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        summary_writer: SummaryWriter,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.step = 0

    def train(
        self,
        epochs: int,
        val_frequency: int,
        log_frequency: int,
        start_epoch: int = 0
    ):
        self.model.train()
        for epoch in range(start_epoch, epochs):
            self.model.train()
            data_load_start_time = time.time()
            for batch, labels in self.train_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                data_load_end_time = time.time()

                logits = self.model.forward(batch)
                loss = self.criterion(logits,  labels)
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step) % log_frequency) == 0:
                    self.log_metrics(epoch, loss, data_load_time, step_time)

                self.step += 1

                data_load_start_time = time.time()

            if ((epoch + 1) % val_frequency) == 0:
                    self.validate()
                    self.model.train()

            if ((epoch + 1) % 100) == 0:
                    for g in self.optimizer.param_groups:
                        if g['lr'] > 0.0001:
                            g['lr'] = g['lr'] * 0.55

            # self.summary_writer.add_scalar("epoch", epoch, self.step)
            

    def evaluate(self):
        dataset = pkl.load(open('/mnt/storage/scratch/wp13824/adl-2020/val.pkl', 'rb'))
        preds = []

        self.model.eval()

        with torch.no_grad():
            for batch, labels in self.val_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(batch)

                preds.extend(logits.tolist())
        
        pkl.dump(preds, open("preds-weightchange.pkl", "wb"))
        pkl.dump(dataset, open("gts-weightchange.pkl", "wb"))


    def validate(self):
        total_loss = 0
        self.model.eval()

        with torch.no_grad():
            for batch, labels in self.val_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(batch)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()

        average_loss = total_loss / len(self.val_loader)

        self.summary_writer.add_scalars(
                "loss",
                {"test": average_loss},
                self.step
        )

    def get_first_layer(self):
        filters, biases = self.model.conv1.weight, self.model.conv1.bias
        pkl.dump(filters, open('conv1_filters.filters', 'wb'))
        min_filter, max_filter = filters.min(), filters.max()
        filters = (filters - min_filter)/(max_filter - min_filter)

        fig = plt.figure(figsize=(2,2))
        fig.set_facecolor('black')
        for i in range(1,33):
            ax = fig.add_subplot(6, 6, i)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.patch.set_visible(False)
            img = filters[i-1].cpu().detach()
            ax.imshow(img.transpose(0,2))
            ax.axis('off')
        fig.savefig('learned-filters.png', bbox_inches='tight', pad_inches=0)


    def log_metrics(self, epoch, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)

        self.summary_writer.add_scalars(
                "loss",
                {"train": float(loss.item())},
                self.step
        )
        self.summary_writer.add_scalar(
                "time/data", data_load_time, self.step
        )
        self.summary_writer.add_scalar(
                "time/data", step_time, self.step
        )


def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    tb_log_dir_prefix = f'CNN Pan et al with weight change--'
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)


if __name__ == "__main__":
    main(parser.parse_args())