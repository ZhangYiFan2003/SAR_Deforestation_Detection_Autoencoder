import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

import sys
import os
import numpy as np

sys.path.append('../')
from models.architectures import CNN_Encoder, CNN_Decoder
from datasets import ProcessedForestDataLoader

class AE_Network(nn.Module):
    def __init__(self, args):
        super(AE_Network, self).__init__()
        output_size = 512  # Intermediate feature size
        self.encoder = CNN_Encoder(output_size)
        self.decoder = CNN_Decoder(output_size)

    def encode(self, x):
        # Get encoder output and skip connection features
        features, encoder_features = self.encoder(x)
        return features, encoder_features

    def decode(self, z, encoder_features):
        # Use z and encoder_features for decoding
        return self.decoder(z, encoder_features)

    def forward(self, x):
        # Encode, generate latent representation and decode it
        features, encoder_features = self.encode(x)
        return self.decode(features, encoder_features)

class EarlyStopping:
    def __init__(self, patience=5, delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        print(f'Validation loss decreased ({self.best_score:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)

class AE(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self._init_dataset()
        self.train_loader = self.data.train_loader
        self.test_loader = self.data.test_loader

        self.model = AE_Network(args)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.step_size, gamma=args.gamma)
        self.early_stopping = EarlyStopping(patience=args.patience, delta=args.delta, path=args.results_path + '/best_model.pth')
        
        self.writer = SummaryWriter(log_dir=args.results_path + '/logs')

    def _init_dataset(self):
        if self.args.dataset == 'FOREST':
            self.data = ProcessedForestDataLoader(self.args)
        else:
            print(f"Dataset not supported: {self.args.dataset}")
            sys.exit()

    def loss_function(self, recon_x, x):
        recon_x = recon_x.view(-1, 2 * 256 * 256)
        x = x.view(-1, 2 * 256 * 256)
        MSE = F.mse_loss(recon_x, x, reduction='sum')
        return MSE

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        for batch_idx, data in enumerate(self.train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            recon_batch = self.model(data)
            loss = self.loss_function(recon_batch, data)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            if batch_idx % self.args.log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(self.train_loader.dataset)} '
                      f'({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')

        avg_loss = train_loss / len(self.train_loader.dataset)
        print(f'====> Epoch: {epoch} Average train loss: {avg_loss:.4f}')
        
        self.writer.add_scalar('Loss/train', avg_loss, epoch)
        self.scheduler.step()

    def test(self, epoch):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for data in self.test_loader:
                data = data.to(self.device)
                recon_batch = self.model(data)
                test_loss += self.loss_function(recon_batch, data).item()

        avg_test_loss = test_loss / len(self.test_loader.dataset)
        print(f'====> Test set loss: {avg_test_loss:.4f}')
        
        self.writer.add_scalar('Loss/test', avg_test_loss, epoch)
        
        self.early_stopping(avg_test_loss, self.model)
        if self.early_stopping.early_stop:
            print("Early stopping")
            return True
        return False
