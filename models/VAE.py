import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append('../')
from architectures import CNN_Encoder, CNN_Decoder
from datasets import ProcessedForestDataLoader

class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()
        output_size = 512  # Intermediate feature size
        self.encoder = CNN_Encoder(output_size)
        
        # Mapping to latent space parameters
        self.fc_mu = nn.Linear(output_size, args.embedding_size)
        self.fc_var = nn.Linear(output_size, args.embedding_size)
        
        self.decoder = CNN_Decoder(args.embedding_size)

    def encode(self, x):
        x = self.encoder(x)  # x shape: [batch_size, 2, 256, 256]
        return self.fc_mu(x), self.fc_var(x)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class VAE(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self._init_dataset()
        self.train_loader = self.data.train_loader
        self.test_loader = self.data.test_loader

        self.model = Network(args)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        
        self.writer = SummaryWriter(log_dir=args.results_path + '/logs')

    def _init_dataset(self):
        if self.args.dataset == 'FOREST':
            self.data = ProcessedForestDataLoader(self.args)
        else:
            print(f"Dataset not supported : {self.args.dataset}")
            sys.exit()

    def loss_function(self, recon_x, x, mu, logvar):
        # Reshape tensors for reconstruction loss
        recon_x = recon_x.view(-1, 2 * 256 * 256)
        x = x.view(-1, 2 * 256 * 256)
        
        # Using MSE loss for reconstruction (similar to the AE implementation)
        MSE = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence loss
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss is the sum of reconstruction loss and KL divergence
        return MSE + KLD, MSE, KLD

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        train_mse = 0
        train_kld = 0
        
        for batch_idx, data in enumerate(self.train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            
            recon_batch, mu, logvar = self.model(data)
            loss, mse, kld = self.loss_function(recon_batch, data, mu, logvar)
            
            loss.backward()
            train_loss += loss.item()
            train_mse += mse.item()
            train_kld += kld.item()
            
            self.optimizer.step()
            
            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tMSE: {:.6f}\tKLD: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    loss.item() / len(data),
                    mse.item() / len(data),
                    kld.item() / len(data)))

        avg_loss = train_loss / len(self.train_loader.dataset)
        avg_mse = train_mse / len(self.train_loader.dataset)
        avg_kld = train_kld / len(self.train_loader.dataset)
        
        print('====> Epoch: {} Average loss: {:.4f}\tMSE: {:.4f}\tKLD: {:.4f}'.format(
            epoch, avg_loss, avg_mse, avg_kld))
        
        # Log metrics to tensorboard
        self.writer.add_scalar('Loss/train/total', avg_loss, epoch)
        self.writer.add_scalar('Loss/train/mse', avg_mse, epoch)
        self.writer.add_scalar('Loss/train/kld', avg_kld, epoch)

    def test(self, epoch):
        self.model.eval()
        test_loss = 0
        test_mse = 0
        test_kld = 0
        
        with torch.no_grad():
            for data in self.test_loader:
                data = data.to(self.device)
                recon_batch, mu, logvar = self.model(data)
                loss, mse, kld = self.loss_function(recon_batch, data, mu, logvar)
                test_loss += loss.item()
                test_mse += mse.item()
                test_kld += kld.item()

        avg_loss = test_loss / len(self.test_loader.dataset)
        avg_mse = test_mse / len(self.test_loader.dataset)
        avg_kld = test_kld / len(self.test_loader.dataset)
        
        print('====> Test set loss: {:.4f}\tMSE: {:.4f}\tKLD: {:.4f}'.format(
            avg_loss, avg_mse, avg_kld))
        
        # Log metrics to tensorboard
        self.writer.add_scalar('Loss/test/total', avg_loss, epoch)
        self.writer.add_scalar('Loss/test/mse', avg_mse, epoch)
        self.writer.add_scalar('Loss/test/kld', avg_kld, epoch)