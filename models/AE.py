import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter


import sys
sys.path.append('../')
from architectures import CNN_Encoder, CNN_Decoder
from datasets import ProcessedForestDataLoader

class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()
        output_size = args.embedding_size
        self.encoder = CNN_Encoder(output_size)
        self.decoder = CNN_Decoder(args.embedding_size)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        recon_x = self.decode(z)
        return recon_x

class AE(object):
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
        if self.args.dataset == 'FOREST':  # 添加ForestDatasetLoader的选项
            self.data = ProcessedForestDataLoader(self.args)  # 使用你之前定义的ForestDataLoader类
        else:
            print(f"Dataset not supported : {self.args.dataset}")
            sys.exit()

    """
    def loss_function(self, recon_x, x):
        # 将重建的图像和原始图像都归一化到 [0, 1] 范围
        recon_x = (recon_x + 1) / 2
        x = (x + 1) / 2

        # 展平 recon_x 和 x，使它们的形状一致
        recon_x = recon_x.view(-1, 2 * 256 * 256)
        x = x.view(-1, 2 * 256 * 256)

        # 使用展平后的张量计算 binary cross entropy
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        return BCE
        """

    def loss_function(self, recon_x, x):
        # 使用 MSELoss 而不是 binary_cross_entropy
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
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    loss.item() / len(data)))

        average_loss = train_loss / len(self.train_loader.dataset)
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, average_loss))
        
        self.writer.add_scalar('Loss/train', average_loss, epoch)

    def test(self, epoch):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for data in self.test_loader:
                data = data.to(self.device)
                recon_batch = self.model(data)
                test_loss += self.loss_function(recon_batch, data).item()

        average_test_loss = test_loss / len(self.test_loader.dataset) 
        print('====> Test set loss: {:.4f}'.format(average_test_loss))
        
        self.writer.add_scalar('Loss/test', average_test_loss, epoch)
