import sys
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

sys.path.append('../')
from models.architectures import CNN_Encoder, CNN_Decoder
from datasets import ProcessedForestDataLoader
from loss_distribution.loss_distribution_analyse import LossDistributionAnalysis
from early_stop.early_stopping import EarlyStopping

#####################################################################################################################################################

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

#####################################################################################################################################################

class AE(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self._init_dataset()
        self.train_loader = self.data.train_loader
        self.validation_loader = self.data.validation_loader
        self.test_loader = self.data.test_loader
        
        self.model = AE_Network(args)
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.step_size, gamma=args.gamma)
        
        self.early_stopping = EarlyStopping(patience=args.patience, delta=args.delta, path=args.results_path + '/best_model.pth')
        
        self.writer = SummaryWriter(log_dir=args.results_path + '/logs')
        
        # 初始化 LossDistributionAnalysis 实例
        self.loss_analysis = LossDistributionAnalysis(model=self.model, train_loader=self.train_loader, validation_loader=self.validation_loader,
                                                      test_loader=self.test_loader, device=self.device, args=args)

#####################################################################################################################################################

    def _init_dataset(self):
        if self.args.dataset == 'FOREST':
            self.data = ProcessedForestDataLoader(self.args)
        else:
            print(f"Dataset not supported: {self.args.dataset}")
            sys.exit()

#####################################################################################################################################################

    def loss_function(self, recon_x, x):
        recon_x = recon_x.view(-1, 2 * 256 * 256)
        x = x.view(-1, 2 * 256 * 256)
        MSE = F.mse_loss(recon_x, x, reduction='sum')
        return MSE

#####################################################################################################################################################

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
        
        self.writer.flush()

#####################################################################################################################################################

    def test(self, epoch):
        self.model.eval()
        validation_loss = 0
        with torch.no_grad():
            for data in self.validation_loader:
                data = data.to(self.device)
                recon_batch = self.model(data)
                validation_loss += self.loss_function(recon_batch, data).item()

        avg_validation_loss = validation_loss / len(self.validation_loader.dataset)
        print(f'====> Validation set loss: {avg_validation_loss:.4f}')

        self.writer.add_scalar('Loss/validation', avg_validation_loss, epoch)

        self.writer.flush()

        self.early_stopping(avg_validation_loss, self.model)
        if self.early_stopping.early_stop:
            print("Early stopping")
            return True, avg_validation_loss
        return False, avg_validation_loss
