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
        
        # 用于存储每个图像的MSE误差
        self.train_losses = []
        self.test_losses = []

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
    
    def calculate_threshold(self):
        # 计算训练和测试误差的均值和标准差
        all_losses = np.array(self.train_losses + self.test_losses)
        mean_loss = np.mean(all_losses)
        std_loss = np.std(all_losses)

        # 设定95%置信区间的阈值
        threshold = mean_loss + 1.96 * std_loss
        print(f"Calculated Threshold (95% CI): {threshold:.4f}")
        
        # 可视化误差分布
        self.writer.add_histogram('Loss_Distribution', all_losses, global_step=0)
        
        return threshold

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
        pixel_losses = []  # 用于存储每个像素的重构误差
        with torch.no_grad():
            for data in self.test_loader:
                data = data.to(self.device)
                recon_batch = self.model(data)
                
                # 计算逐像素的MSE误差
                pixel_loss = F.mse_loss(recon_batch, data, reduction='none')  # 不求和，以保留每个像素的误差
                pixel_loss = pixel_loss.view(-1, 2, 256, 256)  # 恢复为图像形状 (batch, channels, height, width)
                pixel_losses.append(pixel_loss.cpu().numpy())  # 将误差转到CPU并存储
                
                # 计算单个批次的总误差
                batch_loss = pixel_loss.sum().item()  # 总和的MSE
                test_loss += batch_loss
                
                # 每批次的平均误差
                avg_batch_loss = batch_loss / len(data)
                self.test_losses.append(avg_batch_loss)

        avg_test_loss = test_loss / len(self.test_loader.dataset)
        print(f'====> Test set loss: {avg_test_loss:.4f}')
        
        # 记录平均测试损失
        self.writer.add_scalar('Loss/test', avg_test_loss, epoch)
        
        # 将像素误差转化为单个数组以计算阈值
        pixel_losses = np.concatenate(pixel_losses, axis=0)  # 合并所有批次的像素误差
        pixel_mean = np.mean(pixel_losses, axis=0)  # 计算每个像素位置的均值
        pixel_std = np.std(pixel_losses, axis=0)    # 计算每个像素位置的标准差
        
        # 设置95%置信区间的阈值
        confidence_interval = 1.96  # 对应95%的置信水平
        threshold_map = pixel_mean + confidence_interval * pixel_std  # 每个像素位置的阈值

        # 可视化每个像素的误差分布
        self.writer.add_image('Pixelwise_Loss_Mean', torch.tensor(pixel_mean), epoch, dataformats='CHW')
        self.writer.add_image('Pixelwise_Loss_Threshold', torch.tensor(threshold_map), epoch, dataformats='CHW')
        
        # Early Stopping Check
        self.early_stopping(avg_test_loss, self.model)
        if self.early_stopping.early_stop:
            print("Early stopping")
            return True
        return False
