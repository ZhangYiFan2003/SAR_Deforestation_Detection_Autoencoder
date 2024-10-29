import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from pytorch_msssim import ssim, ms_ssim, SSIM  # 导入SSIM模块

import sys
import os
import numpy as np

sys.path.append('../')
from models.architectures import CNN_Encoder, CNN_Decoder
from datasets import ProcessedForestDataLoader

class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()
        output_size = 512  # Intermediate feature size
        self.encoder = CNN_Encoder(output_size)
        
        # 将编码器输出映射到潜在空间的均值和方差参数
        self.fc_mu = nn.Linear(output_size, args.embedding_size)
        self.fc_var = nn.Linear(output_size, args.embedding_size)
        
        # 初始化解码器
        self.decoder = CNN_Decoder(args.embedding_size)

    def encode(self, x):
        # 获取编码器输出和跳跃连接的特征
        features, encoder_features = self.encoder(x)
        mu = self.fc_mu(features)
        logvar = self.fc_var(features)
        return mu, logvar, encoder_features

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z, encoder_features):
        # 使用 z 和 encoder_features 解码
        return self.decoder(z, encoder_features)

    def forward(self, x):
        # 编码，生成 z 及 encoder_features
        mu, logvar, encoder_features = self.encode(x)
        z = self.reparameterize(mu, logvar)
        # 解码并返回重建的结果
        return self.decode(z, encoder_features), mu, logvar

class EarlyStopping:
    def __init__(self, patience=5, delta=0, path='checkpoint.pth'):
        self.patience = patience  # 提前停止的耐心（连续epoch验证损失未降低的最大次数）
        self.delta = delta  # 验证损失降低的最小变化量
        self.path = path  # 模型权重保存路径
        self.best_score = None
        self.early_stop = False
        self.counter = 0  # 未改善epoch计数

    def __call__(self, val_loss, model):
        score = -val_loss  # EarlyStopping基于验证损失监测
        
        # 初始化最佳分数
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        
        # 如果分数没有改善
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        
        # 如果分数改善，重置计数并保存Checkpoint
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''当验证损失下降时，保存模型。'''
        print(f'Validation loss decreased ({self.best_score:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)


class VAE(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self._init_dataset()
        self.train_loader = self.data.train_loader
        self.test_loader = self.data.test_loader

        self.model = Network(args)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        # 初始化学习率调度器，step_size=10表示每10个epoch学习率衰减一次，gamma=0.5表示学习率每次衰减到原来的50%
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.step_size, gamma=args.gamma)
        
        # EarlyStopping对象的实例化
        self.early_stopping = EarlyStopping(patience=args.patience, delta=args.delta, path=args.results_path + '/best_model.pth')
        
        # 设置重构损失的类型和beta参数
        self.reconstruction_loss_type = args.reconstruction_loss_type
        self.beta = args.beta  # 传入自定义的beta值
        
        self.writer = SummaryWriter(log_dir=args.results_path + '/logs')

    def _init_dataset(self):
        if self.args.dataset == 'FOREST':
            self.data = ProcessedForestDataLoader(self.args)
        else:
            print(f"Dataset not supported : {self.args.dataset}")
            sys.exit()
    """
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
    """
    
    def loss_function(self, recon_x, x, mu, logvar, sparsity_weight=0.001, max_value=10):
        """
        计算自编码器的损失，包括MSE重构损失、KL散度和稀疏重构损失。
        在计算过程中，检查并限制输入值的范围，以避免NaN或数值不稳定的情况。

        参数:
        - recon_x: 重构图像
        - x: 原始输入图像
        - mu: 潜在变量的均值
        - logvar: 潜在变量的log方差
        - sparsity_weight: 稀疏性损失的权重（默认为0.001，可根据需要调整）
        - max_value: 限制重构图像和原始图像值范围的最大值

        返回:
        - total_loss: 总损失
        - MSE: 重构损失
        - KLD: KL散度损失
        - sparse_loss: 稀疏性损失
        """

        # 1. 将重构图像和原始图像的值裁剪在合理范围内，避免极端值导致不稳定性
        recon_x = torch.clamp(recon_x, -max_value, max_value).view(-1, 2 * 256 * 256)
        x = torch.clamp(x, -max_value, max_value).view(-1, 2 * 256 * 256)

        # 2. MSE重构损失
        MSE = F.mse_loss(recon_x, x, reduction='sum')

        # 3. 稀疏性惩罚（L1正则化）
        sparse_penalty = torch.sum(torch.abs(recon_x))
        sparse_loss = sparsity_weight * sparse_penalty  # 适当调节权重

        # 4. 对logvar进行裁剪，避免过大或过小值导致数值不稳定
        logvar = torch.clamp(logvar, min=-10, max=10)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # 合并损失
        total_loss = MSE + KLD + sparse_loss

        return total_loss, MSE, KLD

    def train(self, epoch):
            self.model.train()
            train_loss = 0
            train_recon_loss = 0
            train_kld_loss = 0

            for batch_idx, data in enumerate(self.train_loader):
                data = data.to(self.device)
                self.optimizer.zero_grad()

                # 前向传播
                recon_batch, mu, logvar = self.model(data)
                loss, recon_loss, kld_loss = self.loss_function(recon_batch, data, mu, logvar)

                loss.backward()
                train_loss += loss.item()
                train_recon_loss += recon_loss.item()
                train_kld_loss += kld_loss.item()

                self.optimizer.step()

                if batch_idx % self.args.log_interval == 0:
                    print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(self.train_loader.dataset)} '
                        f'({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}\t'
                        f'Recon: {recon_loss.item() / len(data):.6f}\tKLD: {kld_loss.item() / len(data):.6f}')

            avg_loss = train_loss / len(self.train_loader.dataset)
            print(f'====> Epoch: {epoch} Average train loss: {avg_loss:.4f}')
            
            # 记录损失和学习率
            self.writer.add_scalar('Loss/train', avg_loss, epoch)
            self.writer.add_scalar('Learning Rate', self.optimizer.param_groups[0]['lr'], epoch)

            # 更新学习率
            self.scheduler.step()

    def test(self, epoch):
        self.model.eval()
        test_loss = 0
        test_recon_loss = 0
        test_kld_loss = 0
        
        with torch.no_grad():
            for data in self.test_loader:
                data = data.to(self.device)
                recon_batch, mu, logvar = self.model(data)
                loss, recon_loss, kld_loss = self.loss_function(recon_batch, data, mu, logvar)
                test_loss += loss.item()
                test_recon_loss += recon_loss.item()
                test_kld_loss += kld_loss.item()

        avg_loss = test_loss / len(self.test_loader.dataset)
        avg_recon_loss = test_recon_loss / len(self.test_loader.dataset)
        avg_kld_loss = test_kld_loss / len(self.test_loader.dataset)
        
        print(f'====> Test set loss: {avg_loss:.4f}\tRecon: {avg_recon_loss:.4f}\tKLD: {avg_kld_loss:.4f}')
        
        # Log metrics to tensorboard
        self.writer.add_scalar('Loss/test/total', avg_loss, epoch)
        self.writer.add_scalar('Loss/test/recon', avg_recon_loss, epoch)
        self.writer.add_scalar('Loss/test/kld', avg_kld_loss, epoch)

        # 调用EarlyStopping监控
        self.early_stopping(avg_loss, self.model)
        
        # 若满足EarlyStopping条件，停止训练
        if self.early_stopping.early_stop:
            print("Early stopping")
            return True
        return False