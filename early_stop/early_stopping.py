import sys
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

#####################################################################################################################################################

class EarlyStopping:
    def __init__(self, patience=5, delta=0, path='checkpoint.pth', window_size=5):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.val_losses = []
        self.window_size = window_size

#####################################################################################################################################################

    def __call__(self, val_loss, model):
        # 更新损失列表
        self.val_losses.append(val_loss)
        if len(self.val_losses) > self.window_size:
            self.val_losses.pop(0)  # 保持窗口大小固定

        # 计算平滑后的损失（例如简单移动平均）
        smoothed_loss = sum(self.val_losses) / len(self.val_losses)

        score = -smoothed_loss  # 使用平滑后的损失
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

#####################################################################################################################################################

    def save_checkpoint(self, val_loss, model):
        print(f'Validation loss decreased ({self.best_score:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)