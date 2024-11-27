import sys
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

#####################################################################################################################################################

class EarlyStopping:
    """
    Implements Early Stopping to halt training when validation loss stops improving.
    
    Args:
    - patience (int): Number of epochs to wait before stopping if no improvement.
    - delta (float): Minimum change in validation loss to be considered an improvement.
    - path (str): Path to save the model checkpoint with the best validation loss.
    - window_size (int): Size of the window for smoothing validation loss using a moving average.
    """
    def __init__(self, patience=5, delta=0, path='checkpoint.pth', window_size=5):
        self.patience = patience  # Number of epochs to wait for improvement
        self.delta = delta  # Minimum improvement threshold
        self.path = path  # Path to save the best model
        self.best_score = None  # Best smoothed validation score observed
        self.early_stop = False  # Whether to stop training
        self.counter = 0  # Counter for non-improving epochs
        self.val_losses = []  # List to store recent validation losses
        self.window_size = window_size  # Window size for smoothing

#####################################################################################################################################################

    def __call__(self, val_loss, model):
        """
        Checks if training should be stopped early based on validation loss.
        
        Args:
        - val_loss (float): Current epoch's validation loss.
        - model (torch.nn.Module): The model to save if validation loss improves.
        """
        # Update the list of validation losses
        self.val_losses.append(val_loss)
        # Maintain a fixed window size for smoothing
        if len(self.val_losses) > self.window_size:
            self.val_losses.pop(0)  
        
        # Calculate smoothed loss using a moving average
        smoothed_loss = sum(self.val_losses) / len(self.val_losses)
        
        # Calculate the score (negative of smoothed loss for minimization)
        score = -smoothed_loss  
        if self.best_score is None:
            # Initialize best score and save the model
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            # No improvement: increase counter
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                # Stop training if patience is exceeded
                self.early_stop = True
        else:
            # Improvement: reset counter, update best score, and save the model
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

#####################################################################################################################################################

    def save_checkpoint(self, val_loss, model):
        """
        Saves the current model if validation loss improves.
        """
        print(f'Validation loss decreased ({self.best_score:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)