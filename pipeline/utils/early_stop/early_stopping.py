import torch
import torch.utils.data
from pipeline.utils.checkpointing import save_checkpoint

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
    def __init__(self, patience=5, delta=0, path='checkpoint.pth', window_size=5,
                 strategy='legacy_moving_average', checkpoint_metadata=None):
        self.patience = patience  # Number of epochs to wait for improvement
        self.delta = delta  # Minimum improvement threshold
        self.path = path  # Path to save the best model
        self.best_score = None  # Best smoothed validation score observed
        self.early_stop = False  # Whether to stop training
        self.counter = 0  # Counter for non-improving epochs
        self.val_losses = []  # List to store recent validation losses
        self.window_size = window_size  # Window size for smoothing
        self.strategy = strategy
        self.checkpoint_metadata = checkpoint_metadata or {}
        self.best_validation = None

#####################################################################################################################################################

    def __call__(self, val_loss, model, *, epoch=None):
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
        selection_loss = val_loss if self.strategy == 'best_validation' else smoothed_loss
        
        # ``best_validation`` is the exact raw minimum used by Optuna. Legacy
        # moving-average selection intentionally retains the historical delta.
        score = -selection_loss
        improved = (
            self.best_score is None
            or (
                score > self.best_score
                if self.strategy == 'best_validation'
                else score >= self.best_score + self.delta
            )
        )
        if improved:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(val_loss, model, epoch=epoch)
        else:
            # No improvement: increase counter
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                # Stop training if patience is exceeded
                self.early_stop = True

#####################################################################################################################################################

    def save_checkpoint(self, val_loss, model, *, epoch=None):
        """
        Saves the current model if validation loss improves.
        """
        print(f'Validation loss decreased ({self.best_score:.6f} --> {val_loss:.6f}). Saving model ...')
        self.best_validation = val_loss
        metadata = dict(self.checkpoint_metadata)
        if epoch is not None:
            metadata["epoch"] = epoch
        save_checkpoint(
            self.path,
            model,
            best_validation=val_loss,
            early_stopping_state=self.state_dict(),
            **metadata,
        )

    def state_dict(self):
        return {
            'patience': self.patience,
            'delta': self.delta,
            'best_score': self.best_score,
            'best_validation': self.best_validation,
            'early_stop': self.early_stop,
            'counter': self.counter,
            'val_losses': list(self.val_losses),
            'window_size': self.window_size,
            'strategy': self.strategy,
        }

    def load_state_dict(self, state):
        self.best_score = state.get('best_score')
        self.best_validation = state.get('best_validation')
        self.early_stop = state.get('early_stop', False)
        self.counter = state.get('counter', 0)
        self.val_losses = list(state.get('val_losses', []))
        self.window_size = state.get('window_size', self.window_size)
        self.strategy = state.get('strategy', self.strategy)
