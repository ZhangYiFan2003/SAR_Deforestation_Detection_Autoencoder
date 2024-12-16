import sys
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

sys.path.append('../')
from models.architectures import Encoder, Decoder
from datasets.data_loader import ProcessedForestDataLoader
from utils.early_stop.early_stopping import EarlyStopping

#####################################################################################################################################################

class AE_Network(nn.Module):
    def __init__(self, args):
        super(AE_Network, self).__init__()
        output_size = 512  # Intermediate feature size
        self.encoder = Encoder(output_size)
        self.decoder = Decoder(output_size)
    
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
        # Learning rate scheduler to adjust learning rate at regular intervals
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.step_size, gamma=args.gamma)
        # EarlyStopping to halt training when validation performance stops improving
        self.early_stopping = EarlyStopping(patience=args.patience, delta=args.delta, path=args.results_path + '/best_model.pth')
        # TensorBoard SummaryWriter for logging metrics and visualizations
        self.writer = SummaryWriter(log_dir=args.results_path + '/logs')

#####################################################################################################################################################

    def _init_dataset(self):
        # Initialize dataset loader
        if self.args.dataset == 'FOREST':
            self.data = ProcessedForestDataLoader(self.args)
        else:
            print(f"Dataset not supported: {self.args.dataset}")
            sys.exit()

#####################################################################################################################################################

    def loss_function(self, recon_x, x):
        """
        Calculates the Mean Squared Error (MSE) loss between the reconstructed and original input.
        """
        recon_x = recon_x.view(-1, 2 * 256 * 256)
        x = x.view(-1, 2 * 256 * 256)
        MSE = F.mse_loss(recon_x, x, reduction='sum')
        return MSE

#####################################################################################################################################################

    def train(self, epoch):
        """
        Trains the autoencoder for one epoch.
        """
        self.model.train()
        train_loss = 0
        for batch_idx, data in enumerate(self.train_loader):
            # Move data to the selected device
            data = data.to(self.device)
            # Reset gradients
            self.optimizer.zero_grad()
            # Forward pass
            recon_batch = self.model(data)
            # Compute loss
            loss = self.loss_function(recon_batch, data)
            # Backward pass
            loss.backward()
            # Accumulate total loss
            train_loss += loss.item()
            # Update weights
            self.optimizer.step()
            
            # Log progress for every few batches
            if batch_idx % self.args.log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(self.train_loader.dataset)} '
                      f'({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
        
        # Calculate and log average training loss for the epoch
        avg_loss = train_loss / len(self.train_loader.dataset)
        print(f'====> Epoch: {epoch} Average train loss: {avg_loss:.4f}')
        
        # Log to TensorBoard
        self.writer.add_scalar('Loss/train', avg_loss, epoch)
        # Update learning rate using scheduler
        self.scheduler.step()
        
        self.writer.flush()

#####################################################################################################################################################

    def test(self, epoch):
        """
        Evaluates the autoencoder on the validation dataset.
        """
        # Set model to evaluation mode
        self.model.eval()
        validation_loss = 0
        
        # Disable gradient computation
        with torch.no_grad():
            for data in self.validation_loader:
                data = data.to(self.device)
                recon_batch = self.model(data)
                validation_loss += self.loss_function(recon_batch, data).item()
        
        # Calculate and log average validation loss
        avg_validation_loss = validation_loss / len(self.validation_loader.dataset)
        print(f'====> Validation set loss: {avg_validation_loss:.4f}')
        
        self.writer.add_scalar('Loss/validation', avg_validation_loss, epoch)
        
        self.writer.flush()
        
        # Check if early stopping condition is met
        self.early_stopping(avg_validation_loss, self.model)
        if self.early_stopping.early_stop:
            print("Early stopping")
            return True, avg_validation_loss
        return False, avg_validation_loss
