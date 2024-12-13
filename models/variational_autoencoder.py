import sys
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

sys.path.append('../')
from models.architectures import Encoder, Decoder
from datasets.data_loader import ProcessedForestDataLoader
from anomaly_detection.anomaly_detection_pipeline import LossDistributionAnalysis
from early_stop.early_stopping import EarlyStopping

#####################################################################################################################################################

class VAE_Network(nn.Module):
    def __init__(self, args):
        super(VAE_Network, self).__init__()
        # Dimension of intermediate features
        output_size = 512  
        # Initialize the encoder model
        self.encoder = Encoder(output_size) 
        
        # Define layers to map encoder output to latent space parameters (mean and variance)
        self.fc_mu = nn.Linear(output_size, args.embedding_size)
        self.fc_var = nn.Linear(output_size, args.embedding_size)
        
        # Initialize the decoder model
        self.decoder = Decoder(args.embedding_size)
    
    def encode(self, x):
        # Obtain encoder outputs and skip-connection features
        features, encoder_features = self.encoder(x)
        # Compute mean
        mu = self.fc_mu(features)
        # Compute log-variance
        logvar = self.fc_var(features)
        return mu, logvar, encoder_features
    
    def reparameterize(self, mu, logvar):
        # Apply reparameterization trick during training
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
    
    def decode(self, z, encoder_features):
        # Decode latent variable `z` along with skip-connection features
        return self.decoder(z, encoder_features)
    
    def forward(self, x):
        # Forward pass: Encode input, reparameterize, and decode
        mu, logvar, encoder_features = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, encoder_features), mu, logvar

#####################################################################################################################################################

class VAE(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self._init_dataset()
        self.train_loader = self.data.train_loader
        self.validation_loader = self.data.validation_loader
        self.test_loader = self.data.test_loader
        
        self.model = VAE_Network(args)
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # Learning rate scheduler to adjust learning rate at regular intervals
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.step_size, gamma=args.gamma)
        # EarlyStopping to halt training when validation performance stops improving
        self.early_stopping = EarlyStopping(patience=args.patience, delta=args.delta, path=args.results_path + '/best_model.pth')
        # TensorBoard SummaryWriter for logging metrics and visualizations
        self.writer = SummaryWriter(log_dir=args.results_path + '/logs')
        
        self.loss_analysis = LossDistributionAnalysis(model=self.model, train_loader=self.train_loader,validation_loader=self.validation_loader,
                                                      test_loader=self.test_loader,device=self.device,args=args)

#####################################################################################################################################################

    def _init_dataset(self):
        # Initialize dataset loader
        if self.args.dataset == 'FOREST':
            self.data = ProcessedForestDataLoader(self.args)
        else:
            print(f"Dataset not supported : {self.args.dataset}")
            sys.exit()

#####################################################################################################################################################

    def loss_function(self, recon_x, x, mu, logvar, beta=1.0):
        """
        Custom loss function that combines β-VAE strategy.
        Balances reconstruction loss (MSE) and KL divergence (KLD).
        
        Args:
        - recon_x: Reconstructed images
        - x: Original input images
        - mu: Latent mean
        - logvar: Latent log-variance
        - beta: Weight for KLD
        - max_value: Maximum value range for clamping
        
        Returns:
        - total_loss: Combined loss
        - MSE: Reconstruction loss
        - KLD: KL divergence loss
        """
        
        # Reshape inputs for computing MSE
        recon_x = recon_x.view(-1, 2 * 256 * 256)
        x = x.view(-1, 2 * 256 * 256)
        
        # MSE loss for reconstruction
        MSE = F.mse_loss(recon_x, x, reduction='sum')
        
        # Clamp logvar to ensure numerical stability
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        # KL divergence loss
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD *= beta
        
        # Combine losses
        total_loss = MSE + KLD
        
        return total_loss, MSE, KLD

#####################################################################################################################################################

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        train_recon_loss = 0
        train_kld_loss = 0
        
        # Define KL annealing schedule: Linearly increase β from 0 to 1 over epochs
        #total_epochs = self.args.epochs
        # Linear increase
        #beta = min(1.0, epoch / total_epochs)  
        beta = 10000000.0
        
        for batch_idx, data in enumerate(self.train_loader):
            data = data.to(self.device)
            # Clear gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            recon_batch, mu, logvar = self.model(data)
            loss, recon_loss, kld_loss = self.loss_function(recon_batch, data, mu, logvar, beta=beta)
            
            # Backpropagation
            loss.backward()
            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_kld_loss += kld_loss.item()
            
            # Update weights
            self.optimizer.step()
            
            # Log training progress at intervals
            if batch_idx % self.args.log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(self.train_loader.dataset)} '
                    f'({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}\t'
                    f'Recon: {recon_loss.item() / len(data):.6f}\tKLD: {kld_loss.item() / len(data):.6f}')
        
        avg_loss = train_loss / len(self.train_loader.dataset)
        avg_mse_loss = train_recon_loss / len(self.train_loader.dataset)
        print(f'====> Epoch: {epoch} Average train loss: {avg_loss:.4f}')
        print(f'====> Epoch: {epoch} Average train mse loss: {avg_mse_loss:.4f}')
        
        # Log losses and learning rate to TensorBoard
        self.writer.add_scalar('Loss/train', avg_loss, epoch)
        self.writer.add_scalar('Learning Rate', self.optimizer.param_groups[0]['lr'], epoch)
        
        # Update learning rate using the scheduler
        self.scheduler.step()

#####################################################################################################################################################

    def test(self, epoch):
        self.model.eval()
        test_loss = 0
        test_recon_loss = 0
        test_kld_loss = 0
        
        with torch.no_grad():
            for data in self.validation_loader:
                data = data.to(self.device)
                recon_batch, mu, logvar = self.model(data)
                loss, recon_loss, kld_loss = self.loss_function(recon_batch, data, mu, logvar)
                test_loss += loss.item()
                test_recon_loss += recon_loss.item()
                test_kld_loss += kld_loss.item()
        
        # Compute average losses
        avg_loss = test_loss / len(self.validation_loader.dataset)
        avg_recon_loss = test_recon_loss / len(self.validation_loader.dataset)
        avg_kld_loss = test_kld_loss / len(self.validation_loader.dataset)
        
        print(f'====> Validation set loss: {avg_loss:.4f}\tRecon: {avg_recon_loss:.4f}\tKLD: {avg_kld_loss:.4f}')
        
        # Log validation metrics to TensorBoard
        self.writer.add_scalar('Loss/validation/total', avg_loss, epoch)
        self.writer.add_scalar('Loss/validation/recon', avg_recon_loss, epoch)
        self.writer.add_scalar('Loss/validation/kld', avg_kld_loss, epoch)
        
        # Use EarlyStopping to monitor validation loss
        self.early_stopping(avg_loss, self.model)
        
        # Stop training if early stopping condition is met
        if self.early_stopping.early_stop:
            print("Early stopping")
            return True, avg_recon_loss
        return False, avg_recon_loss