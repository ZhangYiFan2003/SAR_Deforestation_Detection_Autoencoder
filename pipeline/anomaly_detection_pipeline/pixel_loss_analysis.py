import torch
import random
import numpy as np

#####################################################################################################################################################

class PixelLossAnalysis:
    def _calculate_pixel_losses(self, loader, dataset_type, num_batches=20):
        """
        Calculates the Mean Squared Error (MSE) loss for each pixel in the dataset.
        
        Args:
        - loader: DataLoader for the dataset.
        - dataset_type: Type of dataset ('Train', 'Validation', or 'Test').
        - num_batches: Number of random batches to sample.
        
        Returns:
        - pixel_losses: Array of pixel-wise MSE losses.
        """
        pixel_losses = []
        self.model.eval()
        with torch.no_grad():
            print("Evaluating model on dataset...")
            sampled_batches = random.sample(list(loader), min(num_batches, len(loader)))
            for i, data in enumerate(sampled_batches):
                data = data.to(self.device)
                recon_batch = self.model(data)
                if isinstance(recon_batch, tuple):
                    recon_batch = recon_batch[0]  
                loss_fn = torch.nn.MSELoss(reduction='none')
                batch_loss = loss_fn(recon_batch, data)
                pixel_losses.extend(batch_loss.view(-1).cpu().numpy())
        pixel_losses = np.array(pixel_losses)
        self.writer.add_histogram(f'{dataset_type}_Pixelwise_MSE_Loss_Distribution', pixel_losses, global_step=0)
        return pixel_losses

#####################################################################################################################################################

    def train_and_validation_and_test_loss_distribution(self):
        self.model.eval()
        train_pixel_losses = self._calculate_pixel_losses(self.train_loader, 'Train')
        hyperparameters = {
            'batch_size': self.args.batch_size,
            'epochs': self.args.epochs,
            'embedding_size': self.args.embedding_size,
            'learning_rate': self.args.lr,
            'weight_decay': self.args.weight_decay,
            'model': self.args.model,
            'step_size': self.args.step_size,
            'gamma': self.args.gamma
        }
        train_mean = np.mean(train_pixel_losses)
        train_std = np.std(train_pixel_losses)
        print(f"Train Pixel-wise MSE - Mean: {train_mean:.6f}, Std: {train_std:.6f}")
        anomaly_threshold = np.quantile(train_pixel_losses, 0.99)