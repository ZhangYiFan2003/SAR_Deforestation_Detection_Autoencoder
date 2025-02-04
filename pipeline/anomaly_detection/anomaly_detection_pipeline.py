from torch.utils.tensorboard import SummaryWriter
from pipeline.anomaly_detection.plot_histogram import Plot
from pipeline.anomaly_detection.anomaly_detection import AnomalyDetection

#####################################################################################################################################################

class AnomalyDetectionPipeline(AnomalyDetection,Plot):
    def __init__(self, model, train_loader, validation_loader, test_loader, device, args):
        """
        Initializes the LossDistributionAnalysis class for analyzing the pixel-level loss distributions.
        
        Args:
        - model: The trained model (Autoencoder or Variational Autoencoder).
        - train_loader: DataLoader for the training set.
        - validation_loader: DataLoader for the validation set.
        - test_loader: DataLoader for the test set.
        - device: The device to run the analysis (CPU or GPU).
        - args: Additional arguments including hyperparameters and result paths.
        """
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.device = device
        self.args = args
        self.writer = SummaryWriter(log_dir=args.results_path + '/logs')
