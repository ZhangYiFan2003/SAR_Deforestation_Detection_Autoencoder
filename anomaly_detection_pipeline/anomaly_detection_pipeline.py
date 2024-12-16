from anomaly_detection_pipeline.base_pipeline import BasePipeline
from anomaly_detection_pipeline.pixel_loss_analysis import PixelLossAnalysis
from anomaly_detection_pipeline.visualization import Visualization
from anomaly_detection_pipeline.anomaly_detection import AnomalyDetection

#####################################################################################################################################################

class AnomalyDetectionPipeline(BasePipeline, PixelLossAnalysis, Visualization, AnomalyDetection):
    def __init__(self, model, train_loader, validation_loader, test_loader, device, args):
        super().__init__(model, train_loader, validation_loader, test_loader, device, args)
