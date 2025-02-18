import os, sys
import torch

from pipeline.anomaly_detection.anomaly_detection_pipeline import AnomalyDetectionPipeline

#####################################################################################################################################################

def test_model(args, autoenc, data_loader):
    
    anomaly_detection = AnomalyDetectionPipeline(autoenc.model, data_loader.train_loader, data_loader.validation_loader, 
                                                 data_loader.test_loader, autoenc.device, args)
    
    weight_path = os.path.join(args.results_path, "best_model.pth")
    if not os.path.exists(weight_path):
        print("No weight file named 'best_model.pth' found for testing.")
        sys.exit()
    
    state_dict = torch.load(weight_path, weights_only=True, map_location=autoenc.device)
    autoenc.model.load_state_dict(state_dict)
    print(f'Loaded weights from {weight_path}')
    
    if data_loader.test_loader is None:
        print("Test loader is not initialized. Please use --add-deforestation-test to add test dataset.")
        sys.exit()
    
    anomaly_detection.reconstruct_and_analyze_images_by_time_sequence(target_date="20220721")
    anomaly_detection.reconstruct_and_analyze_images_by_clustering(target_date="20220721")
    anomaly_detection.generate_large_change_map(target_date="20210912", prev_date="20210409")