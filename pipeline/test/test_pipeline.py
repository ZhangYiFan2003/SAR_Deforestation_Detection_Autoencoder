import os
import torch
from pipeline.utils.checkpointing import load_checkpoint

from pipeline.anomaly_detection.anomaly_detection_pipeline import AnomalyDetectionPipeline

#####################################################################################################################################################

def test_model(args, autoenc, data_loader):
    weight_path = args.checkpoint or args.best_checkpoint
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Checkpoint not found: {weight_path}")
    
    metadata = load_checkpoint(weight_path, autoenc.model, map_location=autoenc.device)
    checkpoint_preprocessing = metadata.get('preprocessing_config') or {}
    expected_preprocessing = data_loader.transform_config.to_dict()
    if checkpoint_preprocessing and checkpoint_preprocessing != expected_preprocessing:
        raise ValueError(
            'Checkpoint preprocessing does not match the configured inference transform: '
            f'{checkpoint_preprocessing} != {expected_preprocessing}'
        )
    print(f'Loaded weights from {weight_path}')
    
    if data_loader.test_loader is None:
        raise RuntimeError("Test loader is not initialized. Configure --test-dir.")
    
    anomaly_detection = AnomalyDetectionPipeline(
        autoenc.model,
        data_loader.train_loader,
        data_loader.validation_loader,
        data_loader.test_loader,
        autoenc.device,
        args,
    )
    anomaly_detection.reconstruct_and_analyze_images_by_time_sequence(
        target_date="20220721", image_dir=args.test_dir
    )
    anomaly_detection.reconstruct_and_analyze_images_by_clustering(
        target_date="20220721", image_dir=args.test_dir
    )
    report = anomaly_detection.generate_large_change_map(
        target_date="20210912", prev_date="20210409", image_dir=args.test_dir
    )
    print(f"Large-area inference status: {report.to_dict()}")
    return report
