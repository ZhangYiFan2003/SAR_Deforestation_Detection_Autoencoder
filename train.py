import os, sys
import torch
from config.parse_args import parse_arguments
from train_pipeline.train_pipeline import train_model
from train_pipeline.test_pipeline import test_model
from models.autoencoder import AE
from models.variational_autoencoder import VAE
from datasets.data_loader import ProcessedForestDataLoader  
from anomaly_detection_pipeline.anomaly_detection_pipeline import AnomalyDetectionPipeline

#####################################################################################################################################################

def main():
    args = parse_arguments()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    try:
        os.stat(args.results_path)
    except:
        os.mkdir(args.results_path)
    
    vae = VAE(args)
    ae = AE(args)
    architectures = {'AE': ae, 'VAE': vae}
    
    try:
        autoenc = architectures[args.model]
    except KeyError:
        print('---------------------------------')
        print('Model architecture not supported.')
        print('---------------------------------')
        sys.exit()
    
    data_loader = ProcessedForestDataLoader(args)
    
    anomaly_detection = AnomalyDetectionPipeline(autoenc.model, data_loader.train_loader, data_loader.validation_loader, 
                                                 data_loader.test_loader, autoenc.device, args)
    
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)
    
    if args.train:
        train_model(args, autoenc, architectures)
    
    if args.test:
        test_model(args, autoenc, data_loader, anomaly_detection)

#####################################################################################################################################################

if __name__ == "__main__":
    
    main()