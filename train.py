import os, sys
import torch
from config.parse_args import parse_arguments
from pipeline.train.train_pipeline import train_model
from pipeline.test.test_pipeline import test_model
from pipeline.models.autoencoder import AE
from pipeline.models.variational_autoencoder import VAE
from pipeline.datasets.data_loader import ProcessedForestDataLoader  

#####################################################################################################################################################

def main():
    args = parse_arguments()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    
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
        print('Model architecture not supported.')
        sys.exit()
    
    data_loader = ProcessedForestDataLoader(args)
    
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)
    
    if args.train:
        train_model(args, autoenc, architectures)
    
    if args.test:
        test_model(args, autoenc, data_loader)

#####################################################################################################################################################

if __name__ == "__main__":
    
    main()