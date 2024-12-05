import argparse, os, sys
import torch
import optuna
from models.vae import VAE
from models.ae import AE
from datasets.datasets import ProcessedForestDataLoader  
from loss_distribution.loss_distribution_analyse import LossDistributionAnalysis
from hyperparameter_optimize.optuna_objective import objective

# Command-line arguments for training and testing options
parser = argparse.ArgumentParser(
    description='Main function to call training for different AutoEncoders')
parser.add_argument('--use-optuna', action='store_true', default=False,
                    help='Enable Optuna for hyperparameter optimization')
parser.add_argument('--train', action='store_true', default=False,
                    help='Choose whether to train the model')
parser.add_argument('--test', action='store_true', default=True,
                    help='Choose whether to test the model with the latest saved weights')

#####################################################################################################################################################

# Training hyperparameters
parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=11, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--embedding-size', type=int, default=128, metavar='N',
                    help='embedding size for latent space') 
parser.add_argument('--results_path', type=str, default='results/', metavar='N',
                    help='Where to store images')
parser.add_argument('--model', type=str, default='AE', metavar='N',
                    help='Which architecture to use')
parser.add_argument('--dataset', type=str, default='FOREST', metavar='N',
                    help='Which dataset to use')
parser.add_argument('--patience', type=int, default=5, 
                    help='Patience for early stopping')
parser.add_argument('--delta', type=float, default=0.01, 
                    help='Minimum change to qualify as improvement for early stopping')

#####################################################################################################################################################

# Optimizer hyperparameters
parser.add_argument('--lr', type=float, default=1e-4, #0.00011674956207899162
                    help='Learning rate for the optimizer')
parser.add_argument('--weight_decay', type=float, default=6e-06, #3.7146436941044483e-06
                    help='Weight decay for the optimizer')
parser.add_argument('--step_size', type=int, default=5, 
                    help='Step size for learning rate scheduler StepLR')
parser.add_argument('--gamma', type=float, default=0.7, 
                    help='Gamma for learning rate scheduler StepLR')

#####################################################################################################################################################

# Parse arguments and set up device
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)

# Instantiate models
vae = VAE(args)
ae = AE(args)
architectures = {'AE': ae,
                 'VAE': vae}

# Load DataLoader
data_loader = ProcessedForestDataLoader(args)

#####################################################################################################################################################

if __name__ == "__main__":

    # Create results directory if it doesn't exist
    try:
        os.stat(args.results_path)
    except:
        os.mkdir(args.results_path)
        
    # Select model architecture
    try:
        autoenc = architectures[args.model]
    except KeyError:
        print('---------------------------------------------------------')
        print('Model architecture not supported. ', end='')
        print('---------------------------------------------------------')
        sys.exit()
        
    # Instantiate LossDistributionAnalysis
    loss_analysis = LossDistributionAnalysis(autoenc.model, data_loader.train_loader, 
                                             data_loader.validation_loader, data_loader.test_loader, autoenc.device, args)

#####################################################################################################################################################

    if args.train:
        if args.use_optuna:
            # Perform hyperparameter optimization using Optuna
            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: objective(trial, args, architectures), n_trials=10)
            
            print("Best hyperparameters: ", study.best_params)
            print("Best validation loss: ", study.best_value)
            
            # Save the best hyperparameters
            with open(os.path.join(args.results_path, 'best_hyperparameters.txt'), 'w') as f:
                f.write(str(study.best_params))
        
#####################################################################################################################################################

        else:
            try:
                if args.cuda:
                    print("Using GPU for training")
                else:
                    print("Using CPU for training")
                
                for epoch in range(1, args.epochs + 1):
                    autoenc.train(epoch)
                    # Test and check EarlyStopping
                    should_stop, val_loss = autoenc.test(epoch)  
                    
                    # Check EarlyStopping conditions, Terminate training early
                    if should_stop:
                        print("Early stopping triggered. Training terminated.")
                        break  
                    
                    # Save model weights
                    save_path = os.path.join(args.results_path, f'{args.model}_epoch_{epoch}.pth')
                    torch.save(autoenc.model.state_dict(), save_path)
                    print(f'Model weights saved at {save_path}')
            
            except (KeyboardInterrupt, SystemExit):
                print("Manual Interruption")
            
#####################################################################################################################################################

    if args.test:
        # Load model weights from specified file
        weight_path = os.path.join(args.results_path, "AE_epoch_11.pth")#AE_epoch_10, VAE_epoch_10, best_model
        if not os.path.exists(weight_path):
            print("No weight file named 'best_model.pth' found for testing.")
            sys.exit()
        
        # Ensure the correct model type is used when loading the model
        if args.model == 'VAE':
            autoenc = architectures['VAE']
        elif args.model == 'AE':
            autoenc = architectures['AE']
        else:
            print("Unsupported model type.")
            sys.exit()
        
        # Load state dictionary using `torch.load()` and set `map_location` to device
        state_dict = torch.load(weight_path, weights_only=True, map_location=autoenc.device)
        autoenc.model.load_state_dict(state_dict)
        print(f'Loaded weights from {weight_path}')
        
        # Evaluate using the test dataset
        if data_loader.test_loader is None:
            print("Test loader is not initialized. Please use --add-deforestation-test to add test dataset.")
            sys.exit()
        
        # test and plot mse error per-pixel distribution
        loss_analysis.train_and_validation_and_test_loss_distribution()