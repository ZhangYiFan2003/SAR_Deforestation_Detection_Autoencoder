import argparse

#####################################################################################################################################################

def parse_arguments():
    # Command-line arguments for training and testing options
    parser = argparse.ArgumentParser(
        description='Main function to call training for different AutoEncoders')
    parser.add_argument('--use-optuna', action='store_true', default=False,
                        help='Enable Optuna for hyperparameter optimization')
    parser.add_argument('--train', action='store_true', default=False,
                        help='Choose whether to train the model')
    parser.add_argument('--test', action='store_true', default=True,
                        help='Choose whether to test the model with the latest saved weights')
    
    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
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
    parser.add_argument('--delta', type=float, default=0.001, 
                        help='Minimum change to qualify as improvement for early stopping')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4, 
                        help='Learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=6e-06,
                        help='Weight decay for the optimizer')
    parser.add_argument('--step_size', type=int, default=5, 
                        help='Step size for learning rate scheduler StepLR')
    parser.add_argument('--gamma', type=float, default=0.7, 
                        help='Gamma for learning rate scheduler StepLR')
    
    return parser.parse_args()