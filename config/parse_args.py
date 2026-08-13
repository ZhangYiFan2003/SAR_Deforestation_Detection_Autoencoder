import argparse

#####################################################################################################################################################

def parse_arguments(argv=None):
    # Command-line arguments for training and testing options
    parser = argparse.ArgumentParser(
        description='Main function to call training for different AutoEncoders')
    parser.add_argument('--use-optuna', action='store_true', default=False,
                        help='Enable Optuna for hyperparameter optimization')
    parser.add_argument('--train', action='store_true', default=False,
                        help='Choose whether to train the model')
    parser.add_argument('--test', action='store_true', default=False,
                        help='Choose whether to test the model with the latest saved weights')
    
    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disable CUDA even when it is available')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--embedding-size', type=int, default=128, metavar='N',
                        help='VAE latent size; legacy AE always uses 512')
    parser.add_argument('--results-path', '--results_path', dest='results_path', type=str, default='runs', metavar='PATH',
                        help='Where to store images')
    parser.add_argument('--model', choices=('AE', 'VAE'), default='AE',
                        help='Which architecture to use')
    parser.add_argument('--dataset', choices=('FOREST',), default='FOREST',
                        help='Which dataset to use')
    parser.add_argument('--train-dir', default='data/train/processed',
                        help='Directory containing training GeoTIFF tiles')
    parser.add_argument('--validation-dir', default='data/validation/processed',
                        help='Directory containing validation GeoTIFF tiles')
    parser.add_argument('--test-dir', default='data/test/processed',
                        help='Directory containing test GeoTIFF tiles')
    parser.add_argument('--dataset-manifest', default=None,
                        help='Dataset manifest path recorded with the run')
    parser.add_argument('--dataset-manifest-version', default='unknown',
                        help='Dataset manifest version recorded with the run')
    parser.add_argument('--min-value', type=float, default=-15.0,
                        help='SAR normalization minimum (default: -15)')
    parser.add_argument('--max-value', type=float, default=-3.0,
                        help='SAR normalization maximum (default: -3)')
    parser.add_argument('--clamp-input', action='store_true',
                        help='Clamp normalized input to [0, 1] (legacy default: disabled)')
    parser.add_argument('--expected-channels', type=int, default=2,
                        help='Expected SAR channel count (default: 2)')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='DataLoader workers (default: CUDA=1, CPU=0)')
    pin_group = parser.add_mutually_exclusive_group()
    pin_group.add_argument('--pin-memory', dest='pin_memory', action='store_true',
                           help='Use pinned host memory')
    pin_group.add_argument('--no-pin-memory', dest='pin_memory', action='store_false',
                           help='Disable pinned host memory')
    parser.set_defaults(pin_memory=None)
    parser.add_argument('--prefetch-factor', type=int, default=2,
                        help='Batches prefetched per worker when workers > 0')
    parser.add_argument('--persistent-workers', action='store_true',
                        help='Keep DataLoader workers alive between epochs')
    parser.add_argument('--non-blocking', action='store_true',
                        help='Use non-blocking device copies; useful with pinned memory')
    parser.add_argument('--deterministic', action='store_true',
                        help='Request deterministic PyTorch algorithms (can reduce performance)')
    parser.add_argument('--run-id', default=None,
                        help='Optional run identifier; generated when omitted')
    parser.add_argument('--checkpoint', default=None,
                        help='Checkpoint to load for --test (legacy state_dict supported)')
    parser.add_argument('--selection-strategy',
                        choices=('legacy_moving_average', 'best_validation'),
                        default='legacy_moving_average',
                        help='Checkpoint selection contract; legacy behavior remains default')
    parser.add_argument('--profile', action='store_true',
                        help='Record lightweight data/H2D/step timing')
    parser.add_argument('--output-activation',
                        choices=('legacy_tanh', 'sigmoid', 'linear'),
                        default='legacy_tanh',
                        help='Experimental decoder output activation')
    parser.add_argument('--fpn-skips', choices=('none', 'p4', 'p4+p3'),
                        default='p4+p3',
                        help='Experimental decoder FPN skip ablation')
    parser.add_argument('--attention-variant', choices=('legacy', 'scaled'),
                        default='legacy',
                        help='Experimental attention scoring variant')
    parser.add_argument('--patience', type=int, default=5, 
                        help='Patience for early stopping')
    parser.add_argument('--delta', type=float, default=0.001, 
                        help='Minimum change to qualify as improvement for early stopping')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4, 
                        help='Learning rate for the optimizer')
    parser.add_argument('--weight-decay', '--weight_decay', dest='weight_decay', type=float, default=6e-06,
                        help='Weight decay for the optimizer')
    parser.add_argument('--step-size', '--step_size', dest='step_size', type=int, default=5,
                        help='Step size for learning rate scheduler StepLR')
    parser.add_argument('--gamma', type=float, default=0.7, 
                        help='Gamma for learning rate scheduler StepLR')
    
    args = parser.parse_args(argv)
    if not args.train and not args.test:
        parser.error('select at least one action: --train and/or --test')
    if args.max_value <= args.min_value:
        parser.error('--max-value must be greater than --min-value')
    if args.num_workers is not None and args.num_workers < 0:
        parser.error('--num-workers must be >= 0')
    if args.prefetch_factor <= 0:
        parser.error('--prefetch-factor must be > 0')
    if args.num_workers == 0 and args.persistent_workers:
        parser.error('--persistent-workers requires --num-workers > 0')
    if args.test and not args.train and not args.checkpoint:
        parser.error('--test without --train requires --checkpoint')
    if args.use_optuna and not args.train:
        parser.error('--use-optuna requires --train')
    if args.use_optuna and args.test:
        parser.error('run Optuna and checkpoint testing as separate commands')
    if args.non_blocking and args.pin_memory is False:
        parser.error('--non-blocking cannot be combined with --no-pin-memory')
    return args
