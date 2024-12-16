import os
import torch
import optuna
from utils.hyperparameter_optimize.optuna_optimization import objective

#####################################################################################################################################################

def train_model(args, autoenc, architectures):
    if args.use_optuna:
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, args, architectures), n_trials=10)
        
        print("Best hyperparameters: ", study.best_params)
        print("Best validation loss: ", study.best_value)
        
        with open(os.path.join(args.results_path, 'best_hyperparameters.txt'), 'w') as f:
            f.write(str(study.best_params))
    else:
        try:
            device = "GPU" if args.cuda else "CPU"
            print(f"Using {device} for training")
            
            for epoch in range(1, args.epochs + 1):
                autoenc.train(epoch)
                should_stop, val_loss = autoenc.test(epoch)
                
                if should_stop:
                    print("Early stopping triggered. Training terminated.")
                    break
                
                save_path = os.path.join(args.results_path, f'{args.model}_epoch_{epoch}.pth')
                torch.save(autoenc.model.state_dict(), save_path)
                print(f'Model weights saved at {save_path}')
        except (KeyboardInterrupt, SystemExit):
            print("Manual Interruption")