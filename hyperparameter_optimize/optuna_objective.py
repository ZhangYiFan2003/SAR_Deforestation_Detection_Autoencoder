import torch

def objective(trial, args, architectures):
    """
    Objective function for Optuna hyperparameter optimization.
    
    Args:
    - trial: An Optuna trial object for hyperparameter suggestion.
    - args: A namespace object containing configuration and hyperparameters.
    - architectures: A dictionary mapping model names to their corresponding architectures.
    
    Returns:
    - val_loss: The validation loss obtained after training the model with suggested hyperparameters.
    """
    try:
        # Suggest hyperparameters using Optuna
        lr = trial.suggest_float('lr', 1e-4, 1e-3, log=True)  
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)  
        
        # Update args with suggested hyperparameters
        args.lr = lr
        args.weight_decay = weight_decay
        
        # Reinitialize the model with the updated hyperparameters
        autoenc = architectures[args.model]
        autoenc.__init__(args)  
        
        # Clear unused GPU memory to avoid out-of-memory errors
        torch.cuda.empty_cache()  
        # Set GPU memory usage limit to 90% of the total memory
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.9)  
        
        # Training loop
        for epoch in range(1, args.epochs + 1):
            autoenc.train(epoch)
            should_stop, val_loss = autoenc.test(epoch)
            
            # Check early stopping condition
            if should_stop:
                print("early stop triggered and stop training")
                break
        # Return validation loss for Optuna to optimize
        return val_loss  
    
    except RuntimeError as e:
        # Handle CUDA-specific errors and re-raise other exceptions
        if "CUDA error" in str(e):
            print(f"CUDA error: {e}")
        raise