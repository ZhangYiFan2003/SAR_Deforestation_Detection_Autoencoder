import torch

def objective(trial, args, architectures):
    try:
        # Suggest hyperparameters using Optuna
        lr = trial.suggest_float('lr', 1e-4, 1e-3, log=True)  # 修改为 suggest_float，添加 log=True
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)  # 修改为 suggest_float，添加 log=True

        # Update args with suggested hyperparameters
        args.lr = lr
        args.weight_decay = weight_decay
        
        # Reinitialize the model with the updated hyperparameters
        autoenc = architectures[args.model]
        autoenc.__init__(args)  # 重新初始化
        
        # Limit GPU memory usage
        torch.cuda.empty_cache()  
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
        return val_loss  # Return validation loss for optimization
    
    except RuntimeError as e:
        if "CUDA error" in str(e):
            print(f"CUDA error: {e}")
        raise