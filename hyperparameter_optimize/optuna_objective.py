import torch

def objective(trial, args, architectures):
    try:
        # 建议超参数
        lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)  # 修改为 suggest_float，添加 log=True
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)  # 修改为 suggest_float，添加 log=True
        step_size = trial.suggest_int('step_size', 1, 5)  
        gamma = trial.suggest_float('gamma', 0.5, 0.9)  # 修改为 suggest_float
        embedding_size = trial.suggest_categorical('embedding_size', [128, 256])

        # 用建议的超参数更新 args
        args.lr = lr
        args.weight_decay = weight_decay
        args.step_size = step_size
        args.gamma = gamma
        args.embedding_size = embedding_size

        # 使用新超参数重新初始化模型
        autoenc = architectures[args.model]
        autoenc.__init__(args)  # 重新初始化
        
        # 限制 GPU 占用量
        torch.cuda.empty_cache()  # 清空缓存
        if torch.cuda.is_available():
            # 设置 GPU 显存分配为按需增长
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.cuda.set_per_process_memory_fraction(0.8)  # 最大占用 80% 的显存
        
        # 训练循环
        for epoch in range(1, args.epochs + 1):
            autoenc.train(epoch)
            should_stop, val_loss = autoenc.test(epoch)

            # 检查早停条件
            if should_stop:
                print("early stop triggered and stop training")
                break
            
            # 每个 epoch 后释放显存
            torch.cuda.empty_cache()

        return val_loss  # 返回验证损失以供优化
    except RuntimeError as e:
        if "CUDA error" in str(e):
            print(f"CUDA error: {e}")
        raise