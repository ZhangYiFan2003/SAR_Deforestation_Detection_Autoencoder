import argparse, os, sys
import torch
from models.VAE import VAE
from models.AE import AE
from utils import get_interpolations
from datasets import ProcessedForestDataLoader  
from loss_distribution.loss_distribution_analyse import LossDistributionAnalysis  

#####################################################################################################################################################

parser = argparse.ArgumentParser(
    description='Main function to call training for different AutoEncoders')
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
                    help='embedding size for latent space') #16, 32, 64, ...
parser.add_argument('--results_path', type=str, default='results/', metavar='N',
                    help='Where to store images')
parser.add_argument('--model', type=str, default='AE', metavar='N',
                    help='Which architecture to use')
parser.add_argument('--dataset', type=str, default='FOREST', metavar='N',
                    help='Which dataset to use')
parser.add_argument('--lr', type=float, default=5e-4, 
                    help='Learning rate for the optimizer')
parser.add_argument('--weight_decay', type=float, default=1e-4, 
                    help='Weight decay for the optimizer')
parser.add_argument('--step_size', type=int, default=5, 
                    help='Step size for learning rate scheduler StepLR')
parser.add_argument('--gamma', type=float, default=0.5, 
                    help='Gamma for learning rate scheduler StepLR')
parser.add_argument('--patience', type=int, default=5, 
                    help='Patience for early stopping')
parser.add_argument('--delta', type=float, default=0.01, 
                    help='Minimum change to qualify as improvement for early stopping')
parser.add_argument('--train', action='store_true', default=False,
                    help='Choose whether to train the model')
parser.add_argument('--test', action='store_true', default=True,
                    help='Choose whether to test the model with the latest saved weights')

#####################################################################################################################################################

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)

vae = VAE(args)
ae = AE(args)
architectures = {'AE': ae,
                 'VAE': vae}

# Load DataLoader
data_loader = ProcessedForestDataLoader(args)

#####################################################################################################################################################

if __name__ == "__main__":

    try:
        os.stat(args.results_path)
    except:
        os.mkdir(args.results_path)

    try:
        autoenc = architectures[args.model]
    except KeyError:
        print('---------------------------------------------------------')
        print('Model architecture not supported. ', end='')
        print('---------------------------------------------------------')
        sys.exit()

    # 实例化 LossDistributionAnalysis
    loss_analysis = LossDistributionAnalysis(autoenc.model, data_loader.train_loader, 
                                             data_loader.validation_loader, data_loader.test_loader, autoenc.device, args)

#####################################################################################################################################################

    if args.train:
        try:
            if args.cuda:
                print("Using GPU for training")
            else:
                print("Using CPU for training")

            for epoch in range(1, args.epochs + 1):
                autoenc.train(epoch)
                should_stop = autoenc.test(epoch)  # 测试并检查EarlyStopping

                # 检查EarlyStopping条件
                if should_stop:
                    print("Early stopping triggered. Training terminated.")
                    break  # 提前结束训练

                # 保存模型权重
                save_path = os.path.join(args.results_path, f'{args.model}_epoch_{epoch}.pth')
                torch.save(autoenc.model.state_dict(), save_path)
                print(f'Model weights saved at {save_path}')

        except (KeyboardInterrupt, SystemExit):
            print("Manual Interruption")
            
#####################################################################################################################################################

    if args.test:
        # 加载 "best_model.pth" 模型权重
        weight_path = os.path.join(args.results_path, "best_model.pth")#AE_epoch_10, VAE_epoch_10, best_model
        if not os.path.exists(weight_path):
            print("No weight file named 'best_model.pth' found for testing.")
            sys.exit()

        # 加载模型时注意模型类型
        if args.model == 'VAE':
            autoenc = architectures['VAE']
        elif args.model == 'AE':
            autoenc = architectures['AE']
        else:
            print("Unsupported model type.")
            sys.exit()

        # 使用 `torch.load()` 并显式设置 `map_location`
        state_dict = torch.load(weight_path, weights_only=True, map_location=autoenc.device)
        autoenc.model.load_state_dict(state_dict)
        print(f'Loaded weights from {weight_path}')
        
        # 使用测试集进行评估
        if data_loader.test_loader is None:
            print("Test loader is not initialized. Please use --add-deforestation-test to add test dataset.")
            sys.exit()
        
        # 使用封装后的方法进行逐像素误差分布测试和绘图
        loss_analysis.train_and_validation_and_test_loss_distribution()