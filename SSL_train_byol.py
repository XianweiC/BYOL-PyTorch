import os
import time
from ml_collections import config_dict
import wandb

import torch
from torch import nn
from torch import optim
from torch.utils import data
from torch.optim import lr_scheduler
from torchvision import datasets

from src.model import ResNet, MLP
from src.dataset import CIFAR10, STL10, TinyImageNet, get_transforms, TVTDataset
from src.criterion import MSE
# from src.criterion import NearestClassCenterClassifier
from src.utils import setup_system
from engine_byol import train_one_epoch
from config import get_args


if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('fork', force=True)

    args = get_args()
    for key, value in vars(args).items():
        print(f'{key}: {value}')

    device = torch.device('mps')

    setup_system(args.random_seed)

    wandb.login()
    run = wandb.init(project=args.model_name,
                     config={
                         'random_seed': args.random_seed,
                         'learning_rate': args.lr,
                         'epochs': args.max_epoch,
                         'batch_size': args.batch_size,
                         'weight_decay': args.weight_decay,
                         'momentum': args.momentum,
                     })

    if args.dataset == "CIFAR-10":
        base_dataset = datasets.CIFAR10(root=args.data_dir, train=True)
        H, W = 32, 32
    elif args.dataset == "CIFAR-100":
        base_dataset = datasets.CIFAR100(root=args.data_dir, train=True)
        H, W = 32, 32
    elif args.dataset == "STL-10":
        data_dir = os.path.join(args.data_root_dir, "STL10")
        base_dataset = datasets.STL10(root=args.data_dir, split='train+unlabeled')
        H, W = 96, 96
    elif args.dataset == "Tiny-Imagenet-200":
        data_dir = os.path.join(args.data_root_dir, "Tiny-Imagenet")
        base_dataset = TinyImageNet(root=args.data_dir, split='train')
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    train_set = TVTDataset(base_dataset, transform=get_transforms('train', H, W))
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    online_network = nn.Sequential(ResNet(), MLP(output_channels=128)).to(device)
    predictor = MLP(input_channels=128, hidden_size=512, output_channels=128).to(device)
    target_network = nn.Sequential(ResNet(), MLP(output_channels=128)).to(device)

    for param_q, param_k in zip(online_network.parameters(), target_network.parameters()):
        param_k.data.copy_(param_q.data)
        param_k.requires_grad = False  # not update by gradient

    trainable_params = list(online_network.parameters()) + list(predictor.parameters())
    optimizer = optim.SGD(trainable_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)

    criterion = config_dict.ConfigDict()
    criterion.loss = MSE()
    # criterion.ncc = NearestClassCenterClassifier(online_network, device)

    start = time.time()
    for epoch in range(args.max_epoch):
        train_loss = train_one_epoch(online_network, target_network, predictor, train_loader, optimizer, criterion, lr_scheduler, device, epoch)
        # acc, error_rate = criterion.ncc(test_loader)
        # if acc > best_acc:
        #     best_acc = acc
        #     best_epoch = epoch
        #     torch.save(online_network.state_dict(), f'{ckpt_path}/model-{epoch}-{acc:.4}.pth')
        torch.save(online_network.state_dict(), f'{args.save_dir}/model-{epoch}.pth')

        wandb.log({'train_loss': train_loss})

    end = time.time() - start
    print(f'time consume: {end * 1000 / 60. :.4} min')

    torch.save(online_network.state_dict(), f'{args.save_dir}/model-final.pth')

    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()
