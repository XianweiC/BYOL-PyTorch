import time
from ml_collections import config_dict
import wandb

import torch
from torch import nn
from torch import optim
from torch.utils import data
from torch.optim import lr_scheduler

from src.model import ResNet
from src.dataset import CIFAR10, CIFAR100, STL10, TinyImageNet, get_transforms
from src.utils import setup_system
from engine import train_one_epoch, validate
from config import get_args


if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('fork', force=True)

    args = get_args()
    for key, value in vars(args).items():
        print(f'{key}: {value}')

    # device = torch.device('cuda')
    device = torch.device('mps')

    setup_system(args.random_seed)

    wandb.login()
    run = wandb.init(project='SSL',
                     config={
                         'random_seed': args.random_seed,
                         'learning_rate': args.lr,
                         'epochs': args.max_epoch,
                         'batch_size': args.batch_size,
                         'weight_decay': args.weight_decay,
                         'momentum': args.momentum,
                     })

    train_transforms = get_transforms('train', args.H, args.W)
    test_transforms = get_transforms('val', args.H, args.W)
    if args.dataset == "CIFAR-10":
        train_set = CIFAR10(root=args.data_dir, train=True, transform=train_transforms)
        val_set = CIFAR10(root=args.data_dir, train=False, transform=test_transforms)
    elif args.dataset == "CIFAR-100":
        train_set = CIFAR100(root=args.data_dir, train=True, transform=train_transforms)
        val_set = CIFAR100(root=args.data_dir, train=False, transform=test_transforms)
    elif args.dataset == "STL-10":
        train_set = STL10(root=args.data_dir, train=True, transform=train_transforms)
        val_set = STL10(root=args.data_dir, train=False, transform=test_transforms)
    elif args.dataset == "Tiny-Imagenet-200":
        train_set = TinyImageNet(root=args.data_dir, split='train', transform=train_transforms)
        val_set = TinyImageNet(root=args.data_dir, split='val', transform=test_transforms)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    num_classes = args.num_classes
    model = ResNet(ssl=False, num_classes=num_classes)
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)

    criterion = config_dict.ConfigDict()
    criterion.ce_loss = nn.CrossEntropyLoss()

    start = time.time()

    best_acc = 0.
    best_epoch = 0
    for epoch in range(args.max_epoch):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, lr_scheduler, device, epoch)

        if (epoch + 1) % args.save_freq == 0:
            acc, val_loss = validate(model, test_loader, criterion, device)
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
            torch.save(model.state_dict(), f'{args.save_dir}/model-{epoch}.pth')
            wandb.log({'acc': acc, 'train_loss': train_loss, 'val_loss': val_loss})

    end = time.time() - start

    print(f'{best_acc:.4f}', best_epoch)
    print(f'time consume: {end * 1000 :.4} ms')

    torch.save(model.state_dict(), f'{args.save_dir}/model-final.pth')

    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()
