import time
import wandb
from ml_collections import config_dict

import torch
from torch import nn
from torch import optim
from torch.utils import data
from torch.optim import lr_scheduler
from torchvision import models

from src.dataset import CIFAR10, get_transforms
from src.utils import setup_system
from engine import train_one_epoch, validate
from config import get_args


if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('fork', force=True)

    args = get_args()

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

    train_set = CIFAR10(train=True, transform=get_transforms('train'))
    test_set = CIFAR10(train=False, transform=get_transforms('test'))
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    num_classes = args.num_classes

    # model = timm.create_model('resnet18', num_classes=num_classes).to(device)
    model = models.resnet18(weights=None, num_classes=num_classes).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-5)
    lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)

    criterion = config_dict.ConfigDict()
    criterion.ce_loss = nn.CrossEntropyLoss()

    start = time.time()

    best_acc = 0.
    best_epoch = 0
    for epoch in range(args.max_epoch):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, lr_scheduler, device, epoch)
        # acc, val_loss = validate(model, test_loader, criterion, device)

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
