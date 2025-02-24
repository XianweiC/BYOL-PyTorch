from tqdm import tqdm
from collections import defaultdict

import torch

from src.utils import AverageMeter


def train_one_epoch(model, dataloader, optimizer, criterion, lr_scheduler, device, epoch):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total = AverageMeter()
    batch_num = len(dataloader)
    pbar = tqdm(dataloader)
    for batch_idx, data in enumerate(pbar):
        img, label = data

        img = img.to(device)
        label = label.to(device)

        bs = img.shape[0]

        pred = model(img)

        total_loss = criterion.ce_loss(pred, label)

        total_loss.backward()
        optimizer.step()
        model.zero_grad()

        total.update(total_loss.item(), bs)
        if batch_idx % 19 == 0:  # print every 20 mini-batches
            pbar.set_description(f'[epoch:{epoch}] train_loss: {total.avg:.3f}')
    lr_scheduler.step()
    return total.avg


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    pbar = tqdm(val_loader)
    for batch_idx, data in enumerate(pbar):
        img, label = data

        img = img.to(device)
        label = label.to(device)

        logits = model(img)
        loss = criterion.ce_loss(logits, label)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(logits, label, topk=(1, 5))
        losses.update(loss.item(), img.shape[0])
        top1.update(prec1, img.shape[0])
        top5.update(prec5, img.shape[0])

        if batch_idx % 19 == 0:  # print every 20 mini-batches
            pbar.set_description(f'val_loss: {losses.avg:.3f}, top1: {top1.avg:.3f}%')

    return top1.avg, losses.avg


@torch.no_grad()
def validate_each_classes(model, val_loader, criterion, num_classes, device):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # 初始化类别统计字典
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    model.eval()
    batch_num = len(val_loader)
    pbar = tqdm(val_loader)
    for batch_idx, data in enumerate(pbar):
        img, label = data

        img = img.to(device)
        label = label.to(device)

        logits = model(img)
        loss = criterion.ce_loss(logits, label)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(logits, label, topk=(1, 5))
        losses.update(loss.item(), img.shape[0])
        top1.update(prec1[0].item(), img.shape[0])
        top5.update(prec5[0].item(), img.shape[0])

        # 统计每个类别的正确预测数和总数
        pred = logits.argmax(dim=1)  # 获取每个样本的预测类别
        for cls in range(num_classes):
            class_correct[cls] += (pred[label == cls] == cls).sum().item()
            class_total[cls] += (label == cls).sum().item()

        if batch_idx % 19 == 0:  # print every 20 mini-batches
            pbar.set_description(f'[{batch_idx}/{batch_num}] val_loss: {losses.avg:.3f}, top1: {top1.avg:.3f}%')

    # 计算每个类别的准确率
    class_accuracies = {}
    for cls in range(num_classes):
        if class_total[cls] > 0:
            class_accuracies[cls] = class_correct[cls] / class_total[cls]
        else:
            class_accuracies[cls] = None  # 如果某类别没有样本，则标记为 None

    return top1.avg, class_accuracies

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.shape[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res
