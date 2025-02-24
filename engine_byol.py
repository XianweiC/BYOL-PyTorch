from tqdm import tqdm

import torch

from src.utils import AverageMeter


def train_one_epoch(online_net, target_net, predictor, dataloader, optimizer, criterion, lr_scheduler, device, epoch):
    online_net.train()
    target_net.train()

    total = AverageMeter()
    batch_num = len(dataloader)

    pbar = tqdm(dataloader)
    for batch_idx, data in enumerate(pbar):
        view1, view2 = data

        bs = view1.shape[0]

        view1 = view1.to(device)
        view2 = view2.to(device)

        pred_embed1 = predictor(online_net(view1))
        pred_embed2 = predictor(online_net(view2))

        with torch.no_grad():
            tgt_embed1 = target_net(view2)
            tgt_embed2 = target_net(view1)

        loss = criterion.loss(pred_embed1, tgt_embed1) + criterion.loss(pred_embed2, tgt_embed2)
        loss = loss.mean()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        ema_update(online_net, target_net)

        total.update(loss.item(), bs)
        if batch_idx % 10 == 0:
            pbar.set_description(f'[epoch: {epoch}, {batch_idx}/{batch_num}] train_loss: {total.avg:.3f}')

    lr_scheduler.step()
    return total.avg

@torch.no_grad()
def ema_update(online_net, target_net, tau=0.996):
    for param_q, param_k in zip(online_net.parameters(), target_net.parameters()):
        param_k.data = param_k.data * tau + param_q.data * (1. - tau)
