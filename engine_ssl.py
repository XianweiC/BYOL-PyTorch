import os
from PIL import Image
from tqdm import tqdm

import torch

from src.utils import AverageMeter


def train_one_epoch(model, dataloader, optimizer, criterion, lr_scheduler, device, epoch):
    model.train()
    lr_scheduler.step()

    total = AverageMeter()
    batch_num = len(dataloader)

    pbar = tqdm(dataloader)
    for batch_idx, data in enumerate(pbar):
        view1, view2 = data

        bs = view1.shape[0]

        view1 = view1.to(device)
        view2 = view2.to(device)

        embed1 = model(view1)
        embed2 = model(view2)

        loss = criterion.infoNCE(embed1, embed2, model.tau)

        loss.backward()
        optimizer.step()
        model.zero_grad()

        total.update(loss.item(), bs)
        pbar.set_description(f'[epoch: {epoch}, {batch_idx}/{batch_num}] train_loss: {total.avg:.3f}')

    return total.avg

@torch.no_grad()
def val_visualize(model, val_loader, epoch):
    model.eval()

    if not os.path.exists(f'./results/epoch-{epoch}'):
        os.mkdir(f'./results/epoch-{epoch}')

    img, label = val_loader.dataset.__getitem__(0)
    img = img.unsqueeze(0)
    img = img.cuda(non_blocking=True)
    ori = img.squeeze(0).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    ori = Image.fromarray(ori)
    ori.save(f'./results/epoch-{epoch}/img1.png')

    masked_img, pred_img = model.visualization(img)
    masked_img = masked_img.squeeze(0).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    pred_img = pred_img.squeeze(0).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    masked_img = Image.fromarray(masked_img)
    masked_img.save(f'./results/epoch-{epoch}/masked_img1.png')
    pred_img = Image.fromarray(pred_img)
    pred_img.save(f'./results/epoch-{epoch}/pred_img1.png')

    img, label = val_loader.dataset.__getitem__(1)
    img = img.unsqueeze(0)
    img = img.cuda(non_blocking=True)
    ori = img.squeeze(0).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    ori = Image.fromarray(ori)
    ori.save(f'./results/epoch-{epoch}/img2.png')
    masked_img, pred_img = model.visualization(img)
    masked_img = masked_img.squeeze(0).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    pred_img = pred_img.squeeze(0).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    masked_img = Image.fromarray(masked_img)
    masked_img.save(f'./results/epoch-{epoch}/masked_img2.png')
    pred_img = Image.fromarray(pred_img)
    pred_img.save(f'./results/epoch-{epoch}/pred_img2.png')


    # plt.savefig(f'./results/res-{epoch}.png', bbox_inches='tight')

    return



# @torch.no_grad()
# def val_visualize(model, val_loader, epoch, n=2):
#     model.eval()
#     ranger = len(val_loader)
#     random_numbers = np.random.randint(low=0, high=ranger, size=n)
#
#     plt.figure(figsize=(50, 100))
#     for i, idx in enumerate(random_numbers):
#         img, label = val_loader.dataset.__getitem__(idx)
#         img = img.unsqueeze(0)
#         img = img.cuda(non_blocking=True)
#
#         masked_img, pred_img = model.visualization(img)
#
#         masked_img = masked_img.squeeze(0).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
#         pred_img = pred_img.squeeze(0).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
#
#         # plt.subplot(n, 2, i * 2 + 1)
#         # plt.imshow(masked_img)
#         # plt.subplot(n, 2, i * 2 + 2)
#         # plt.imshow(pred_img)
#
#         masked_img = Image.fromarray(masked_img)
#         masked_img.save(f'./results/epoch-{epoch}/masked_img.png')
#
#         pred_img = Image.fromarray(pred_img)
#         pred_img.save(f'./results/epoch-{epoch}/masked_img.png')
#
#
#     # plt.savefig(f'./results/res-{epoch}.png', bbox_inches='tight')
#
#     return
