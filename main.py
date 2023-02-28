
# import ImageFolder
import os
from os.path import join
from tqdm import tqdm

import torch
from torch import nn
from utils.plot_utils import show_imgrid, save_imgrid, saveallforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.models import resnet50
from lpips import LPIPS

#%%
preprocess = transforms.Compose([
    # transforms.Resize(256),
    # transforms.RandomCrop(224),
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                ratio=(3.0 / 4.0, 4.0 / 3.0),),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
denormalizer = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                    std=[1/0.229, 1/0.224, 1/0.225])
#%%
dataset = ImageFolder(root="E:\Datasets\imagenet-valid", transform=preprocess)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
#%%
from resnet_inverse import ResNetInverse, ResNetWrapper

Dist = LPIPS(net="alex").cuda().eval()
resnet_robust = resnet50(pretrained=True)
resnet_robust.load_state_dict(torch.load(
    join(torch.hub.get_dir(), "checkpoints", "imagenet_linf_8_pure.pt")))
resnet_repr = ResNetWrapper(resnet_robust).cuda().eval().requires_grad_(False)
#%%
# for imgtsrs, _ in dataloader:
#     imgtsrs = imgtsrs.cuda()
#     with torch.no_grad():
#         act_vec, acttsr = resnet_repr(imgtsrs)
#     break
# #%
# show_imgrid(denormalizer(imgtsrs), nrow=8)
#%%
# an CNN architecture that inverts resnet50
invert_resnet = ResNetInverse([3, 4, 6, 3]).cuda().eval()
#%%
saveroot = r"D:\DL_Projects\Vision\Resnet_inverter"
savedir = join(saveroot, "pilot_alex_lpips2")
figdir = join(savedir, "imgs")
os.makedirs(figdir, exist_ok=True)
os.makedirs(figdir, exist_ok=True)
#%%
from torch.utils.tensorboard import SummaryWriter
beta_lpips = 1.0
beta_l2 = 1.0
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
invert_resnet.requires_grad_(True)
optim = torch.optim.Adam(invert_resnet.parameters(), lr=1e-3)
writer = SummaryWriter(savedir)
try:
    for epoch in range(10):
        for i, (imgtsrs, _) in enumerate(tqdm(dataloader)):
            imgtsrs = imgtsrs.cuda()
            imgtsrs_denorm = denormalizer(imgtsrs)
            with torch.no_grad():
                act_vec, acttsr = resnet_repr(imgtsrs)

            img_recon = invert_resnet(acttsr)
            img_recon_denorm = denormalizer(img_recon)
            L2_loss = torch.mean((img_recon - imgtsrs)**2, dim=(1, 2, 3))
            TanhL2_loss = torch.mean((torch.tanh(imgtsrs_denorm * 2 - 1) -
                                      torch.tanh(img_recon_denorm * 2 - 1)) ** 2, dim=(1, 2, 3))
            # lpipsLoss = Dist(imgtsrs_denorm,
            #                  img_recon_denorm.clamp(0, 1),
            #                  normalize=True).squeeze()
            lpipsLoss = Dist(imgtsrs,
                             img_recon,).squeeze()
            loss = beta_l2 * L2_loss + lpipsLoss * beta_lpips # TanhL2_loss +
            loss.sum().backward()
            optim.step()
            optim.zero_grad()
            print("L2 loss %.3f  TanhL2 loss %.3f  LPIPS loss %.3f" % \
                  (L2_loss.mean().item(), TanhL2_loss.mean().item(), lpipsLoss.mean().item()))
            writer.add_scalar("L2_loss", L2_loss.mean().item(), epoch * len(dataloader) + i)
            writer.add_scalar("LPIPS_loss", lpipsLoss.mean().item(), epoch * len(dataloader) + i)
            writer.add_scalar("TanhL2_loss", TanhL2_loss.mean().item(), epoch * len(dataloader) + i)
            if i % 20 == 0:
                savename = "epoch%d_batch%d"%(epoch, i)
                save_imgrid(imgtsrs_denorm.detach().cpu(),
                            join(savedir, "imgs", f"{savename}_orig.jpg"), nrow=8)
                save_imgrid(denormalizer(img_recon.detach().cpu()).clamp(0, 1),
                            join(savedir, "imgs", f"{savename}_recon.jpg"), nrow=8)

        torch.save(invert_resnet.state_dict(), join(savedir, f"model_ep{epoch:03d}.pth"))
except KeyboardInterrupt:
    pass