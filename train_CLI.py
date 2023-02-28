
# import ImageFolder
import os
from os.path import join
from tqdm import tqdm

import argparse
import torch
from torch import nn
from utils.plot_utils import show_imgrid, save_imgrid, saveallforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.models import resnet50
from lpips import LPIPS
from resnet_inverse import ResNetInverse, ResNetWrapper

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

parser = argparse.ArgumentParser()
parser.add_argument("--dataroot", type=str, default=r"E:\Datasets\imagenet-valid")
parser.add_argument("--save_root", type=str, default=r"D:\DL_Projects\Vision\Resnet_inverter")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=1E-3)
parser.add_argument("--beta_lpips", type=float, default=5.0)
parser.add_argument("--lpips_net", type=str, default="alex")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--runname", type=str, default="pilot1")
parser.add_argument("--save_every", type=int, default=1)
parser.add_argument("--save_img_every", type=int, default=20)
args = parser.parse_args()

lpips_net = args.lpips_net
beta_lpips = args.beta_lpips
lr = args.lr
batch_size = args.batch_size
max_epochs = args.epochs
save_every = args.save_every
save_img_every = args.save_img_every
saveroot = args.save_root


savedir = join(saveroot, args.save_dir)
figdir = join(savedir, args.fig_dir)
os.makedirs(figdir, exist_ok=True)
os.makedirs(figdir, exist_ok=True)
train_dataroot = args.dataroot # = "E:\Datasets\imagenet-valid"
savedir  # = "E:\Datasets\imagenet-valid"


from resnet_inverse import ResNetInverse
Dist = LPIPS(net=lpips_net).cuda().eval()
Dist.requires_grad_(False)
resnet_robust = resnet50(pretrained=True)
resnet_robust.load_state_dict(torch.load(
    join(torch.hub.get_dir(), "checkpoints", "imagenet_linf_8_pure.pt")))
resnet_repr = ResNetWrapper(resnet_robust).cuda().eval().requires_grad_(False)
dataset = ImageFolder(root=train_dataroot, transform=preprocess)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
invert_resnet = ResNetInverse([3, 4, 6, 3]).cuda().eval()
invert_resnet.requires_grad_(True)
optim = torch.optim.Adam(invert_resnet.parameters(), lr=lr)
try:
    for epoch in range(max_epochs):
        for i, (imgtsrs, _) in enumerate(tqdm(dataloader)):
            imgtsrs = imgtsrs.cuda()
            with torch.no_grad():
                act_vec, acttsr = resnet_repr(imgtsrs)

            img_recon = invert_resnet(acttsr)
            L2_loss = torch.mean((img_recon - imgtsrs)**2, dim=(1,2,3))
            lpipsLoss = Dist(denormalizer(imgtsrs),
                             denormalizer(img_recon).clamp(0, 1),
                             normalize=True).squeeze()
            loss = L2_loss + lpipsLoss * beta_lpips
            loss.sum().backward()
            optim.step()
            optim.zero_grad()
            print("L2 loss %.3f  LPIPS loss %.3f"%(L2_loss.mean().item(), lpipsLoss.mean().item()))
            if i % save_img_every == 0:
                savename = "epoch%d_batch%d"%(epoch, i)
                save_imgrid(denormalizer(imgtsrs.detach().cpu()).clamp(0, 1),
                            join(savedir, "imgs", f"{savename}_orig.jpg"), nrow=8)
                save_imgrid(denormalizer(img_recon.detach().cpu()).clamp(0, 1),
                            join(savedir, "imgs", f"{savename}_recon.jpg"), nrow=8)
        if epoch % save_every == 0:
            torch.save(invert_resnet.state_dict(), join(savedir, f"model_ep{epoch:03d}.pth"))
except KeyboardInterrupt:
    pass