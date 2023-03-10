
# import ImageFolder
import os
from os.path import join
from tqdm import tqdm

import json
import datetime
import argparse
import torch
from torch import nn
from utils.plot_utils import show_imgrid, save_imgrid, saveallforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.models import resnet50
from torch.utils.tensorboard import SummaryWriter
from lpips import LPIPS
from resnet_inverse import ResNetInverse, ResNetWrapper, \
    Bottleneck_Up_Antichecker, Bottleneck_Up
from modulated_resnet import ModulatedResNetInverse


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
parser.add_argument("--dataroot", type=str, default="/home/biw905/Datasets/imagenet-valid")
#default=r"E:\Datasets\imagenet-valid")
parser.add_argument("--save_root", type=str, default="/n/scratch3/users/b/biw905/resnet_inverter")
#default=r"D:\DL_Projects\Vision\Resnet_inverter")
parser.add_argument("--ckpt_path", type=str, default=None)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=1E-4)
parser.add_argument("--gradclipnorm", type=float, default=10.0)
parser.add_argument("--leaky_relu_rgb", type=bool, default=True)
parser.add_argument("--beta_lpips", type=float, default=1.0)
parser.add_argument("--beta_l2", type=float, default=1.0)
parser.add_argument("--lpips_net", type=str, default="alex")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--runname", type=str, default="run1")
parser.add_argument("--save_every", type=int, default=1)
parser.add_argument("--save_img_every", type=int, default=20)
# taking a list of integer as input
parser.add_argument("--layer_nums", type=int, nargs='+', default=[2, 2, 3, 2])
# parser.add_argument("--blockclass", type=str, default="Bottleneck_Up_Antichecker")
args = parser.parse_args()

layer_nums = args.layer_nums  # [2, 2, 3, 2]
lpips_net = args.lpips_net
beta_lpips = args.beta_lpips
beta_l2 = args.beta_l2
lr = args.lr
batch_size = args.batch_size
max_epochs = args.epochs
save_every = args.save_every
save_img_every = args.save_img_every
leaky_relu_rgb = args.leaky_relu_rgb
gradclipnorm = args.gradclipnorm
saveroot = args.save_root
train_dataroot = args.dataroot  # = "E:\Datasets\imagenet-valid"
if args.ckpt_path is None:
    ckpt_path = None

curtime = datetime.datetime.now()
savedir = join(saveroot, f"{args.runname}_{curtime.strftime('%Y%m%d_%H%M%S')}")
figdir = join(savedir, "imgs")
os.makedirs(savedir, exist_ok=True)
os.makedirs(figdir, exist_ok=True)
json.dump(vars(args), open(join(savedir, f"train_args.json"), "w"), indent=4)

Dist = LPIPS(net=lpips_net).cuda().eval()
Dist.requires_grad_(False)
resnet_robust = resnet50(pretrained=True)
resnet_robust.load_state_dict(torch.load(
    join(torch.hub.get_dir(), "checkpoints", "imagenet_linf_8_pure.pt")))
resnet_repr = ResNetWrapper(resnet_robust).cuda().eval().requires_grad_(False)
dataset = ImageFolder(root=train_dataroot, transform=preprocess)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
invert_resnet = ModulatedResNetInverse(layer_nums, leaky_relu_rgb=True,).cuda().eval()
invert_resnet.requires_grad_(True)
if ckpt_path is not None:
    invert_resnet.load_state_dict(torch.load(ckpt_path))
optim = torch.optim.Adam(invert_resnet.parameters(), lr=lr)
writer = SummaryWriter(savedir)
for epoch in range(max_epochs):
    for i, (imgtsrs, _) in enumerate(tqdm(dataloader)):
        imgtsrs = imgtsrs.cuda()
        imgtsrs_denorm = denormalizer(imgtsrs)
        with torch.no_grad():
            act_vec, acttsr = resnet_repr(imgtsrs)

        img_recon = invert_resnet(acttsr)
        L2_loss = torch.mean((img_recon - (imgtsrs_denorm * 2 - 1)) ** 2, dim=(1, 2, 3))
        lpipsLoss = Dist(img_recon, (imgtsrs_denorm * 2 - 1), normalize=False).squeeze()
        loss = beta_l2 * L2_loss + lpipsLoss * beta_lpips  # TanhL2_loss +
        loss.sum().backward()
        grad_norm = nn.utils.clip_grad_norm_(invert_resnet.parameters(), 10.0)
        optim.step()
        optim.zero_grad()
        print("L2 loss %.3f   LPIPS loss %.3f  Gradnorm %.3f" % \
              (L2_loss.mean().item(), lpipsLoss.mean().item(), grad_norm.item()))
        writer.add_scalar("L2_loss", L2_loss.mean().item(), epoch * len(dataloader) + i)
        writer.add_scalar("LPIPS_loss", lpipsLoss.mean().item(), epoch * len(dataloader) + i)
        writer.add_scalar("grad_norm", grad_norm.item(), epoch * len(dataloader) + i)
        writer.add_scalar("epoch", epoch, epoch * len(dataloader) + i)
        if i % save_img_every == 0:
            savename = "epoch%d_batch%d" % (epoch, i)
            save_imgrid(imgtsrs_denorm.detach().cpu(),
                        join(savedir, "imgs", f"{savename}_orig.jpg"), nrow=8)
            save_imgrid(((img_recon.detach().cpu() + 1) / 2).clamp(0, 1),
                            join(savedir, "imgs", f"{savename}_recon.jpg"), nrow=8)

    torch.save(invert_resnet.state_dict(), join(savedir, f"model_ep{epoch:03d}.pth"))
