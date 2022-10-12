import os
import pdb
import argparse

import numpy as np
import torch
import torch.backends.cudnn as cudnn


from config import *
from dataset import *
from models import *
from utils import *


def compute_pro(args, net, dataloader):
    torch.set_grad_enabled(False)
    net.eval()
    # Device Init

    print("\n=====  start to compute pro feature =====")

    for idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        feature, outputs = net(inputs)  # 155, 2, 240, 240

        progress_bar(idx, len(dataloader))
        pseudo = torch.zeros(155, 2, 240, 240).cuda()
        pseudo[:,1:,...][outputs[:,1:,...] >= 0.75] = 1
        pseudo[:,0:1,...] = 1 - pseudo[:,1:,...]
        initc_bck = torch.sum(feature * pseudo[:, 0:1, ...] * outputs[:,0:1,...], dim=[0,2,3], keepdim=True)
        initc_bck /= torch.sum(pseudo[:, 0:1, ...] * outputs[:,0:1,...], dim=[0,2,3], keepdim=True)

        initc_obj = torch.sum(feature * pseudo[:, 1:, ...] * outputs[:, 1:, ...], dim=[0, 2, 3], keepdim=True)
        if pseudo[:,1:,...].max() == 1:
            initc_obj /= torch.sum(pseudo[:, 1:, ...] * outputs[:, 1:, ...], dim=[0, 2, 3], keepdim=True)

        if idx == 0:
            pro_bck = initc_bck
            pro_obj = initc_obj
        else:
            pro_bck = torch.cat((pro_bck, initc_bck), dim=0)
            if pseudo[:, 1:, ...].max() == 1:
                pro_obj = torch.cat((pro_obj, initc_obj), dim=0)
    del inputs, targets, feature, outputs

    assert torch.isnan(pro_obj).any() == False
    pro_bck = torch.mean(pro_bck, dim=0, keepdim=True)
    pro_obj = torch.mean(pro_obj, dim=0, keepdim=True)
    ratios = compute_ratios(net, dataloader, pro_bck, pro_obj)
    np.save(os.path.join(args.output_root, "pro_bck.npy"), pro_bck.cpu().numpy())
    np.save(os.path.join(args.output_root, "pro_obj.npy"), pro_obj.cpu().numpy())
    np.save(os.path.join(args.output_root, "ratios.npy"), ratios)
    return ratios, pro_bck, pro_obj

def compute_ratios(net, dataloader, pro_bck, pro_obj):
    print("\n===== start to compute obj ratios based on new pseudo labels  =====")
    torch.cuda.empty_cache()
    torch.set_grad_enabled(False)
    net.eval()

    ratios = []
    for idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        feature, outputs = net(inputs)  # 155, 2, 240, 240
        dd_bck = torch.sum(torch.pow(feature - pro_bck, 2), dim=1, keepdim=True).detach() # 155, 1, 240, 240
        dd_obj = torch.sum(torch.pow(feature - pro_obj, 2), dim=1, keepdim=True).detach()
        pseudo = torch.zeros(155, 2, 240, 240).cuda().detach()
        pseudo[:, 1:, ...][dd_obj < dd_bck] = 1
        pseudo[:, 0:1, ...] = 1 - pseudo[:, 1:, ...]

        progress_bar(idx, len(dataloader))
        ratio = compute_ratio(pseudo)
        ratios.append(ratio.cpu())

    ratios = torch.stack(ratios, dim=0).view(155 * 500, 1).numpy()
    del inputs, targets, feature, outputs
    torch.set_grad_enabled(True)
    return ratios

def get_new_pseudo(args, feature, outputs, pro_bck, pro_obj):
    # feature: bs, 64, 240, 240
    # output: bs, 2, 240, 240
    # pro_bck & pro_obj: 1,64,1,1
    # This can generate a new pseudo label by the distance between feature and prototype

    bs = feature.size(0)
    pseudo = outputs.clone().detach()
    pseudo[:, 1:, ...][outputs[:, 1:, ...] >= args.pseudo_thresh] = 1
    pseudo[:, 1:, ...][outputs[:, 1:, ...] < args.pseudo_thresh] = 0
    pseudo[:, 0:1, ...] = 1 - pseudo[:, 1:, ...]
    initc_bck = torch.sum(feature * pseudo[:, 0:1, ...] * outputs[:, 0:1, ...], dim=[0, 2, 3], keepdim=True)
    initc_bck /= torch.sum(pseudo[:, 0:1, ...] * outputs[:, 0:1, ...], dim=[0, 2, 3], keepdim=True)
    pro_bck = pro_bck * args.moment + initc_bck * (1 - args.moment)

    if pseudo[:, 1:, ...].max() == 1:
        initc_obj = torch.sum(feature * pseudo[:, 1:, ...] * outputs[:, 1:, ...], dim=[0, 2, 3],
                              keepdim=True)  # 1, 64, 240, 240
        initc_obj /= torch.sum(pseudo[:, 1:, ...] * outputs[:, 1:, ...], dim=[0, 2, 3], keepdim=True)
        pro_obj = pro_obj * args.moment + initc_obj * (1 - args.moment)
    assert torch.isnan(pro_obj).any() == False
    dd_bck = torch.sum(torch.pow(feature - pro_bck, 2), dim=1, keepdim=True)  # 155, 1, 240, 240
    dd_obj = torch.sum(torch.pow(feature - pro_obj, 2), dim=1, keepdim=True)

    pseudo = torch.zeros(bs, 2, 240, 240).cuda()
    # one-hot mode
    pseudo[:, 1:, ...][(dd_obj < dd_bck)] = 1  #cross_entropy
    pseudo[:, 0:1, ...] = 1 - pseudo[:, 1:, ...]
    del feature, outputs
    return pseudo, pro_bck.detach(), pro_obj.detach()
