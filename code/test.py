import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from config import *
from dataset import *
from models import *
from utils import *
from dice_loss import DiceLoss

def test(args):

    # System Init.
    device = config.device
    cudnn.benchmark = True
    criterian = DiceLoss(molecule=2)

    # Data Load.
    # source_val_loader = data_loader(args, mode='val', domain='source', bs=155)
    # source_test_loader = data_loader(args, mode='test', domain='source', bs=155)
    # target_val_loader = data_loader(args, mode='val', domain='target', bs=155)
    target_test_loader = data_loader(args, mode='test', domain='target', bs=155)

    # Model Load.
    net, _, _, _ = load_model(args, class_num=config.class_num, mode='test')

    # Do Test.
    net.eval()
    torch.set_grad_enabled(False)
    for dataloader in (target_test_loader,): # source_val_loader, source_test_loader,
        loss = 0
        for idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            _, outputs = net(inputs)
            batch_loss = criterian(outputs, targets).detach()
            loss += float(batch_loss.item())
            progress_bar(idx, len(dataloader), 'Dice-Coef: %.5f' % (1 - loss / (idx + 1)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    # System settings.
    parser.add_argument("--data", type=str, default="complete",
                        help="Label data type.")
    parser.add_argument("--model", type=str, default='pspnet_res50', # Need to be fixed
                        help="Model Name")

    # Data paths.
    parser.add_argument("--source_root", type=str, default="/home/featurize/data/BraTS2D/t1ce",
                        help="The directory containing the training image dataset.")
    parser.add_argument("--target_root", type=str, default="/home/featurize/data/BraTS2D/t2",
                        help="The directory containing the training image dataset.")
    parser.add_argument("--label_root", type=str, default="/home/featurize/data/BraTS2D/segs",
                        help="The directory containing the training label datgaset")

    # Model path.
    parser.add_argument("--ckpt_root", type=str, default="/home/featurize/work/finalyearproject/Segmentation/log/bbuda/complete/2022_08_29_09_24/checkpoint_complete_1.tar",
                        help="The directory containing the trained model checkpoint")

    args = parser.parse_args()
    assert args.data in args.ckpt_root, "Check data option and ckpt_root option"
    test(args)
