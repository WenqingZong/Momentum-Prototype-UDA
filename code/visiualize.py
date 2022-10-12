import argparse

import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from config import *
from dataset import *
from models import *
from utils import *
import os

def test(args):

    # Device Init
    device = config.device
    cudnn.benchmark = True
    # Data Load
    validloader = data_loader(args, mode=args.mode, domain=args.domain)

    # Model Load
    net, _, _, _ = load_model(args, class_num=config.class_num, mode='test')
    net.eval()

    s_model, _, _, _ = load_model(args, class_num=config.class_num, mode='test', s_model=args.ckpt_root_s)
    s_model.eval()
    torch.set_grad_enabled(False)

    post_id = 0
    for idx, (inputs, targets) in tqdm(enumerate(validloader)):
        inputs = inputs.to(device)
        targets = targets.to(device)
        _, outputs = net(inputs)
        _, s_outputs = s_model(inputs)
        post_id = post_process(args, inputs.cpu(), outputs.cpu(), targets.cpu(), s_outputs.cpu(), post_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    # System settings.
    parser.add_argument("--model", type=str, default='pspnet_res50', # Need to be fixed
                        help="Model Name")
    parser.add_argument("--data", type=str, default="enhancing",
                        help="Label data type: [complete, core, enhancing].")
    parser.add_argument("--domain", type=str, default="target",
                        help="select domain: [source, target].")
    parser.add_argument("--mode", type=str, default="test",
                        help="select data mode: [valid, test].")

    # Hyperparameters.
    parser.add_argument("--batch_size", type=int, default=155, # Need to be fixed
                        help="The batch size to load the data")

    # Data paths.
    parser.add_argument("--source_root", type=str, default="/home/featurize/data/BraTS2D/t1ce",
                        help="The directory containing the training image dataset.")
    parser.add_argument("--target_root", type=str, default="/home/featurize/data/BraTS2D/t2",
                        help="The directory containing the training image dataset.")
    parser.add_argument("--label_root", type=str, default="/home/featurize/data/BraTS2D/segs",
                        help="The directory containing the training label datgaset")
    parser.add_argument("--ckpt_root", type=str, default="/home/featurize/work/finalyearproject/Segmentation/log/bbuda/enhancing/2022_08_31_18_00/checkpoint.tar",
                        help="The directory containing the trained model checkpoint")
    parser.add_argument("--ckpt_root_s", type=str, default="/home/featurize/work/finalyearproject/Segmentation/log/source/enhancing/2022_08_25_16_32/checkpoint.pth",
                        help="The directory containing the trained model checkpoint")

    # Output Path.
    parser.add_argument("--output_root", type=str, default="./log/visualise/",
                        help="The directory containing the results.")

    args = parser.parse_args()

    args.output_root += args.data + '/'
    os.makedirs(args.output_root, exist_ok=True)
    test(args)
