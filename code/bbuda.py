import argparse
import copy
from datetime import datetime
import os

import torch
import torch.backends.cudnn as cudnn
import numpy as np

from config import *
from dataset import *
from models import *
from utils import *
from dice_loss import DiceLoss

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def train(args):

    # System Init
    device = config.device
    cudnn.benchmark = True
    criterian = DiceLoss()
    get_logger()

    # Data Load
    validloader = data_loader(args, mode='val', domain="target", bs=155)
    testloader = data_loader(args, mode='test', domain="target", bs=155)
    trainloader = data_loader(args, mode='target', method="bbuda")

    # Model Load
    net, _, best_score, start_epoch =\
        load_model(args, class_num=config.class_num, mode='train_target')
    optimizer = Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.99))

    source_model = copy.deepcopy(net)
    source_model.eval()

    total_iter = args.epochs * len(trainloader)
    for epoch in range(start_epoch, start_epoch + args.epochs):
        # generate pseudo labels
        torch.cuda.empty_cache()
        # Train Model
        print('\n\n\nEpoch: {}\n<Train>'.format(epoch))
        net.train(True)
        loss_all, loss_kl = 0, 0
        loss_all_true = 0
        im_loss_all = 0
        lr = args.lr * (0.5 ** (epoch // 4))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        torch.set_grad_enabled(True)

        for idx, (inputs, true_labels) in enumerate(trainloader):
            alpha = 5 * (1.0 - ((epoch-1)*len(trainloader)+idx) / total_iter)

            torch.cuda.empty_cache()
            inputs = inputs.to(device)
            true_labels = true_labels.to(device)

            _, outputs = net(inputs)
            with torch.no_grad():
                _, source_out = source_model(inputs)

            kl_loss = KL_loss(outputs, source_out, (epoch-1)*len(trainloader)+idx)
            im_loss = Entropy(outputs)

            loss = alpha * im_loss + kl_loss # batch_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss_true = criterian(outputs, true_labels).detach()

            loss_all_true += float(batch_loss_true)
            im_loss_all += float(im_loss)
            loss_all += float(loss)
            loss_kl += float(kl_loss)

            progress_bar(idx, len(trainloader), 'Loss: %.5f, im: %.4f, kl: %.4f, T Di: %.4f'
                         %(float(loss), (im_loss_all/(idx+1)), loss_kl/(idx+1),
                           (1-(loss_all_true/(idx+1)))))
        del inputs, outputs, true_labels


        # Save Model
        loss_all /= (idx + 1)
        score = 1 - loss_all
        if score > best_score:
            checkpoint = Checkpoint(net, optimizer, epoch, score)
            checkpoint.save(os.path.join(args.output_root, "checkpoint.tar"))
            best_score = score
            print("Saving...")

        torch.cuda.empty_cache()
        # Validate Model
        if (epoch + 1) % args.valid_num == 0:
            print('\n\n<Validation target>')
            net.eval()
            loss_all = 0
            torch.set_grad_enabled(False)
            for idx, (inputs, targets) in enumerate(validloader):
                inputs, targets = inputs.to(device), targets.to(device)
                _, outputs = net(inputs)
                # batch_loss, batch_obj_loss = dice_coef(outputs, targets, backprop=False)
                batch_loss = criterian(outputs, targets).detach()
                loss_all += float(batch_loss)
                progress_bar(idx, len(validloader), 'Loss: %.5f, Dice-Coef: %.4f'
                             % ((loss_all / (idx + 1)), (1 - (loss_all / (idx + 1)))))
            del inputs, outputs

            with open(os.path.join(args.output_root, 'args.txt'), 'a') as file:
                file.write('Epoch %d Val set Dice score %.5f\n' % (epoch, 1 - (loss_all / (idx + 1))))


    # Validate Model
    print('\n\n<Test target>')
    net.eval()
    loss_all = 0
    torch.set_grad_enabled(False)
    for idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        _, outputs = net(inputs)
        batch_loss = criterian(outputs, targets).detach()
        loss_all += float(batch_loss)
        progress_bar(idx, len(testloader), 'Loss: %.5f, Dice-Coef: %.4f'
                        % ((loss_all / (idx + 1)), (1 - (loss_all / (idx + 1)))))
    del inputs, outputs
    with open(os.path.join(args.output_root, 'args.txt'), 'a') as file:
        file.write('Epoch %d Test set Dice score %.5f\n' % (epoch, 1 - (loss_all / (idx + 1))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    # System settings.
    parser.add_argument("--resume", type=bool, default=True,
                        help="Model Trianing resume.")
    parser.add_argument("--data", type=str, default="enhancing",
                        help="Label data type: (complete, core, enhancing).")
    parser.add_argument("--model", type=str, default='pspnet_res50',
                        help="Model Name (unet, pspnet_squeeze, pspnet_res50,\
                        pspnet_res34, pspnet_res50, deeplab)")
    parser.add_argument("--method", type=str, default='bbuda',
                        help="method: (ours, bbuda)")
    parser.add_argument("--in_channel", type=int, default=1,
                        help="A number of images to use for input")

    # Hyperparameters.
    parser.add_argument("--batch_size", type=int, default=40,
                        help="The batch size to load the data")
    parser.add_argument("--epochs", type=int, default=1,
                        help="The training epochs to run.")
    parser.add_argument("--valid_num", type=int, default=1,
                        help="The valid epochs to run.")
    parser.add_argument("--drop_rate", type=float, default=0.1,
                        help="Drop-out Rate")
    parser.add_argument("--lr", type=float, default=0.0005,
                        help="Learning rate to use in training")

    # Data paths.
    parser.add_argument("--source_root", type=str, default="/home/featurize/data/BraTS2D/t1ce",
                        help="The directory containing the training image dataset.")
    parser.add_argument("--target_root", type=str, default="/home/featurize/data/BraTS2D/t2",
                        help="The directory containing the training image dataset.")
    parser.add_argument("--ckpt_root", type=str, default="/home/featurize/work/newstart/Segmentation/log/source/enhancing/2022_08_25_16_32/checkpoint.pth",
                        help="The directory containing the checkpoint files")

    # Output root.
    parser.add_argument("--output_root", type=str, default="/home/featurize/work/newstart/Segmentation/log/bbuda/",
                        help="The directory containing the result predictions")
    args = parser.parse_args()

    assert args.data in args.ckpt_root, "Check data option and ckpt_root option"

    args.output_root += args.data + '/' + datetime.now().strftime("%Y_%m_%d_%H_%M") + '/'
    os.makedirs(args.output_root, exist_ok=True)

    options = vars(args)
    with open(os.path.join(args.output_root, 'args.txt'), 'w') as file:
        for key, value in options.items():
            file.write("%s: %s\n" % (key, value))
    set_seed(666)
    train(args)
