import argparse
from datetime import datetime
import logging
import os

import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt
import torch
import torch.backends.cudnn as cudnn

from generate_pseudo import compute_pro, get_new_pseudo, compute_ratios
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
    # System init.
    device = config.device
    cudnn.benchmark = True
    get_logger()
    criterian = DiceLoss()

    # Data Load
    validloader = data_loader(args, mode='val', domain="target", bs=155)
    train_eval_loader = data_loader(args, mode='generate_pseudo', domain="target", bs=155)

    # Model Load
    net, optimizer, best_score, start_epoch = load_model(args, class_num=config.class_num, mode='train_target')
    log_msg = '\n'.join(['%s Train Start'%(args.model)])
    logging.info(log_msg)

    # Plot data.
    entropy_loss = []
    dice_all_loss = []
    true_dice_all = []
    val_dice_all = []

    for epoch in range(start_epoch, start_epoch + args.epochs):
        torch.cuda.empty_cache()
        net.eval()
        print("\n< Generate pseudo labels >")

        if epoch == start_epoch and os.path.exists(args.ratios) and os.path.exists(args.pro_bck) and os.path.exists(args.pro_obj):
            pro_bck = torch.from_numpy(np.load(args.pro_bck)).to(device)
            pro_obj = torch.from_numpy(np.load(args.pro_obj)).to(device)
            ratios = np.load(args.ratios)
        else:
            ratios, pro_bck, pro_obj = compute_pro(args, net, train_eval_loader)  # Update Prototype feature
        pro_bck, pro_obj = pro_bck.to(device), pro_obj.to(device)

        print("\n< re-split target data >")
        trainloader = data_loader(args, mode='target', ratio=ratios)

        # Train Model
        print('\n\n\nEpoch: {}\n<Train>'.format(epoch))
        net.train(True)
        lr = args.lr * (0.5 ** (epoch / 4))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        torch.set_grad_enabled(True)

        im_loss_total, batch_loss_total, batch_loss_true_total = 0, 0, 0

        for idx, (inputs, true_labels) in enumerate(trainloader):
            torch.cuda.empty_cache()
            inputs = inputs.to(device)
            true_labels = true_labels.to(device)
            feature, outputs = net(inputs)
            new_targets, pro_bck, pro_obj = get_new_pseudo(args, feature, outputs, pro_bck, pro_obj)
            im_loss = 10 * Entropy(outputs)
            batch_loss = criterian(outputs, new_targets)

            loss = im_loss + batch_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss_true  = criterian(outputs, true_labels).detach().item()

            im_loss_total += im_loss.item()
            batch_loss_total += batch_loss.item()
            batch_loss_true_total += batch_loss_true

            entropy_loss.append(im_loss_total / (1 + idx))
            dice_all_loss.append(batch_loss_total / (1 + idx))
            true_dice_all.append(1 - batch_loss_true_total / (1 + idx))

            progress_bar(idx, len(trainloader), 'Loss: %.5f, im: %.4f, Dice: %.4f, T Dice: %.4f'
                         %(float(loss), (im_loss / (1 + idx)), (1 - batch_loss_total / (1 + idx)), 1 - batch_loss_true_total / (1 + idx)))

            if idx % 300 == 0:
                del inputs, feature, outputs, new_targets, true_labels

                print('\n\n<Validation target>')
                net.eval()
                torch.cuda.empty_cache()
                torch.set_grad_enabled(False)
                loss_all_val = 0
                for idx_v, (inputs, targets) in enumerate(validloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    feature, outputs = net(inputs)
                    batch_loss = criterian(outputs, targets).detach()
                    loss_all_val += float(batch_loss.item())
                    progress_bar(idx_v, len(validloader), 'Loss: %.5f, Dice-Coef: %.4f'
                                    % (loss_all_val / (idx_v + 1), 1 - (loss_all_val / (idx_v + 1))))
                loss_all_val /= (idx_v + 1)
                score = 1 - loss_all_val
                val_dice_all.append(score)
                torch.set_grad_enabled(True)
                net.train()

                if score > best_score:
                    checkpoint = Checkpoint(net, optimizer, epoch, score)
                    checkpoint.save(os.path.join(args.output_root, f'checkpoint.pth'))
                    best_score = score
                    print("Saving...")

                # Plot.
                plt.figure()
                plt.title('Loss During Target Domain Adaptation')
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.plot(entropy_loss, label='Entropy Loss')
                plt.plot(dice_all_loss, label='Dice Loss')
                plt.legend()
                plt.savefig(os.path.join(args.output_root, 'target_training_loss.jpg'))
                plt.close()

                plt.figure()
                plt.title('Target Domain True Dice (Compared With Ground Truth Label)')
                plt.ylabel('Dice')
                plt.xlabel('Iteration')
                plt.plot(true_dice_all, label='True Dice')
                plt.legend()
                plt.savefig(os.path.join(args.output_root, 'target_training_true_dice.jpg'))
                plt.close()

                # Plot.
                plt.figure()
                plt.title('Target Domain Validation Set Dice')
                plt.ylabel('Dice')
                plt.xlabel('Epoch')
                plt.plot(val_dice_all, label='Dice')
                plt.legend()
                plt.savefig(os.path.join(args.output_root, 'target_val_dice.jpg'))
                plt.close()

                # Save plot data.
                torch.save({
                    'entropy_loss': entropy_loss,
                    'dice_all_loss': dice_all_loss,
                    'true_dice_all': true_dice_all,
                    'val_dice_all': val_dice_all,
                }, os.path.join(args.output_root, 'plot_data.pth'))

                del inputs, feature, outputs
                with open(os.path.join(args.output_root, 'args.txt'), 'a') as file:
                    file.write("Epoch %d, val set true dice: %.5f\n" % (epoch, score))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    # System settings.
    parser.add_argument("--resume", type=bool, default=True,
                        help="Model Trianing resume.")
    parser.add_argument("--data", type=str, default="enhancing",
                        help="Label data type, can be one of [complete, core, enhancing]")
    parser.add_argument("--model", type=str, default='pspnet_res50',
                        help="Model Name (unet, pspnet_squeeze, pspnet_res50,\
                        pspnet_res34, pspnet_res50, deeplab)")
    parser.add_argument("--valid_interval", type=int, default=1,
                        help="Validation Intervals")
    parser.add_argument("--in_channel", type=int, default=1,
                        help="A number of images to use for input")

    # Hyperparameters.
    parser.add_argument("--batch_size", type=int, default=40,
                        help="The batch size to load the data")
    parser.add_argument("--epochs", type=int, default=1,
                        help="The training epochs to run.")
    parser.add_argument("--drop_rate", type=float, default=0.1,
                        help="Drop-out Rate")
    parser.add_argument("--lr", type=float, default=0.0005,
                        help="Learning rate to use in training")
    parser.add_argument("--moment", type=float, default=0.995,
                        help="The ratio of updating the prototype")
    parser.add_argument("--pseudo_thresh", type=float, default=0.75,
                        help="Threshold used to filter out output probability")

    # Data paths.
    parser.add_argument("--source_root", type=str, default="/home/featurize/data/BraTS2D/t1ce",
                        help="The directory containing the training image dataset.")
    parser.add_argument("--target_root", type=str, default="/home/featurize/data/BraTS2D/t2",
                        help="The directory containing the training image dataset.")
    parser.add_argument("--ckpt_root", type=str, default="/home/featurize/work/finalyearproject/Segmentation/log/source/enhancing/2022_08_25_16_32/checkpoint.pth",
                        help="The path to the checkpoint file")
    parser.add_argument("--pro_bck", type=str, default="",
                        help="If source model is not changed, directly load this data to save time")
    parser.add_argument("--pro_obj", type=str, default="",
                        help="If source model is not changed, directly load this data to save time")
    parser.add_argument("--ratios", type=str, default="",
                        help="If source model is not changed, directly load this data to save time")

    # Output paths.
    parser.add_argument("--output_root", type=str, default="/home/featurize/work/finalyearproject/Segmentation/log/target/",
                        help="The directory containing the result predictions")
    args = parser.parse_args()

    args.output_root += args.data + '/' + datetime.now().strftime("%Y_%m_%d_%H_%M") + '/'
    os.makedirs(args.output_root, exist_ok=True)
    print("Output path:", args.output_root)

    options = vars(args)
    with open(os.path.join(args.output_root, 'args.txt'), 'w') as file:
        for key, value in options.items():
            file.write("%s: %s\n" % (key, value))
    set_seed(666)
    train(args)
