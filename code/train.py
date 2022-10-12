import argparse
from datetime import datetime
import logging

import torch
import torch.backends.cudnn as cudnn
import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt

from config import *
from dataset import *
from models import *
from utils import *
from dice_loss import DiceLoss

def train(args):

    # Variables and logger Init
    device = config.device
    cudnn.benchmark = True
    get_logger()
    criterian = DiceLoss(molecule=1)
    # Data Load
    trainloader = data_loader(args, mode='train', domain="source")
    s_validloader = data_loader(args, mode='val', domain="source")
    t_validloader = data_loader(args, mode='val', domain="target")

    # Model Load
    net, optimizer, best_score, start_epoch = load_model(args, class_num=config.class_num, mode='train')
    log_msg = '\n'.join(['%s Train Start'%(args.model)])
    logging.info(log_msg)
    best_epoch = 0

    # Plot data
    dice_all = []
    dice_obj = []
    source_val_dice = []
    source_val_obj_dice = []
    target_val_dice = []
    target_val_obj_dice = []

    label = {'complete': 'Whole Tumor', 'core': 'Tumor Core', 'enhancing': 'Enhanced Tumor'}

    for epoch in range(start_epoch, start_epoch + args.epochs):

        # Train Model
        print('\n\n\nEpoch: {}\n<Train>'.format(epoch))
        net.train(True)
        loss = 0
        loss_obj = 0
        lr = args.lr * (0.5 ** (epoch // 4))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        torch.set_grad_enabled(True)
        for idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            _, outputs = net(inputs)
            batch_loss = criterian(outputs, targets)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            loss += float(batch_loss)
            progress_bar(idx, len(trainloader), 'Loss: %.5f, Dice-Coef: %.5f'
                         %((loss / (idx + 1)), (1 - (loss / (idx + 1)))))

            dice_all.append((1 - batch_loss).cpu())

        log_msg = '\n'.join(['Epoch: %d  Loss: %.5f,  Dice-Coef:  %.5f'\
                         %(epoch, loss / (idx + 1), 1 - (loss / (idx + 1)))])
        logging.info(log_msg)

        # Validate Model
        if (epoch + 1) % args.valid_interval == 0:
            print('\n\n<Validation source>')
            net.eval()
            loss = 0
            loss_obj = 0
            torch.set_grad_enabled(False)
            for idx, (inputs, targets) in enumerate(s_validloader):
                inputs, targets = inputs.to(device), targets.to(device)
                _, outputs = net(inputs)
                batch_loss = criterian(outputs, targets)
                loss += float(batch_loss)
                progress_bar(idx, len(s_validloader), 'Loss: %.5f, Dice-Coef: %.5f'
                             % ((loss / (idx + 1)), (1 - (loss / (idx + 1)))))
            log_msg = '\n'.join(['Epoch: %d  Loss: %.5f,  Source: Dice-Coef:  %.5f'
                            %(epoch, loss / (idx + 1), 1 - (loss / (idx + 1)))])
            logging.info(log_msg)
            source_val_dice.append(1 - (loss / (idx + 1)))
            source_val_obj_dice.append(1 - (loss_obj / (idx + 1)))

            # Save Model
            loss /= (idx + 1)
            score = 1 - loss
            print("Best Epoch:", best_epoch)
            if score > best_score:
                checkpoint = Checkpoint(net, optimizer, epoch, score)
                checkpoint.save(os.path.join(args.output_root, 'checkpoint.pth'))
                best_score = score
                best_epoch = epoch
                print("Saving...")

            print('\n\n<Validation target>')
            loss = 0
            loss_obj = 0
            for idx, (inputs, targets) in enumerate(t_validloader):
                inputs, targets = inputs.to(device), targets.to(device)
                _, outputs = net(inputs)
                batch_loss = criterian(outputs, targets)
                loss += float(batch_loss)
                progress_bar(idx, len(t_validloader), 'Target: Loss: %.5f, Dice-Coef: %.5f'
                             % ((loss / (idx + 1)), (1 - (loss / (idx + 1)))))
            log_msg = '\n'.join(['Epoch: %d  Loss: %.5f,  Target: Dice-Coef:  %.5f'
                            %(epoch, loss / (idx + 1), 1 - (loss / (idx + 1)))])
            logging.info(log_msg)
            target_val_dice.append(1 - (loss / (idx + 1)))
            target_val_obj_dice.append(1 - (loss_obj / (idx + 1)))

        # Plot
        plt.figure()
        plt.title("Source Domain Training Dice (Label: %s)" % label[args.data])
        plt.xlabel("Iteration")
        plt.ylabel("Dice")
        plt.plot(dice_all, label="Dice")
        plt.legend()
        plt.savefig(os.path.join(args.output_root, 'source_dice.jpg'))

        plt.figure()
        plt.title("Source Domain Validation Dice (Label: %s)" % label[args.data])
        plt.xlabel("Epoch")
        plt.ylabel("Dice")
        plt.plot(source_val_dice, label="Loss")
        plt.legend()
        plt.savefig(os.path.join(args.output_root, 'source_val_dice.jpg'))

        plt.figure()
        plt.title("Target Domain Validation Dice (Label: %s)" % label[args.data])
        plt.xlabel("Epoch")
        plt.ylabel("Dice")
        plt.plot(target_val_dice, label="Loss")
        plt.legend()
        plt.savefig(os.path.join(args.output_root, 'target_val_dice.jpg'))

        # Save plot data.
        torch.save({
            "dice_all": dice_all,
            "source_val_dice": source_val_dice,
            "target_val_dice": target_val_dice,
        }, os.path.join(args.output_root, "plot_data.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    # System settings.
    parser.add_argument("--resume", type=bool, default=False,
                        help="Model Trianing resume.")
    parser.add_argument("--ckpt_root", type=str, default="",
                        help="load pretrain model path.")
    parser.add_argument("--data", type=str, default="complete",
                        help="Label data type, can be one of [complete, core, enhancing]")
    parser.add_argument("--model", type=str, default='pspnet_res50',
                        help="Model Name (unet, pspnet_squeeze, pspnet_res50,\
                        pspnet_res34, pspnet_res50, deeplab)")
    parser.add_argument("--valid_interval", type=int, default=1,
                        help="Validation Intervals")
    parser.add_argument("--in_channel", type=int, default=1,
                        help="A number of images to use for input")

    # Hyperparameters.
    parser.add_argument("--batch_size", type=int, default=48,
                        help="The batch size to load the data")
    parser.add_argument("--epochs", type=int, default=5,
                        help="The training epochs to run.")
    parser.add_argument("--drop_rate", type=float, default=0.1,
                        help="Drop-out Rate")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate to use in training")

    # Data paths.
    parser.add_argument("--source_root", type=str, default="/home/featurize/data/BraTS2D/t1ce",
                        help="The directory containing the training image dataset.")
    parser.add_argument("--target_root", type=str, default="/home/featurize/data/BraTS2D/t2",
                        help="The directory containing the training image dataset.")

    # Output paths.
    parser.add_argument("--output_root", type=str, default="./log/source/",
                        help="The directory containing the result predictions")
    args = parser.parse_args()

    args.output_root += args.data + '/' + datetime.now().strftime("%Y_%m_%d_%H_%M") + '/'
    print("Output path:", args.output_root)
    os.makedirs(args.output_root, exist_ok=True)
    train(args)
