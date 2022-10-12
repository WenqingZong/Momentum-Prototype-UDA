import pdb
import cv2
import os
import numpy as np
import nibabel as nib
import torch
import sys
import time
import logging
import logging.handlers
import pydensecrf.densecrf as dcrf
import torch.nn as nn
from dice_loss import DiceLoss
from matplotlib import pyplot as plt

from pydensecrf.utils import compute_unary, create_pairwise_bilateral,\
         create_pairwise_gaussian, softmax_to_unary, unary_from_softmax

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
criterian = DiceLoss(reduction='none')

def Entropy(outputs_target):
    # input_: bs, 2, 240, 240
    # input_ = input_.permute(0,2,3,1).contiguous().view(-1, 2)
    bs = outputs_target.size(0)
    epsilon = 1e-5
    entropy = -outputs_target * torch.log(outputs_target + epsilon)
    entropy = torch.mean(torch.sum(entropy, dim=1))

    return entropy

def dice_coef(preds, targets, backprop=True):
    smooth = 1.0
    class_num = 2
    if backprop:
        for i in range(class_num):
            pred = preds[:,i,:,:]
            target = targets[:,i,:,:]
            intersection = (pred * target).sum()
            loss_ = 1 - ((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))
            if i == 0:
                loss = loss_
            else:
                loss = loss + loss_
        loss = loss/class_num
        return loss, loss_
    else:
        # Need to generalize
        targets = targets.argmax(1).cpu().numpy()
        if len(preds.shape) > 3:
            preds = preds.argmax(1).cpu().numpy()
        for i in range(class_num):
            pred = (preds==i).astype(np.uint8)
            target= (targets==i).astype(np.uint8)
            intersection = (pred * target).sum()
            loss_ = 1 - ((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))
            if i == 0:
                loss = loss_
            else:
                loss = loss + loss_
        loss = loss/class_num
        return loss, loss_

def KL_loss(outputs, source_outputs, iter):
    # outputs： bs, 2, 240, 240
    # source_outputs: bs, 2, 240, 240
    outputs = outputs.permute(0, 2, 3, 1).contiguous().view(-1, 2)
    source_outputs = source_outputs.permute(0, 2, 3, 1).contiguous().view(-1, 2)

    epsilon = 1e-5
    lamb = torch.exp(torch.tensor(-iter * 1.0))
    y_final = lamb * source_outputs + (1 - lamb) * outputs.detach()

    outputs = torch.log(outputs + epsilon)

    kl_loss = -torch.mean((outputs * y_final).mean(dim=1))
    return kl_loss


def get_crf_img(inputs, outputs):
    for i in range(outputs.shape[0]):
        img = inputs[i]
        softmax_prob = outputs[i]
        unary = unary_from_softmax(softmax_prob)
        unary = np.ascontiguousarray(unary)
        d = dcrf.DenseCRF(img.shape[0] * img.shape[1], 2)
        d.setUnaryEnergy(unary)
        feats = create_pairwise_gaussian(sdims=(10,10), shape=img.shape[:2])
        d.addPairwiseEnergy(feats, compat=3, kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)
        feats = create_pairwise_bilateral(sdims=(50,50), schan=(20,20,20),
                                          img=img, chdim=2)
        d.addPairwiseEnergy(feats, compat=10, kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)
        Q = d.inference(5)
        res = np.argmax(Q, axis=0).reshape((img.shape[0], img.shape[1]))
        if i == 0:
            crf = np.expand_dims(res,axis=0)
        else:
            res = np.expand_dims(res,axis=0)
            crf = np.concatenate((crf,res),axis=0)
    return crf


def erode_dilate(outputs, kernel_size=7):
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    outputs = outputs.astype(np.uint8)
    for i in range(outputs.shape[0]):
        img = outputs[i]
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        outputs[i] = img
    return outputs


def compute_ratio(pseudo):
    # compute the label ratio
    # pseudo: bs, 2, 240, 240
    tumor_ratio = torch.sum(pseudo[:,1:,...], dim=[1,2,3])  # bs
    tumor_ratio /= pseudo.shape[2] * pseudo.shape[3]  # bs
    return tumor_ratio


def obj_dice(preds, targets):
    targets = targets.argmax(1).cpu().numpy()  # bs, 240, 240
    if len(preds.shape) > 3:
        preds = preds.argmax(1).cpu().numpy()
    smooth = 1.0
    pred = (preds == 1).astype(np.uint8)
    target = (targets == 1).astype(np.uint8)
    intersection = (pred * target).sum(axis=1).sum(axis=1)
    loss = 1 - ((2.0 * intersection + smooth) / (pred.sum(axis=1).sum(axis=1) + target.sum(axis=1).sum(axis=1) + smooth))
    return 1 - loss


def over_lap(img, input, max_v):
    # img: 240, 240
    # input: 240, 240
    img = np.expand_dims(img, axis=2)  # 240， 240， 1
    zeros = np.zeros(img.shape)
    img = np.concatenate((zeros, zeros, img), axis=2)  # 240， 240， 3
    img = np.array(img).astype(np.float32)
    img = input + img
    if img.max() > 0:
        img = (img / max_v) * 255
    else:
        img = (img / 1) * 255
    return img

def post_process(args, inputs, outputs, targets, s_outputs, post_id=0, thres1=0.01, thres2=0.1,
                 crf_flag=True, erode_dilate_flag=True,
                 save=True, overlap=True):

    tumor_ratio = compute_ratio(targets)
    if tumor_ratio.max() < thres1:
        print("Max tumor ratio:", tumor_ratio.max())
        return post_id

    dice_obj = 1 - criterian(outputs[:,1:,...], targets[:,1:,...])

    if dice_obj[tumor_ratio >= thres1].max() < thres2 or dice_obj.mean() < 0.5:
        print(dice_obj[tumor_ratio >= thres1].max(), dice_obj.mean())
        return post_id

    post_id += 1
    if post_id >= 31:
        sys.exit()
    # inputs: bs, 1, 240, 240
    # outputs: bs, 2, 240, 240
    # targets: bs, 2, 240, 240

    inputs = ((np.array(inputs.squeeze()).astype(np.float32)) * 255).reshape(-1, 240, 240)
    inputs = np.expand_dims(inputs, axis=3)
    inputs = np.concatenate((inputs,inputs,inputs), axis=3)
    outputs = np.array(outputs)
    s_outputs = np.array(s_outputs)
    targets = np.array(targets)

    # Conditional Random Field
    if crf_flag:
        outputs = get_crf_img(inputs, outputs)
        s_outputs = get_crf_img(inputs, s_outputs)
    else:
        outputs = outputs.argmax(1)
        s_outputs = s_outputs.argmax(1)
    targets = targets.argmax(1)

    # Erosion and Dilation
    if erode_dilate_flag:
        outputs = erode_dilate(outputs, kernel_size=7)
        s_outputs = erode_dilate(s_outputs, kernel_size=7)
    if save == False:
        return outputs

    outputs = outputs * 255
    s_outputs = s_outputs * 255
    targets = targets * 255
    print(outputs.shape, s_outputs.shape, targets.shape)
    print(outputs.min(), outputs.max(), s_outputs.min(), s_outputs.max(), targets.min(), targets.max())
    os.makedirs(args.output_root + str(post_id) + '/', exist_ok=True)
    max_v = inputs.max() + max(outputs.max(), targets.max(), s_outputs.max())
    for i in range(outputs.shape[0]):
        img_out = over_lap(outputs[i], inputs[i], max_v)
        img_target = over_lap(targets[i], inputs[i], max_v)
        img_out_s = over_lap(s_outputs[i], inputs[i], max_v)
        cv2.imwrite(args.output_root + str(post_id) + '/' + str(i) + 'target_output.jpg', img_out)
        cv2.imwrite(args.output_root + str(post_id) + '/' + str(i) + 'source_output.jpg', img_out_s)
        cv2.imwrite(args.output_root + str(post_id) + '/' + str(i) + 'ground_truth.jpg', img_target)
    return post_id


class Checkpoint:
    def __init__(self, model, optimizer=None, epoch=0, best_score=0):
        self.model = model
        self.optimizer = optimizer
        self.epoch = epoch
        self.best_score = best_score

    def load(self, path, mode):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state"])
        if mode != "train_target":
            self.epoch = checkpoint["epoch"]
            self.best_score = checkpoint["best_score"]
            if self.optimizer:
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
                for state in self.optimizer.state.values():
                      for k, v in state.items():
                               if torch.is_tensor(v):
                                        state[k] = v.cuda()

    def save(self, path):
        state_dict = self.model.module.state_dict()
        torch.save({"model_state": state_dict,
                    "optimizer_state": self.optimizer.state_dict(),
                    "epoch": self.epoch,
                    "best_score": self.best_score}, path)


def progress_bar(current, total, msg=None):
    ''' Source Code from 'kuangliu/pytorch-cifar'
        (https://github.com/kuangliu/pytorch-cifar/blob/master/utils.py)
    '''
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    ''' Source Code from 'kuangliu/pytorch-cifar'
        (https://github.com/kuangliu/pytorch-cifar/blob/master/utils.py)
    '''
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def get_logger(level="DEBUG", file_level="DEBUG"):
    logger = logging.getLogger(None)
    logger.setLevel(level)
    fomatter = logging.Formatter(
            '%(asctime)s  [%(levelname)s]  %(message)s  (%(filename)s:  %(lineno)s)')
    fileHandler = logging.handlers.TimedRotatingFileHandler(
            'result.log', when='d', encoding='utf-8')
    fileHandler.setLevel(file_level)
    fileHandler.setFormatter(fomatter)
    logger.addHandler(fileHandler)
    return logger
