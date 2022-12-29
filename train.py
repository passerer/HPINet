import os
import random
import argparse
import warnings

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
import skimage.color as sc
from torch.utils.data import DataLoader

from HPINet import HPINet
from dataset import DIV2KTrainDataset, FolderDataset
import utils

# settings
# below we suggest you check and customize them
parser = argparse.ArgumentParser(description='Train HPINet')
parser.add_argument('--exp_name', type=str, default='HPINet',
                    help='experiment name')
parser.add_argument('--model', type=str, default='M', choices=['S', 'M', 'L'],
                    help='model size')
parser.add_argument('--root', type=str, default='Train_Datasets/DIV2K/',
                    help='dataset directory')
parser.add_argument('--ext', type=str, choices=['.npy', '.png'], default='.png',
                    help='image suffix. npy or png is required')
parser.add_argument('--scale', type=int, default=4,
                    help='upscale factor')
parser.add_argument('--isY', action='store_true', default=True,
                    help='evaluate on y channel, if False evaluate on RGB channels')
parser.add_argument('--save_interval', type=int, default=20)
parser.add_argument('--test_interval', type=int, default=1)
parser.add_argument('--log_interval', type=int, default=100)
# if you want to reproduce results in paper, arguments below had better remain unchanged;
# otherwise, you are more than welcome to modify them for something new
parser.add_argument('--epochs', type=int, default=420,
                    help='number of epochs')
parser.add_argument('--start-epoch', default=1, type=int,
                    help='manual start epoch number')
parser.add_argument('--lr', type=float, default=1.5e-4,
                    help='learning rate')
parser.add_argument('--step_size', type=int, default=60,
                    help='learning rate decay per step_size epochs')
parser.add_argument('--max_batch_size', type=int, default=64,
                    help='maximum training batch size')
parser.add_argument('--min_batch_size', type=int, default=8,
                    help='minimum training batch size')
parser.add_argument('--gamma', type=int, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use cuda')
parser.add_argument('--resume', default="", type=str,
                    help='path to checkpoint')
parser.add_argument('--pretrained', default="", type=str,
                    help='path to pretrained models')
parser.add_argument('--threads', type=int, default=8,
                    help='number of threads for data loading')
parser.add_argument('--max_patch_size', type=int, default=720,
                    help='maximum hr size')
parser.add_argument('--min_patch_size', type=int, default=192,
                    help='minimum hr size')
parser.add_argument('--seed', type=int, default=2,
                    help='random seed')
parser.add_argument('--tb_logger', action='store_true', default=False,
                    help='use tb_logger')

args = parser.parse_args()
print(args)
if args.tb_logger:
    from torch.utils.tensorboard import SummaryWriter
    tb_logger = SummaryWriter(log_dir='./tb_logger/' + args.exp_name)

# random seed
seed = args.seed
if seed is None:
    seed = random.randint(1, 10000)
print('Seed: ', seed)
random.seed(seed)
torch.manual_seed(seed)

cuda = args.cuda
device = torch.device('cuda' if cuda else 'cpu')

print('===> Loading datasets')
trainset = DIV2KTrainDataset(args.scale, args.max_patch_size, args.root, args.ext)
testset = FolderDataset('benchmarks/Set5/', 'benchmarks/Set5_LR/x{}/'.format(args.scale), args.scale)
testing_data_loader = DataLoader(dataset=testset, num_workers=1, batch_size=1,
                                 shuffle=False)

print('===> Building models')
args.is_train = True
model = HPINet(args.model, args.scale)

print('===> Setting GPU')
if cuda:
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

if args.pretrained:
    if os.path.isfile(args.pretrained):
        pretrained = utils.load_state_dict(args.pretrained)
        model.load_state_dict(pretrained, strict=False)
    else:
        print('=====> No pretrained models found at {}. Continue.'.format(args.pretrained))

print('===> Setting Optimizer')
optimizer = optim.Adam(model.parameters(), lr=args.lr)


def adjust_learning_rate(optimizer, epoch, step_size, lr_init, gamma):
    factor = epoch // step_size
    lr = lr_init * (gamma ** factor)
    lr = max(lr, 1e-6)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_patch_size(min_patch_size, max_patch_size, scale, epoch, min_divider=8):
    return min_patch_size // scale + \
           int((epoch - 1) * min_divider / scale) % ((max_patch_size - min_patch_size) // scale)


def adjust_batch_size(min_batch_size, max_batch_size, patch_size, min_patch_size, scale):
    return max(min_batch_size, int((max_batch_size / (
            patch_size * scale / min_patch_size) // torch.cuda.device_count() * torch.cuda.device_count())))


def save_checkpoint(epoch):
    ckpt_dir = '{}_checkpoint_x{}/'.format(args.exp_name, args.scale)
    ckpt_path = os.path.join(ckpt_dir, 'epoch_{}.pth'.format(epoch))
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    torch.save(model.state_dict(), ckpt_path)
    print('===> Checkpoint saved to {}'.format(ckpt_path))


def train_epoch(epoch):
    model.train()
    adjust_learning_rate(optimizer, epoch, args.step_size, args.lr, args.gamma)
    ps_lr = adjust_patch_size(args.min_patch_size, args.max_patch_size, args.scale, epoch)
    bs = adjust_batch_size(args.min_batch_size, args.max_batch_size, ps_lr,
                           args.min_patch_size, args.scale)
    trainset.patch_size = ps_lr * args.scale
    trainset.repeat = 1000 // (800 // bs)
    training_data_loader = DataLoader(dataset=trainset, num_workers=args.threads, batch_size=bs,
                                      shuffle=True, pin_memory=True, drop_last=True)
    for iteration, (lr_tensor, hr_tensor) in enumerate(training_data_loader, 1):
        if args.cuda:
            lr_tensor = lr_tensor.to(device)  # ranges from [0, 1]
            hr_tensor = hr_tensor.to(device)  # ranges from [0, 1]

        optimizer.zero_grad()
        loss_l1 = model(lr_tensor, hr_tensor)
        loss_sr = loss_l1.mean()
        loss_sr.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
        optimizer.step()
        if iteration % args.log_interval == 0:
            print('===> Epoch[{}]({}/{}): Loss_l1: {:.5f}'.format(epoch, iteration,
                                                                  len(training_data_loader),
                                                                  loss_sr.item()))
            if args.tb_logger:
                tb_logger.add_scalar('loss', loss_sr.item(),
                                     (epoch - 1) * len(training_data_loader) + iteration)


def valid(epoch):
    model.eval()
    avg_psnr, avg_ssim = 0, 0
    for batch in testing_data_loader:
        lr_tensor, hr_tensor = batch[0], batch[1]
        if args.cuda:
            lr_tensor = lr_tensor.to(device)
            hr_tensor = hr_tensor.to(device)

        with torch.no_grad():
            pre = model(lr_tensor)

        sr_img = utils.tensor2np(pre.detach()[0])
        gt_img = utils.tensor2np(hr_tensor.detach()[0])
        crop_size = args.scale
        cropped_sr_img = utils.shave(sr_img, crop_size)
        cropped_gt_img = utils.shave(gt_img, crop_size)
        if args.isY is True:
            im_label = utils.quantize(sc.rgb2ycbcr(cropped_gt_img)[:, :, 0])
            im_pre = utils.quantize(sc.rgb2ycbcr(cropped_sr_img)[:, :, 0])
        else:
            im_label = cropped_gt_img
            im_pre = cropped_sr_img
        avg_psnr += utils.compute_psnr(im_pre, im_label)
        avg_ssim += utils.compute_ssim(im_pre, im_label)
    avg_psnr = avg_psnr / len(testing_data_loader)
    avg_ssim = avg_ssim / len(testing_data_loader)
    print('===> Valid psnr: {:.4f}, ssim: {:.4f}'.format(avg_psnr, avg_ssim))
    if args.tb_logger:
        tb_logger.add_scalar('psnr', avg_psnr, (epoch - 1))
    return avg_psnr


def train():
    print('===> Training')
    for epoch in range(args.start_epoch, args.epochs + 1):
        train_epoch(epoch)
        if epoch % args.test_interval == 0:
            valid(epoch)
        if epoch % args.save_interval == 0:
            save_checkpoint(epoch)


if __name__ == '__main__':
    train()
