import os
import argparse
import warnings
warnings.filterwarnings('ignore')

import torch
import cv2
import numpy as np
import skimage.color as sc

import utils
from HPINet import HPINet


# testing settings
parser = argparse.ArgumentParser(description='HPINet')
parser.add_argument('--model', type=str, default='M', choices=['S','M','L'],
                    help='model size')
parser.add_argument('--scale', type=int, default=4,
                    help='upscaling factor')
parser.add_argument('--is_y', default=True,
                    help='evaluate on y channel, if False evaluate on RGB channels')
opt = parser.parse_args()
print(opt)

# please modify test dataset path if needed: {'HR_PATH': 'LR_PATH'}
test_dataset_folder = {'benchmarks/Set5/': 'benchmarks/Set5_LR/x{}/'.format(opt.scale),
                       'benchmarks/Set14/HR/': 'benchmarks/Set14/LR_bicubic/X{}/'.format(opt.scale),
                       'benchmarks/B100/HR/': 'benchmarks/B100/LR_bicubic/X{}/'.format(opt.scale),
                       'benchmarks/Urban100/HR/': 'benchmarks/Urban100/LR_bicubic/X{}/'.format(opt.scale),
                       'benchmarks/Manga109/HR/': 'benchmarks/Manga109/LR_bicubic/X{}/'.format(opt.scale),
                       }
checkpoints = {'M': {'2': 'checkpoints/HPINet-M-x2.pth',
                     '3': 'checkpoints/HPINet-M-x3.pth',
                     '4': 'checkpoints/HPINet-M-x4.pth'},
               'S': {'4': 'checkpoints/HPINet-S-x4.pth'},
               'L': {'4': 'checkpoints/HPINet-L-x4.pth'},
               }
model_settings = \
    {'M': dict(dim=56, block_num=8, heads=1, qk_dim=32, mlp_dim=100, patch_size=[12,16,20,24,12,16,20,24]),
     'S': dict(dim=40, block_num=8, heads=1, qk_dim=24, mlp_dim=72, patch_size=[12,16,20,24,12,16,20,24]),
     'L': dict(dim=64, block_num=10, heads=1, qk_dim=36, mlp_dim=128, patch_size=[12,14,16,20,24,12,14,16,20,24]),
     }

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_dic = model_settings[opt.model]
model_dic.update(upscale=opt.scale)
model = HPINet(**model_dic)
model.eval()
pretrained = utils.load_state_dict(checkpoints[opt.model][str(opt.scale)])
model.load_state_dict(pretrained, strict=True)
model = model.to(device)

def test():
    for GT_folder, LR_folder in test_dataset_folder.items():
        if 'Set5' in GT_folder:
            ext = '.bmp'
        else:
            ext = '.png'
        filelist = utils.get_list(GT_folder, ext=ext)
        psnr_list = np.zeros(len(filelist))
        ssim_list = np.zeros(len(filelist))
        for i, imname in enumerate(filelist):
            im_gt = cv2.imread(imname, cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR to RGB
            im_gt = utils.modcrop(im_gt, opt.scale)
            lr_name = LR_folder + imname.split('/')[-1].split('.')[0].replace('_HR_x2', '') + 'x' + str(
                opt.scale) + ext
            im_l = cv2.imread(lr_name, cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR to RGB
            im_input = im_l / 255.0
            im_input = np.transpose(im_input, (2, 0, 1))
            im_input = im_input[np.newaxis, ...]
            im_input = torch.from_numpy(im_input).float()
            im_input = im_input.to(device)
            b, _, H, W = im_input.size()
            with torch.no_grad():
                out = model.forward(im_input)
            out_img = utils.tensor2np(out.detach()[0])
            crop_size = opt.scale
            cropped_sr_img = utils.shave(out_img, crop_size)
            cropped_gt_img = utils.shave(im_gt, crop_size)
            if opt.is_y is True:
                im_label = utils.quantize(sc.rgb2ycbcr(cropped_gt_img)[:, :, 0])
                im_pre = utils.quantize(sc.rgb2ycbcr(cropped_sr_img)[:, :, 0])
            else:
                im_label = cropped_gt_img
                im_pre = cropped_sr_img
            psnr_list[i] = utils.compute_psnr(im_pre, im_label)
            ssim_list[i] = utils.compute_ssim(im_pre, im_label)
        print('{}, Mean PSNR: {:.2f}, SSIM: {:.4f}'.format(GT_folder.split('/')[1],np.mean(psnr_list), np.mean(ssim_list)))

if __name__ == '__main__':
    test()