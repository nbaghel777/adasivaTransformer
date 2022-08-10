import os
import glob
import torch
import torchvision
import torchvision.transforms as T
from importlib import import_module
from torch.utils.data import Dataset, Sampler, DataLoader

class CrypkoDataset(Dataset):
    def __init__(self, fnames, transform):
        self.transform = transform
        self.fnames = fnames
        self.num_samples = len(self.fnames)

    def __getitem__(self,idx):
        fname = self.fnames[idx]
        # 1. Load the image
        img = torchvision.io.read_image(fname)
        # 2. Resize and normalize the images using torchvision.
        img = self.transform(img)
        return img

    def __len__(self):
        return self.num_samples


class InfiniteSampler(Sampler):
    def __init__(self, data_source):
        super(InfiniteSampler, self).__init__(data_source)
        self.N = len(data_source)


    def __iter__(self):
        while True:
            for idx in torch.randperm(self.N):
                yield idx
                
                

def get_dataset(root, img_size = 64):
    fnames = glob.glob(os.path.join(root, '*'))
    #print(len(fnames),root )
    compose = [
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)),
    ]
    transform = T.Compose(compose)
    dataset = CrypkoDataset(fnames, transform)
    return dataset

def get_dataloader(root, img_size = 64, batch_size = 64):
    data = get_dataset(root, img_size)
    noofbatch=0
    if len(data)%batch_size>0:
        noofbatch=int(len(data)/batch_size)+1
    else:
        noofbatch=int(len(data)/batch_size)
    data_loader = iter(
        DataLoader(
            data,
            batch_size = batch_size,
            num_workers = 1,
            sampler = InfiniteSampler(data)
        )
    )
    return data_loader,noofbatch
    
#--------------ssim------------
import numpy as np
import math
import cv2
from math import log10, sqrt
def calc_ssim(img1, img2):
    def ssim(img1, img2):
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    ### args:
        # img1: [h, w, c], range [0, 255]
        # img2: [h, w, c], range [0, 255]
        # the same outputs as MATLAB's
    border = 0
    img1_y = np.dot(img1, [65.738,129.057,25.064])/256.0+16.0
    img2_y = np.dot(img2, [65.738,129.057,25.064])/256.0+16.0
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1_y = img1_y[border:h-border, border:w-border]
    img2_y = img2_y[border:h-border, border:w-border]

    if img1_y.ndim == 2:
        return ssim(img1_y, img2_y)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def calc_psnr(sr, hr, scale=1, rgb_range=255, cal_type='y'):
    #if hr.nelement() == 1: return 0
    #print(sr.shape,hr.shape)
    diff = (sr - hr) / rgb_range
    
    if cal_type=='y':
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        diff = diff.mul(convert).sum(dim=1)
    
    if scale == 1:
        valid = diff
    else:
        valid = diff[..., scale:-scale, scale:-scale]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)

def calc_psnr_and_ssim(sr, hr):  
    ### prepare data
    sr = (sr+1.) * 127.5
    hr = (hr+1.) * 127.5
    
    if (sr.size() != hr.size()):
        h_min = min(sr.size(2), hr.size(2))
        w_min = min(sr.size(3), hr.size(3))
        sr = sr[:, :, :h_min, :w_min]
        hr = hr[:, :, :h_min, :w_min]

    img1 = np.transpose(sr.squeeze().round().cpu().numpy(), (1,2,0))
    img2 = np.transpose(hr.squeeze().round().cpu().numpy(), (1,2,0))
    psnr = calc_psnr(sr, hr)
    ssim = calc_ssim(img1, img2)
    return psnr, ssim

#-------------------------------------------------------------------
import shutil
import logging
class Logger(object):
    def __init__(self, log_file_name, logger_name, log_level=logging.DEBUG):
        ### create a logger
        self.__logger = logging.getLogger(logger_name)

        ### set the log level
        self.__logger.setLevel(log_level)

        ### create a handler to write log file
        file_handler = logging.FileHandler(log_file_name)

        ### create a handler to print on console
        console_handler = logging.StreamHandler()

        ### define the output format of handlers
        #formatter = logging.Formatter('[%(asctime)s] - [%(filename)s file line:%(lineno)d] - %(levelname)s: %(message)s')
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        ### add handler to logger
        self.__logger.addHandler(file_handler)
        #self.__logger.addHandler(console_handler)

    def get_log(self):
        return self.__logger

def mkExpDir(save_dir):
    if (os.path.exists(save_dir)):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    os.makedirs(os.path.join(save_dir, 'saliency'))
    os.makedirs(os.path.join(save_dir, 'save_results'))
    os.makedirs(os.path.join(save_dir, 'diff'))
    os.makedirs(os.path.join(save_dir, 'test_results'))
    os.makedirs(os.path.join(save_dir, 'weights'))
    args_file = open(os.path.join(save_dir, 'args.txt'), 'w')
    _logger = Logger(log_file_name=os.path.join(save_dir,'model.log'), 
        logger_name='modellog').get_log()
    return _logger