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
    
