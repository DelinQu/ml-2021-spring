import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class COVID19Dataset(Dataset):
    def __init__(self, path, mode='train', target_only=False):
        self.mode = mode

        # read data from file
        with open(path,'r') as fp:
            data = list(csv.reader(fp))
            data = np.array(data[1:])[:, 1:].astype(float)

        # Set feature by target_only
        if not target_only:
            feats = list(range(93))
        else:
            # TODO: Using 40 states & 2 tested_positive features (indices = 57 & 75)
            feats = list(range(40)) + [57,75]

        # Set the dataset
        if mode == 'test':
            data = data[:,feats]
            self.data = torch.FloatTensor(data)
        else:
            target = data[:,-1]
            data = data[:,feats]

            # splitting training data into train & dev sets
            if mode =='train':
                idx = [i for i in range(len(data)) if i%10 !=0]
            elif mode=='dev':
                idx = [i for i in range(len(data)) if i%10 ==0]

            self.data = torch.FloatTensor(data[idx])
            self.target = torch.FloatTensor(target[idx])

        # normalize features
        self.data[:, 40:] = (self.data[:, 40:] - self.data[:, 40:].mean(dim=0,keepdim=True)) \
            / self.data[:, 40:].std(dim=0, keepdim=True)

        self.dim = self.data.shape[1]

        print('Finished reading the {} set of COVID19 Dataset ({} samples found, each dim = {})'
              .format(mode, len(self.data), self.dim))

    def __getitem__(self, item):
        if self.mode in ['train','dev']:
            return self.data[item], self.target[item]
        else:
            return self.data[item]

    def __len__(self):
        return len(self.data)

'''
DataLoader
'''
def COVID19Dataloader(path, mode, batch_size, n_jobs=0, target_only=False):
    dataset = COVID19Dataset(path, mode=mode, target_only = target_only)

    return DataLoader(dataset,batch_size,shuffle=(mode == 'train'),
                      drop_last=False, num_workers=n_jobs, pin_memory=True)
