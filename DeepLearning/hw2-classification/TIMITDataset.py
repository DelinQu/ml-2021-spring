import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class TIMITDataset(Dataset):
    def __init__(self, mode='train', VAL_RATIO = 0.1):
        self.mode, self.VAL_RATIO = mode, VAL_RATIO
        # load data
        data_root = './timit_11/'
        if self.mode == 'test':
            data = np.load(data_root + 'test_11.npy').astype(float)
            self.data = torch.FloatTensor(data)
        else:
            data, label = np.load(data_root + 'train_11.npy').astype(float), np.load(data_root + 'train_label_11.npy').astype(int)

            CHUNK = int(100*VAL_RATIO)
            if mode == 'train':
                idx = [i for i in range(len(data)) if i % CHUNK != 0]
            elif mode == 'val':
                idx = [i for i in range(len(data)) if i % CHUNK == 0]

            self.data, self.label = torch.FloatTensor(data[idx,:]), torch.LongTensor(label[idx])

        self.dim = self.data.shape[1]
        print('Finished reading the {} set of TIMIT Dataset ({} samples found, each dim = {})'
              .format(mode, len(self.data), self.dim))

    def __getitem__(self, idx):
        if self.mode != 'test':
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)


def TIMITDataLoader(mode, batch_size, n_jobs = 0, VAL_RATIO = 0.1):
    dataset = TIMITDataset(mode=mode, VAL_RATIO=VAL_RATIO)
    return DataLoader(dataset, batch_size, shuffle=(mode == 'train'),
                      drop_last=False, num_workers=n_jobs, pin_memory=True)