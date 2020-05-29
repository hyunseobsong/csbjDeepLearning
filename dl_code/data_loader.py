from abc import abstractmethod

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

import pickle

class IbMDataset(Dataset):
    """IbM dataset."""

    def __init__(self, ibm_data, transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = ibm_data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    @abstractmethod
    def get_sample(self, idx): raise NotImplementedError

    def __getitem__(self, idx):
        sample = self.get_sample(idx)
        if self.transform: sample = self.transform(sample)
        return sample

def get_train_test_split(pkl_file, test_split, random_seed=42):
    """
    Args:
        pkl_file (string): Path to the pkl file with parameters.
        test_split
    """
    with open(pkl_file,'rb') as f:
        data = pickle.load(f)

        data_size = len(data)
        indices = list(range(data_size))

        if test_split<1:
            split = int(np.floor(test_split * data_size))
        
            np.random.seed(random_seed)
            np.random.shuffle(indices)

            train_indices, test_indices = indices[split:], indices[:split]
        
            train_data = [data[idx] for idx in train_indices]
            test_data = [data[idx] for idx in test_indices]

            train_matrix = np.array([i['matrix'] for i in train_data])
            train_mean = train_matrix.mean()
            train_std = train_matrix.std()
            print("Data loaded, train_data:{}, test_data:{} ({:.2f}%)".format(
                data_size-split, split, 100*split/data_size))
            return train_data, test_data, train_mean, train_std
        elif test_split==1:
            test_data = [data[idx] for idx in indices]
            print("Data loaded, train_data:0, test_data:{} (100%)".format(data_size))
            return [], test_data, np.nan, np.nan
        else:
            print("[ERROR] Data loading failed. test_split should be in [0,1].")
            return [], [], np.nan, np.nan
    
    print("[ERROR] Data loading failed. Please check out pkl_file:", pkl_file)
    return [], [], np.nan, np.nan

def get_train_test_split_from_multiple_files(pkl_files, test_split, random_seed=42):
    """
    Args:
        pkl_file (string): Path to the pkl file with parameters.
        test_split
    """
    # do the random split once for the experiments
    indices = None
    train_data = []
    test_data = []
    train_mean = np.nan
    train_std = np.nan

    for pkl_file in pkl_files:
        with open(pkl_file,'rb') as f:
            data = pickle.load(f)

            data_size = len(data)

            # shuffle indices
            if indices is None:
                indices = list(range(data_size))
                np.random.seed(random_seed)
                np.random.shuffle(indices)

            if test_split<1:
                split = int(np.floor(test_split * data_size))
                train_indices, test_indices = indices[split:], indices[:split]
            
                train_data += [data[idx] for idx in train_indices]
                test_data += [data[idx] for idx in test_indices]
                print("[{}] Data loaded, train_data:{}, test_data:{} ({:.2f}%)".format(
                    pkl_file, data_size-split, split, 100*split/data_size))
            elif test_split==1:
                test_data += [data[idx] for idx in indices]
                print("[{}] Data loaded, train_data:0, test_data:{} (100%)".format(pkl_file, data_size))
            else:
                print("[ERROR] Data loading failed. test_split should be in [0,1].")

    if len(train_data)>0:            
        train_matrix = np.array([i['matrix'] for i in train_data])
        train_mean = train_matrix.mean()
        train_std = train_matrix.std()
    print("--"*50)
    print(" Data loaded, train_data:{}, test_data:{} ({:.2f}%), train_mean:{}, train_std:{}".format(
        len(train_data), len(test_data), 100*len(test_data)/(len(train_data)+len(test_data)),
        train_mean, train_std))
    return train_data, test_data, train_mean, train_std
    
class IbMDataset3D(IbMDataset):
    """IbM dataset."""

    def __init__(self, ibm_data, transform=None):
        super().__init__(ibm_data=ibm_data, transform=transform)

    def get_sample(self, idx):
        _, _, nframes = self.data[idx]['matrix'].shape
        sample = {
            # 'matrix': self.data[idx]['matrix'][:,:,np.linspace(0,nframes-1,5).astype(int)],
            'matrix': self.data[idx]['matrix'][:,:,np.linspace(0,nframes-1,10).astype(int)],
            'ruv': self.data[idx]['params']['ruv'],
            'rvu': self.data[idx]['params']['rvu'],
            'file': self.data[idx]['filename']
        }
        return sample

class IbMDataset2D(IbMDataset):
    """IbM dataset."""

    def __init__(self, ibm_data, transform=None):
        super().__init__(ibm_data=ibm_data, transform=transform)

    def get_sample(self, idx):
        sample = {'matrix': self.data[idx]['matrix'],
                  'ruv': self.data[idx]['params']['ruv'],
                  'rvu': self.data[idx]['params']['rvu'],
                  'file': self.data[idx]['filename']}
        return sample

class ToTensorMode1D(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        matrix, ruv, rvu = np.expand_dims(sample['matrix'], axis=2), sample['ruv'], sample['rvu']
        
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = matrix.transpose((2, 0, 1)).astype(np.float32)
        
        # interaction mode
        # bins[i-1] <= x < bins[i]
        # bins=[-1, -0.6, -0.2, 0.2, 0.6, 1]
        # bins=[-0.6, -0.2, 0.2, 0.6]
        bins=[-0.2, 0.2]
        ruv_bin_idx = np.digitize(ruv, bins)
        rvu_bin_idx = np.digitize(rvu, bins)
        interaction_mode = ruv_bin_idx*len(bins)+rvu_bin_idx
        return {'matrix': torch.from_numpy(image),
                'interaction': torch.tensor(interaction_mode)}

class ToTensorMode2D(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        matrix, ruv, rvu = np.expand_dims(sample['matrix'], axis=2), sample['ruv'], sample['rvu']
        
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = matrix.transpose((2, 0, 1)).astype(np.float32)
        
        # interaction mode
        # bins[i-1] <= x < bins[i]
        bins=[-0.2, 0.2]
        ruv_bin_idx = np.digitize(ruv, bins)
        rvu_bin_idx = np.digitize(rvu, bins)
        interaction_mode = np.array([ruv_bin_idx, rvu_bin_idx])
        return {'matrix': torch.from_numpy(image),
                'interaction': torch.tensor(interaction_mode)}

class ToTensor3D(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        matrix, ruv, rvu = sample['matrix'], sample['ruv'], sample['rvu']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = matrix.transpose((2, 0, 1)).astype(np.float32)
        image = np.expand_dims(image, axis=0)  # for Conv3D
        return {'matrix': torch.from_numpy(image),
                'interaction': torch.from_numpy(np.array([ruv, rvu]).astype(np.float32))}

class ToTensor2D(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        matrix, ruv, rvu = np.expand_dims(sample['matrix'], axis=2), sample['ruv'], sample['rvu']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = matrix.transpose((2, 0, 1)).astype(np.float32)
        return {'matrix': torch.from_numpy(image),
                'interaction': torch.from_numpy(np.array([ruv, rvu]).astype(np.float32))}

class Normalize(object):
    """Normalize."""
    def __init__(self, mean, stdv):
        self.mean = mean
        self.stdv = stdv

    def __call__(self, sample):
        matrix = (sample['matrix'] - self.mean) / self.stdv
        return {'matrix': matrix,
                'interaction': sample['interaction']}

def get_train_valid_split(ibm_data,
                          to_tensor,
                          valid_split,
                          batch_size,
                          valid_batch_size,
                          mean_for_normalize,
                          std_for_normalize,
                          k_fold_idx=0,
                          num_workers=4,
                          pin_memory=False,
                          use_cuda=False,
                          random_seed=42,
                          shuffle=False
                         ):
    '''
    Params
    ------
    - ibm_data: ibm data
    - valid_split: percentage split of the training set used for validation.
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to True if using GPU.
    - random_seed: fix seed for reproducibility.
    - k_fold_idx: K-Fold CV. this should be less than (1/valid_split)

    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    '''
    error_msg = "[ERROR] valid_split should be in the range [0, 1]."
    assert ((valid_split >= 0) and (valid_split <= 1)), error_msg

    normalize = Normalize(mean_for_normalize, std_for_normalize)
    transform=transforms.Compose([to_tensor(), normalize])
    dataset = IbMDataset2D(ibm_data=ibm_data, transform=transform)

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(valid_split * dataset_size))
    
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    val_indices = indices[k_fold_idx*split:(k_fold_idx+1)*split]
    train_indices = [i for i in indices if i not in val_indices]
    
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {'num_workers': num_workers, 'pin_memory': pin_memory}

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
        sampler=train_sampler, shuffle=shuffle, **kwargs)
    valid_loader = torch.utils.data.DataLoader(dataset, batch_size=valid_batch_size,
        sampler=valid_sampler, **kwargs)

    return dataset, train_loader, valid_loader

def get_test_loader(ibm_data,
                    to_tensor,
                    batch_size,
                    mean_for_normalize,
                    std_for_normalize,
                    num_workers=4,
                    pin_memory=False,
                    use_cuda=False,
                    shuffle=False
                    ):
    '''
    Params
    ------
    - ibm_data: ibm data
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to True if using GPU.
    
    Returns
    -------
    - test_loader: test set iterator.
    '''
    
    normalize = Normalize(mean_for_normalize, std_for_normalize)
    transform=transforms.Compose([to_tensor(), normalize])
    dataset = IbMDataset2D(ibm_data=ibm_data, transform=transform)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {'num_workers': num_workers, 'pin_memory': pin_memory}

    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
        shuffle=shuffle, **kwargs)

    return test_loader
    
def show_samples(ibm_dataset, num_samples=5, random_seed=None, fout=None):
    fig = plt.figure()

    dataset_size = len(ibm_dataset)
    indices = list(range(dataset_size))

    if random_seed:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    for idx in indices[:num_samples]:
        sample = ibm_dataset[idx]

        print(idx, sample['file'], sample['matrix'].shape, sample['ruv'], sample['rvu'])

        ax = plt.subplot(1, num_samples, idx + 1)
        ax.imshow(sample['matrix'])
        ax.axis('off')

    if fout: plt.savefig('test.png', dpi=300)
    else: plt.show()
    