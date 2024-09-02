import torch
import numpy as np

def kFold(K, data, IDs, train_ratio):
    un_IDs = np.unique(IDs)

    total_samples = un_IDs.shape[0]
    train_size = int(train_ratio * total_samples)
    test_size = total_samples - train_size

    trainval_map, test_map = torch.utils.data.random_split(torch.arange(total_samples), [train_size, test_size])

    folds = torch.utils.data.random_split(torch.arange(total_samples)[trainval_map], [train_size/K]*K)
    
    return folds, test_map

def get_fold_k(k, IDs, folds, test_map):
    un_IDs = np.unique(IDs)
    test_map = np.argwhere(np.isin(IDs, un_IDs[test_map.indices])).squeeze().tolist()
    train_map = []
    for fold, i in enumerate(folds):
        if i == k:
            val_map = np.argwhere(np.isin(IDs, un_IDs[fold.indices])).squeeze().tolist()
        else:
            train_map.append(np.argwhere(np.isin(IDs, un_IDs[fold.indices])).squeeze().tolist())
    train_map = np.concatenate(train_map)
    return train_map, val_map, test_map
