import numpy as np 


def gaussian(f, mean=0.0, std=1.0, add=False):
    if add:
        return np.random.normal(mean, std, size=f.shape) + f
    else:
        return np.random.normal(mean, std, size=f.shape)


def gaussian_mimic(f, add=False):
    gaussian_feat = np.ndarray(shape=f.shape)
    for i, v in enumerate(f):
        mean = np.mean(v)
        std = np.std(v)
        if add:
            gaussian_feat[i,:] = v + np.random.normal(mean, std, size=v.shape)
        else:
            gaussian_feat[i,:] = np.random.normal(mean, std, size=v.shape)
    return gaussian_feat


def superimpose_gaussian(f, k=1.5):
    """
    Extend the feature vector by k-times and paste the true feature vector onto
    the middle. Other locations are filled with gaussian noise of similar mean 
    and std.
    """
    new_dim = int(np.ceil(f.shape[1] * k))
    offset = int(np.floor((new_dim - f.shape[1])/2))
    mean = np.mean(f)
    std = np.std(f)
    new_feat = np.random.normal(mean, std, size=(f.shape[0], new_dim))
    for i, v in enumerate(f):
        new_feat[i,offset:offset+len(v)] = v
    return new_feat


def superimpose_gaussian_class(f, labels):
    """
    Superimpose the features by labels. The feature vectors will be extended by
    number of label times. Labels are expected to start from 0 and continously
    increase by 1.
    """
    assert np.min(labels) == 0
    num_labels = np.max(labels) + 1
    org_dim = f.shape[1]
    new_dim = int(np.ceil(org_dim*num_labels))
    mean = np.mean(f)
    std = np.std(f)
    new_feat = np.random.normal(mean, std, size=(f.shape[0], new_dim))
    for i, (v, l) in enumerate(zip(f, labels)):
        new_feat[i,l*org_dim:(l+1)*org_dim] = v
    return new_feat


def superimpose_gaussian_random(f, k=1.5):
    """
    Superimpose the features by random offset.
    """
    new_dim = int(np.ceil(f.shape[1] * k))
    offset_range = (0, new_dim-f.shape[1])
    mean = np.mean(f)
    std = np.std(f)
    new_feat = np.random.normal(mean, std, size=(f.shape[0], new_dim))
    for i, v in enumerate(f):
        offset = np.random.randint(*offset_range)
        new_feat[i,offset:offset+len(v)] = v
    return new_feat


def zero_idx(f, idx):
    """
    Set vector of idx to all zeros
    """
    dim = f.shape[1]
    f[idx,:] = np.zeros((dim,))
    return f

def gaussian_idx(f, idx):
    """
    Set vector of idx to similar mean and std gaussian
    """
    