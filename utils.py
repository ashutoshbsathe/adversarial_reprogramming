import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data as data_utils 
import numpy as np 
import sys 
DATA_ROOT = './cifar10_data/'
def get_cifar10_data_loaders(batch_size=64, n_train=40000, n_val=10000,n_test=10000):
    train_transform = transforms.Compose([
        transforms.RandomCrop(size=32, padding=4), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.,0.,0.),(1.,1.,1.))
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.,0.,0.),(1.,1.,1.))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.,0.,0.),(1.,1.,1.))
    ])

    train_set = datasets.CIFAR10(root=DATA_ROOT, download=True, train=True, \
        transform=train_transform)
    val_set = datasets.CIFAR10(root=DATA_ROOT, download=True, train=True, \
        transform=val_transform)
    test_set = datasets.CIFAR10(root=DATA_ROOT, download=True, train=False, \
        transform=test_transform)

    # Generated as follows
    # indices = np.arange(0, 50000)
    # np.random.shuffle
    # np.save(...)
    indices = np.load(DATA_ROOT + '/CIFAR10_indices.npy')
    
    train_sampler = SubsetRandomSampler(indices[:n_train])
    val_sampler = SubsetRandomSampler(indices[n_train:])
    test_sampler = SubsetRandomSampler(np.arange(n_test))

    train_loader = data_utils.DataLoader(train_set, batch_size=batch_size,\
        sampler=train_sampler)
    val_loader = data_utils.DataLoader(val_set, batch_size=batch_size,\
        sampler=val_sampler)
    test_loader = data_utils.DataLoader(test_set, batch_size=batch_size,\
        sampler=test_sampler)
    return train_loader, val_loader, test_loader

DATA_ROOT_MNIST = './mnist_data/'
def get_mnist_data_loaders(batch_size=64, n_train=50000, n_val=10000,n_test=10000):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = datasets.MNIST(root=DATA_ROOT_MNIST, download=True, train=True, \
        transform=train_transform)
    val_set = datasets.MNIST(root=DATA_ROOT_MNIST, download=True, train=True, \
        transform=val_transform)
    test_set = datasets.MNIST(root=DATA_ROOT_MNIST, download=True, train=False, \
        transform=test_transform)

    # Generated as follows
    # indices = np.arange(0, 60000)
    # np.random.shuffle
    # np.save(...)
    indices = np.load(DATA_ROOT_MNIST + '/MNIST_indices.npy')
    
    train_sampler = SubsetRandomSampler(indices[:n_train])
    val_sampler = SubsetRandomSampler(indices[n_train:])
    test_sampler = SubsetRandomSampler(np.arange(n_test))

    train_loader = data_utils.DataLoader(train_set, batch_size=batch_size,\
        sampler=train_sampler)
    val_loader = data_utils.DataLoader(val_set, batch_size=batch_size,\
        sampler=val_sampler)
    test_loader = data_utils.DataLoader(test_set, batch_size=batch_size,\
        sampler=test_sampler)
    return train_loader, val_loader, test_loader

    
def progress(curr, total, suffix=''):
    bar_len = 48
    filled = int(round(bar_len * curr / float(total)))
    if filled == 0:
        filled = 1
    bar = '=' * (filled - 1) + '>' + '-' * (bar_len - filled)
    sys.stdout.write('\r[%s] .. %s' % (bar, suffix))
    sys.stdout.flush()
    if curr == total:
        bar = bar_len * '='
        sys.stdout.write('\r[%s] .. %s .. Completed\n' % (bar, suffix))