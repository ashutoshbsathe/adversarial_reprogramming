import torch 
import torch.nn as nn 
import torchvision
import pprint
from models import ResNet18
import utils 
from collections import OrderedDict
import numpy as np
import os 
from datetime import datetime

# This file uses a lot of hard-coded values unlike the `train.py`
# TODO: Remove most of the hard-coded values and make them configurable

SAVE_DIR = './saved_models/reprogram/'
MODEL_LOAD_PATH = './saved_models/resnet18/resnet_at_epoch_49.pt'

class ReProgramCIFAR10ToMNIST(nn.Module):
    def __init__(self, model, input_size=32, adv_input_size=14, device='cuda'):
        super(ReProgramCIFAR10ToMNIST, self).__init__()
        assert model.training is not None
        self.model = model
        self.input_size = input_size
        self.adv_input_size = adv_input_size
        self.device = device
        self.reprogram_weights = nn.Parameter(torch.randn(3, input_size, input_size))
        self.weight_matrix = torch.ones(3, input_size, input_size)
        lower_limit = int(input_size/2) - int(adv_input_size/2)
        higher_limit = int(input_size/2) + int(adv_input_size/2)
        self.weight_matrix[:, lower_limit:higher_limit, lower_limit:higher_limit] = \
            torch.zeros(3, adv_input_size, adv_input_size)
        self.pad = nn.ConstantPad2d(int(self.input_size/2) - int(self.adv_input_size/2), 0)
        self.resize = nn.UpsamplingBilinear2d(size=(self.adv_input_size, self.adv_input_size))

    def forward(self, x):
        # resize = nn.UpsamplingBilinear2d(size=(self.adv_input_size, self.adv_input_size))
        """
        A hack to get 3 channeled image for MNIST
        """
        x = x.repeat(1, 3, 1, 1).to('cpu')
        """
        Since I have trained the CIFAR10 model with data in the range[0, 1],
        I will be using sigmoid instead of tanh as described in the paper
        """
        x_new = nn.functional.sigmoid(self.reprogram_weights * self.weight_matrix) + \
            self.pad(self.resize(x))
        y = self.model(x_new.to(self.device))
        return y
    
    def visualize(self, x):
        time_str = datetime.now().strftime('%Y%m%d_%H:%M:%S')
        torchvision.utils.save_image(self.weight_matrix * self.reprogram_weights, \
            SAVE_DIR + 'reprogram_weights_{}.png'.format(time_str))
        x = x.repeat(1, 3, 1, 1).to('cpu')
        x_new = nn.functional.sigmoid(self.reprogram_weights * self.weight_matrix) + \
            self.pad(self.resize(x))
        torchvision.utils.save_image(x_new, \
            SAVE_DIR + 'adversarial_input_{}.png'.format(time_str))

def main():
    model = ResNet18().to('cuda')
    model_weights = torch.load(MODEL_LOAD_PATH).copy()
    """
    Uncomment following block if you trained a network on newer versions of PyTorch 
    and are now copying the .pt file to your old machine
    all_keys = model_weights.items()
    valid_weights = OrderedDict()
    print(type(all_keys))
    for i, (k,v) in enumerate(all_keys):
        if 'num_batches_tracked' in k:
            print('Found num_batches_tracked')
        else:
            valid_weights[k] = v 
    """
    model.load_state_dict(model_weights)
    model.eval()
    _, _, test_loader = utils.get_cifar10_data_loaders()
    correct = 0
    for i, (images, labels) in enumerate(test_loader):
        images, labels = images.to('cuda'), labels.to('cuda')
        logits = model(images)
        correct += (torch.max(logits, 1)[-1] == labels).sum().item()
        utils.progress(i+1, len(test_loader), 'Batch [{}/{}]'.format(i+1, len(test_loader)))
    print('Accuracy on test set of CIFAR10 = {}%'.format(float(correct) * 100.0/10000))
    reprogrammed = ReProgramCIFAR10ToMNIST(model)
    save_tensor = reprogrammed.weight_matrix * reprogrammed.reprogram_weights
    torchvision.utils.save_image(save_tensor.view(1, 3, 32, 32), SAVE_DIR + 'reprogram_init.png')
    train_loader, val_loader, test_loader = utils.get_mnist_data_loaders()
    """
    These parameters seem to be working best
    Feel free to play around these values
    """
    optim = torch.optim.SGD([reprogrammed.reprogram_weights], lr=1e-1, momentum=0.9)
    xent = nn.CrossEntropyLoss()
    n_epochs = 64
    for epoch in range(n_epochs):
        print('Epoch {}'.format(epoch+1))
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to('cuda'), labels.to('cuda')
            logits = reprogrammed(images)
            logits = logits.to('cuda')
            optim.zero_grad()
            loss = xent(logits, labels)
            loss.backward()
            optim.step()
            reprogrammed.visualize_adversarial_program()
            utils.progress(i+1, len(train_loader), 'Batch [{}/{}] Loss = {} Batch Acc = {}%'.format(i+1, len(train_loader), loss.item(),\
                ((torch.max(logits, 1)[-1] == labels).sum().item() * 100.0/images.size(0))))
        reprogrammed.visualize(images)
    correct = 0
    reprogrammed.eval()
    for i, (images, labels) in enumerate(test_loader):
        images, labels = images.to('cuda'), labels.to('cuda')
        logits = reprogrammed(images)
        logits = logits.to('cuda')
        correct += (torch.max(logits, 1)[-1], labels).sum().item()
        utils.progress(i+1, len(test_loader), 'Batch [{}/{}]'.format(i+1, len(test_loader)))
    print('Accuracy on MNIST test set = {}%'.format(float(correct) * 100.0/10000))
    print('Done')



if __name__ == '__main__':
    main()

