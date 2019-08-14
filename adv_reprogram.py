import torch 
import torch.nn as nn 
import torchvision
import pprint
from models import ResNet18
import utils 
from collections import OrderedDict
import numpy as np
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
        pprint.pprint(self.weight_matrix)
        self.pad = nn.ConstantPad2d(int(self.input_size/2) - int(self.adv_input_size/2), 0)

    def forward(self, x, visualize=False):
        resize = nn.UpsamplingBilinear2d(size=(self.adv_input_size, self.adv_input_size))
        x = x.repeat(1, 3, 1, 1).to('cpu')
        if visualize:
            torchvision.utils.save_image(x, 'repeated_image.png')
        x_new = nn.functional.sigmoid(self.reprogram_weights * self.weight_matrix) + \
            self.pad(resize(x))
        if visualize:
            torchvision.utils.save_image(x_new.cpu(), 'forward_pass.png')
        y = self.model(x_new.to(self.device))
        return y
    
    def visualize_adversarial_program(self):
        torchvision.utils.save_image(self.weight_matrix * self.reprogram_weights, 'reprogram_weights.png')

def main():
    model = ResNet18().to('cuda')
    model_weights = torch.load('./resnet18_at_epoch_49.pt').copy()
    all_keys = model_weights.items()
    valid_weights = OrderedDict()
    print(type(all_keys))
    for i, (k,v) in enumerate(all_keys):
        if 'num_batches_tracked' in k:
            print('Found num_batches_tracked')
        else:
            valid_weights[k] = v 
    model.load_state_dict(valid_weights)
    model.eval()
    """
    _, _, test_loader = utils.get_cifar10_data_loaders()
    correct = 0
    for i, (images, labels) in enumerate(test_loader):
        images, labels = images.to('cuda'), labels.to('cuda')
        logits = model(images)
        correct += (torch.max(logits, 1)[-1] == labels).sum().item()
        utils.progress(i+1, len(test_loader), 'Batch [{}/{}]'.format(i+1, len(test_loader)))
    print('Accuracy = {}%'.format(float(correct) * 100.0/10000))
    """
    reprogrammed = ReProgramCIFAR10ToMNIST(model)
    save_tensor = reprogrammed.weight_matrix * reprogrammed.reprogram_weights
    torchvision.utils.save_image(save_tensor.view(1, 3, 32, 32), 'reprogram_init.png')
    train_loader, val_loader, test_loader = utils.get_mnist_data_loaders()
    optim = torch.optim.SGD([reprogrammed.reprogram_weights], lr=1e-1, momentum=0.9)
    xent = nn.CrossEntropyLoss()
    n_epochs = 8
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
    correct = 0
    reprogrammed.eval()
    for i, (images, labels) in enumerate(test_loader):
        images, labels = images.to('cuda'), labels.to('cuda')
        logits = reprogrammed(images)
        logits = logits.to('cuda')
        correct += (torch.max(logits, 1)[-1], labels).sum().item()
        utils.progress(i+1, len(test_loader), 'Batch [{}/{}]'.format(i+1, len(test_loader)))
    print('Accuracy = {}%'.format(float(correct) * 100.0/10000))


if __name__ == '__main__':
    main()

