import torch 
import torch.nn as nn
import torchvision
from models import ResNet18
import utils

SAVE_DIR = './saved_models/resnet18/'
MODEL_NAME = 'resnet18_at_epoch_{}.pt'
LOSS_NAME = 'loss_log.csv'

ef update_learning_rate(optim, old_lr, new_lr, print_msg=True):
    if print_msg:
        print('\nReducing learning rate from {} to {}'.format(old_lr, new_lr))
    for g in optim.param_groups:
        g['lr'] = new_lr
    return optim 

def main():
    model = ResNet18().cuda()
    train_loader, val_loader, test_loader = utils.get_cifar10_data_loaders()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)
    xent = nn.CrossEntropyLoss()
    step = 0
    n_epochs = 50
    f = open(SAVE_DIR + LOSS_NAME, 'w')
    f.truncate(0)
    f.write('train_step, train_loss\n')
    f.close()
    MODEL_SAVE_PATH = SAVE_DIR + MODEL_NAME
    assert model.training is True
    for i in range(n_epochs):
        for j, (images, labels) in enumerate(train_loader):
            images, labels = images.cuda(),labels.cuda()

            optimizer.zero_grad()
            outputs = model(images)
            loss = xent(outputs, labels)
            loss.backward()
            optimizer.step()
            if step % 16 == 0:
                with open(SAVE_DIR + LOSS_NAME, 'a') as f:
                    f.write('{}, {}\n'.format(step, loss.item()))
            step += 1
            utils.progress(j+1, len(train_loader), 'Batch [{}/{}] Epoch [{}/{}]'.format(j+1,len(train_loader),i+1,n_epochs))
        torch.save(model.state_dict(), MODEL_SAVE_PATH.format(i))
    print('Done')

if __name__ == '__main__':
    main()