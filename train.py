import torch 
import torch.nn as nn
import torchvision
from models import ResNet18
import utils
import os 

SAVE_DIR = './saved_models/resnet18/'
MODEL_NAME = 'resnet18_at_epoch_{}.pt'
LOSS_NAME = 'loss_log.csv'

N_TRAIN = 40000
N_VAL   = 10000
N_TEST  = 10000

def update_learning_rate(optim, old_lr, new_lr, print_msg=True):
    if print_msg:
        print('\nReducing learning rate from {} to {}'.format(old_lr, new_lr))
    for g in optim.param_groups:
        g['lr'] = new_lr
    return optim 

def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using PyTorch device : {}'.format(device.upper()))

    model = ResNet18().to(device)
    train_loader, val_loader, test_loader = utils.get_cifar10_data_loaders(n_train=N_TRAIN, \
        n_val=N_VAL, n_test=N_TEST)
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
            images, labels = images.to(device),labels.to(device)

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
    print('Training Complete')
    model.eval()
    correct = 0
    for j, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device) 
        logits = model(images)
        correct += (torch.max(logits, 1)[-1] == labels).sum().item()
        utils.progress(j+1, len(test_loader), 'Batch [{}/{}]'.format(j+1, len(test_loader)))
    print('Test Accuracy = {}%'.format(float(correct) * 100.0/N_TEST))
    print('Done')


if __name__ == '__main__':
    main()