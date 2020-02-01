# Adversarial Reprogramming

Code for PyTorch implementation of [adversarial reprogramming](https://arxiv.org/abs/1806.11146).

### Requirements :
1. PyTorch 0.4+
2. Python 3.6

### Instructions :
Train a ResNet18 on CIFAR10 using following code 
```
python train.py
```
Then reprogram the ResNet18 to classify MNIST digits
```
python adv_reprogram.py
```

<!-- End of README -->
