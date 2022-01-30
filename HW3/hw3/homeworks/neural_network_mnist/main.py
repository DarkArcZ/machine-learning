# When taking sqrt for initialization you might want to use math package,
# since torch.sqrt requires a tensor, and math.sqrt is ok with integer
import math
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
import torchvision.datasets as datasets
from torchvision import transforms
from torch.distributions import Uniform
from torch.nn import Module
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils import load_dataset, problem

sns.set()

def prepare_dataset():

    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=256, shuffle=True)

    return train_loader, test_loader

class F1(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h: int, d: int, k: int):
        """Create a F1 model as described in pdf.

        Args:
            h (int): Hidden dimension.
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        self.alpha0 = 1/np.sqrt(d)
        self.alpha1 = 1/np.sqrt(h)
        self.w0 = torch.FloatTensor(d, h).uniform_(-self.alpha0, self.alpha0)
        self.b0 = torch.FloatTensor(1, h).uniform_(-self.alpha0, self.alpha0)
        self.w1 = torch.FloatTensor(h, k).uniform_(-self.alpha1, self.alpha1)
        self.b1 = torch.FloatTensor(1, k).uniform_(-self.alpha1, self.alpha1)

        self.params = [self.w0, self.b0, self.w1, self.b1]
        for param in self.params:
            param.requires_grad = True

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F1 model.

        It should perform operation:
        W_1(sigma(W_0*x + b_0)) + b_1

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: LongTensor of shape (n, k). Prediction.
        """
        res = torch.matmul(x, self.w0) + self.b0
        res = F.relu(res)
        res = torch.matmul(res, self.w1) + self.b1
        return res



class F2(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h0: int, h1: int, d: int, k: int):
        """Create a F2 model as described in pdf.

        Args:
            h0 (int): First hidden dimension (between first and second layer).
            h1 (int): Second hidden dimension (between second and third layer).
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        self.alpha0 = 1/np.sqrt(d)
        self.alpha1 = 1/np.sqrt(h1)
        self.w0 = torch.FloatTensor(d, h0).uniform_(-self.alpha0, self.alpha0)
        self.b0 = torch.FloatTensor(1, h0).uniform_(-self.alpha0, self.alpha0)
        self.w1 = torch.FloatTensor(h1, h1).uniform_(-self.alpha1, self.alpha1)
        self.b1 = torch.FloatTensor(1, h1).uniform_(-self.alpha1, self.alpha1)
        self.w2 = torch.FloatTensor(h1, k).uniform_(-self.alpha1, self.alpha1)
        self.b2 = torch.FloatTensor(1, k).uniform_(-self.alpha1, self.alpha1) 

        self.params = [self.w0, self.b0, self.w1, self.b1, self.w2, self.b2]
        for param in self.params:
            param.requires_grad = True

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F2 model.

        It should perform operation:
        W_2(sigma(W_1(sigma(W_0*x + b_0)) + b_1) + b_2)

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: LongTensor of shape (n, k). Prediction.
        """
        res = torch.matmul(x, self.w0) + self.b0
        res = F.relu(res)
        res = torch.matmul(res, self.w1) + self.b1
        res = F.relu(res)
        res = torch.matmul(res, self.w2) + self.b2
        return res


@problem.tag("hw3-A")
def train(model: Module, optimizer: optim.Adam, train_loader: DataLoader, test_loader: DataLoader, part) -> List[float]:
    """
    Train a model until it reaches 99% accuracy on train set, and return list of training crossentropy losses for each epochs.

    Args:
        model (Module): Model to train. Either F1, or F2 in this problem.
        optimizer (Adam): Optimizer that will adjust parameters of the model.
        train_loader (DataLoader): DataLoader with training data.
            You can iterate over it like a list, and it will produce tuples (x, y),
            where x is FloatTensor of shape (n, d) and y is LongTensor of shape (n,).

    Returns:
        List[float]: List containing average loss for each epoch.
    """
    losses = []
    acc = 0
    if part == "a":
        epochs = 32
    elif part == "b":
        epochs = 64
    for i in range(epochs):
        loss_epoch=0
        acc = 0
        for batch in tqdm(train_loader):
            images, labels = batch
            images, lavels = images, labels
            images = images.view(-1, 784)
            optimizer.zero_grad()
            logits = model.forward(images)
            preds = torch.argmax(logits, 1)
            acc += torch.sum(preds == labels).item()
            loss = F.cross_entropy(logits, labels)
            loss_epoch += loss.item()
            loss.backward()
            optimizer.step()
        print("Epoch ", i)
        print("Loss:", loss_epoch / len(train_loader.dataset))
        print("Acc:", acc / len(train_loader.dataset))
        losses.append(loss_epoch / len(train_loader.dataset))
        if acc / len(train_loader.dataset) > 0.99:
            break

    loss_epoch = 0
    acc = 0
    for batch in tqdm(test_loader):
        images, labels = batch
        images, labels = images, labels
        images = images.view(-1, 784)

        logits = model(images)
        preds = torch.argmax(logits, 1)
        acc += torch.sum(preds == labels).item()
        loss = F.cross_entropy(logits, labels)
        loss_epoch += loss.item()
    print('Testing dataset')
    print("Loss:", loss_epoch / len(test_loader))
    print("Acc:", acc / len(test_loader.dataset))
    return losses

@problem.tag("hw3-A", start_line=5)
def main():
    """
    Main function of this problem.
    For both F1 and F2 models it should:
        1. Train a model
        2. Plot per epoch losses
        3. Report accuracy and loss on test set
        4. Report total number of parameters for each network

    Note that we provided you with code that loads MNIST and changes x's and y's to correct type of tensors.
    We strongly advise that you use torch functionality such as datasets, but as mentioned in the pdf you cannot use anything from torch.nn other than what is imported here.
    """
    # (x, y), (x_test, y_test) = load_dataset("mnist")
    # x = torch.from_numpy(x).float()
    # y = torch.from_numpy(y).long()
    # x_test = torch.from_numpy(x_test).float()
    # y_test = torch.from_numpy(y_test).long()
    train_loader, test_loader = prepare_dataset()
    model1 = F1(h=64, d=784, k=10)
    optimizer = optim.Adam(model1.params, lr=5e-3)
    losses = train(model1, optimizer, train_loader=train_loader, test_loader=test_loader, part = "a")
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(losses)), losses, '--x', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('A4a.png')

    train_loader, test_loader = prepare_dataset()
    model2 = F2(h0=32, h1=32, d= 784, k=10)
    optimizer = optim.Adam(model2.params, lr=5e-3)
    losses = train(model2, optimizer, train_loader=train_loader, test_loader=test_loader, part="b")
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(losses)), losses, '--x', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('A4b.png')


if __name__ == "__main__":
    main()
