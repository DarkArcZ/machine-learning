import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import torchvision.datasets as datasets
from torchvision import transforms
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from utils import problem


def load_dataset():
    mnist_trainset = datasets.MNIST(root='./data/', train=True, download=True, transform=transforms.ToTensor())
    mnist_testset = datasets.MNIST(root='./data/', train=False, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=16, shuffle=False)

    mnist_visset = datasets.MNIST(root='./data/', train=False, download=True, transform=transforms.ToTensor())
    current_digit = 0
    mnist_visset.data = []
    mnist_visset.targets = []
    for data, label in zip(mnist_trainset.data, mnist_trainset.targets):
        if label == current_digit:
            mnist_visset.data.append(data)
            mnist_visset.targets.append(label)
            current_digit += 1
        if label >= 10:
            break

    vis_loader = torch.utils.data.DataLoader(mnist_visset, batch_size=10, shuffle=False)

    return train_loader, test_loader, vis_loader

@problem.tag("hw4-A")
def F1(h: int) -> nn.Module:
    """Model F1, it should performs an operation W_d * W_e * x as written in spec.

    Note:
        - While bias is not mentioned explicitly in equations above, it should be used.
            It is used by default in nn.Linear which you can use in this problem.

    Args:
        h (int): Dimensionality of the encoding (the hidden layer).

    Returns:
        nn.Module: An initialized autoencoder model that matches spec with specific h.
    """
    input_size=784
    f1 = nn.Sequential(
        nn.Linear(input_size, h),
        nn.Linear(h, input_size)
    ) 
    return f1


@problem.tag("hw4-A")
def F2(h: int) -> nn.Module:
    """Model F1, it should performs an operation ReLU(W_d * ReLU(W_e * x)) as written in spec.

    Note:
        - While bias is not mentioned explicitly in equations above, it should be used.
            It is used by default in nn.Linear which you can use in this problem.

    Args:
        h (int): Dimensionality of the encoding (the hidden layer).

    Returns:
        nn.Module: An initialized autoencoder model that matches spec with specific h.
    """
    input_size=784
    f2 = nn.Sequential(
        nn.Linear(input_size, h),
        nn.ReLU(),
        nn.Linear(h, input_size),
        nn.ReLU()
    )
    return f2

@problem.tag("hw4-A")
def train(
    model: nn.Module, optimizer: Adam, train_loader: DataLoader, epochs: int = 15
) -> float:
    """
    Train a model until convergence on train set, and return a mean squared error loss on the last epoch.

    Args:
        model (Module): Model to train. Either F1, or F2 in this problem.
        optimizer (Adam): Optimizer that will adjust parameters of the model.
            Hint: You can try using learning rate of 5e-5.
        train_loader (DataLoader): DataLoader with training data.
            You can iterate over it like a list, and it will produce x
            where x is FloatTensor of shape (n, d).

    Note:
        - Unfortunately due to how DataLoader class is implemented in PyTorch
            "for x_batch in train_loader:" will not work. Use:
            "for (x_batch,) in train_loader:" instead.

    Returns:
        float: Final training error/loss
    """
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        loss = 0
        for i, (data, _) in enumerate(train_loader):
            data = data.view(-1, 28*28).to(DEVICE)
            optimizer.zero_grad()

            outputs = model(data)

            train_loss = criterion(outputs, data)
            train_loss.backward()

            optimizer.step()

            loss += train_loss.item()

        loss /= len(train_loader)
        if epoch + 1 == epochs:
            print("Epoch : {}/{}, Train loss = {:.6f}".format(epoch + 1, epochs, loss))


@problem.tag("hw4-A")
def evaluate(model: nn.Module, loader: DataLoader) -> float:
    """Evaluates a model on a provided dataset.
    It should return an average loss of that dataset.

    Args:
        model (Module): TRAINED Model to evaluate. Either F1, or F2 in this problem.
        loader (DataLoader): DataLoader with some data.
            You can iterate over it like a list, and it will produce x
            where x is FloatTensor of shape (n, d).

    Returns:
        float: Mean Squared Error on the provided dataset.
    """
    loss = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for i, (data, _) in enumerate(loader):

            data = data.view(-1, 28*28).to(DEVICE)
            outputs = model(data)
            test_loss = criterion(outputs, data)
            loss += test_loss.item()

        loss /= len(loader)
        print("Test loss = {:.6f}".format(loss))


def plot_original_and_reconstruction(data_loader, model, save_path):

    for data, labels in data_loader:
        data = data.view(-1, 28*28).to(DEVICE)
        with torch.no_grad():
            original = data
            reconstruct = model(data)

    fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(10, 3))
    for i, row in enumerate(axes):
        for j, cell in enumerate(row):
            if i == 0:
                cell.imshow(original.cpu().data[j, :].reshape((28, 28)))
                cell.set_title(j)
                if j == 0:
                    cell.set_ylabel('Original')
                cell.set(xticklabels=[])
                cell.set(yticklabels=[])
            else:
                cell.imshow(reconstruct.cpu().data[j, :].reshape((28, 28)))
                if j == 0:
                    cell.set_ylabel('Reconstructed')
    plt.tight_layout()
    plt.savefig(save_path)

def main_A6a(ks=[32, 64, 128]):

    train_loader, test_loader, vis_loader = load_dataset()

    for k in ks:
        model = F1(h=k).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(model, optimizer, train_loader)
        plot_original_and_reconstruction(vis_loader, model, 'A6a_{}.png'.format(k))

        evaluate(model, test_loader)

    return model

def main_A6b(ks=[32, 64, 128]):

    train_loader, test_loader, vis_loader = load_dataset()

    for k in ks:
        model = F2(h=k).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(model, optimizer, train_loader)
        plot_original_and_reconstruction(vis_loader, model, 'A6b_{}.png'.format(k))

        evaluate(model, test_loader)

    return model

@problem.tag("hw4-A", start_line=9)
def main():
    """
    Main function of autoencoders problem.

    It should:
        A. Train an F1 model with hs 32, 64, 128, report loss of the last epoch
            and visualize reconstructions of 10 images side-by-side with original images.
        B. Same as A, but with F2 model
        C. Use models from parts A and B with h=128, and report reconstruction error (MSE) on test set.

    Note:
        - For visualizing images feel free to use images_to_visualize variable.
            It is a FloatTensor of shape (10, 784).
        - For having multiple axes on a single plot you can use plt.subplots function
        - For visualizing an image you can use plt.imshow (or ax.imshow if ax is an axis)
    """
    # (x_train, y_train), (x_test, _) = load_dataset("mnist")
    # x = torch.from_numpy(x_train).float()
    # x_test = torch.from_numpy(x_test).float()

    # # Neat little line that gives you one image per digit for visualization in parts a and b
    # images_to_visualize = x[[np.argwhere(y_train == i)[0][0] for i in range(10)]]

    # train_loader = DataLoader(TensorDataset(x), batch_size=32, shuffle=True)
    # test_loader = DataLoader(TensorDataset(x_test), batch_size=32, shuffle=True)
    # raise NotImplementedError("Your Code Goes Here")
    model_a = main_A6a()
    model_b = main_A6b()

if __name__ == "__main__":
    main()
