if __name__ == "__main__":
    from layers import LinearLayer, ReLULayer, SigmoidLayer
    from losses import MSELossLayer
    from optimizers import SGDOptimizer
    from train import plot_model_guesses, train
else:
    from .layers import LinearLayer, ReLULayer, SigmoidLayer
    from .optimizers import SGDOptimizer
    from .losses import MSELossLayer
    from .train import plot_model_guesses, train


from typing import Any, Dict, Generator

import numpy as np
from homeworks.intro_pytorch.losses import MSE
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem

RNG = torch.Generator()
RNG.manual_seed(446)


@problem.tag("hw3-A")
def accuracy_score(model: nn.Module, dataloader: DataLoader) -> float:
    """Calculates accuracy of model on dataloader. Returns it as a fraction.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): Dataloader for MSE.
            Each example is a tuple consiting of (observation, target).
            Observation is a 2-d vector of floats.
            Target is also a 2-d vector of floats, but specifically with one being 1.0, while other is 0.0.
            Index of 1.0 in target corresponds to the true class.

    Returns:
        float: Vanilla python float resprenting accuracy of the model on given dataset/dataloader.
            In range [0, 1].

    Note:
        - For a single-element tensor you can use .item() to cast it to a float.
        - This is similar to CrossEntropy accuracy_score function,
            but there will be differences due to slightly different targets in dataloaders.
    """
    matches = 0
    count = 0
    with torch.no_grad():
        for x, y in dataloader:
            y_true = np.array(y)[:,1]
            prediction_values = model(x)
            y_predict = torch.argmax(prediction_values, dim=1).numpy()
            matches += np.sum(y_predict == y_true)
            count += y_predict.shape[0]
    accuracy = (matches/count)*100
    return accuracy



@problem.tag("hw3-A")
def mse_parameter_search(
    dataloader_train: DataLoader, dataloader_val: DataLoader
) -> Dict[str, Any]:
    """
    Main subroutine of the MSE problem.
    It's goal is to perform a search over hyperparameters, and return a dictionary containing training history of models, as well as models themselves.

    Models to check (please try them in this order):
        - Linear Regression Model
        - Network with one hidden layer of size 2 and sigmoid activation function after the hidden layer
        - Network with one hidden layer of size 2 and ReLU activation function after the hidden layer
        - Network with two hidden layers (each with size 2)
            and Sigmoid, ReLU activation function after corresponding hidden layers
        - Network with two hidden layers (each with size 2)
            and ReLU, Sigmoid activation function after corresponding hidden layers

    Args:
        dataloader_train (DataLoader): Dataloader for training dataset.
        dataloader_val (DataLoader): Dataloader for validation dataset.

    Returns:
        Dict[str, Any]: Dictionary/Map containing history of training of all models.
            You are free to employ any structure of this dictionary, but we suggest the following:
            {
                name_of_model: {
                    "train": Per epoch losses of model on train set,
                    "val": Per epoch losses of model on validation set,
                    "model": Actual PyTorch model (type: nn.Module),
                }
            }
    """
    model_1 = nn.Sequential(
        LinearLayer(2, 2, generator = RNG)
    )
    model_2 = nn.Sequential(
        LinearLayer(2, 2, generator = RNG),
        SigmoidLayer(),
        LinearLayer(2, 2, generator = RNG),
    )
    model_3 = nn.Sequential(
        LinearLayer(2, 2, generator = RNG),
        ReLULayer(),
        LinearLayer(2, 2, generator = RNG)
    )
    model_4 = nn.Sequential(
        LinearLayer(2, 2, generator = RNG),
        SigmoidLayer(),
        LinearLayer(2, 2, generator = RNG),
        ReLULayer(),
        LinearLayer(2, 2, generator = RNG),
    )
    model_5 = nn.Sequential(
        LinearLayer(2, 2, generator = RNG),
        ReLULayer(),
        LinearLayer(2, 2, generator = RNG),
        SigmoidLayer(),
        LinearLayer(2, 2, generator = RNG),
    )       
    dict = {"model_1": {"train": [], "val":[], "model": model_1},
             "model_2": {"train": [], "val":[], "model": model_2},
             "model_3": {"train": [], "val":[], "model": model_3},
             "model_4": {"train": [], "val":[], "model": model_4}, 
            "model_5": {"train": [], "val":[], "model": model_5}
    }     

    for model in dict.keys():
        md = dict[model]["model"]
        loss_fn = MSELossLayer()
        optimizer = SGDOptimizer(md.parameters(), lr=5e-3)
        res = train(train_loader=dataloader_train, model=md, criterion=loss_fn, optimizer=optimizer, val_loader=dataloader_val)
        dict[model]["train"] = res["train"]
        dict[model]["val"] = res["val"]
        plot_model_guesses(dataloader_train, md)
    return dict

@problem.tag("hw3-A", start_line=21)
def main():
    """
    Main function of the MSE problem.
    It should:
        1. Call mse_parameter_search routine and get dictionary for each model architecture/configuration.
        2. Plot Train and Validation losses for each model all on single plot (it should be 10 lines total).
            x-axis should be epochs, y-axis should me MSE loss, REMEMBER to add legend
        3. Choose and report the best model configuration based on validation losses.
            In particular you should choose a model that achieved the lowest validation loss at ANY point during the training.
        4. Plot best model guesses on test set (using plot_model_guesses function from train file)
        5. Report accuracy of the model on test set.

    Starter code loads dataset, converts it into PyTorch Datasets, and those into DataLoaders.
    You should use these dataloaders, for the best experience with PyTorch.
    """
    (x, y), (x_val, y_val), (x_test, y_test) = load_dataset("xor")

    mse_dataloader_train = DataLoader(
        TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(to_one_hot(y))),
        batch_size=32,
        shuffle=True,
        generator=RNG,
    )
    mse_dataloader_val = DataLoader(
        TensorDataset(
            torch.from_numpy(x_val).float(), torch.from_numpy(to_one_hot(y_val))
        ),
        batch_size=32,
        shuffle=False,
    )
    mse_dataloader_test = DataLoader(
        TensorDataset(
            torch.from_numpy(x_test).float(), torch.from_numpy(to_one_hot(y_test))
        ),
        batch_size=32,
        shuffle=False,
    )
    mse_configs = mse_parameter_search(mse_dataloader_train, mse_dataloader_val)
    for model_res in mse_configs.keys():
        acc = accuracy_score(mse_configs[model_res]["model"], mse_dataloader_test)
        print("The testing accuracy for", model_res, "is:", acc)

        plt.plot(mse_configs[model_res]["train"], label=model_res + "_train")
        plt.plot(mse_configs[model_res]["val"], label=model_res + "_val")

    plt.title("Losses for MSE Search")
    plt.xlabel("Epoches")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def to_one_hot(a: np.ndarray) -> np.ndarray:
    """Helper function. Converts data from categorical to one-hot encoded.

    Args:
        a (np.ndarray): Input array of integers with shape (n,).

    Returns:
        np.ndarray: Array with shape (n, c), where c is maximal element of a.
            Each element of a, has a corresponding one-hot encoded vector of length c.
    """
    r = np.zeros((len(a), 2))
    r[np.arange(len(a)), a] = 1
    return r


if __name__ == "__main__":
    main()
