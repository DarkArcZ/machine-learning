from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem

# When choosing your batches / Shuffling your data you should use this RNG variable, and not `np.random.choice` etc.
RNG = np.random.RandomState(seed=446)
Dataset = Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]


def load_2_7_mnist() -> Dataset:
    """
    Loads MNIST data, extracts only examples with 2, 7 as labels, and converts them into -1, 1 labels, respectively.

    Returns:
        Dataset: 2 tuples of numpy arrays, each containing examples and labels.
            First tuple is for training, while second is for testing.
            Shapes as as follows: ((n, d), (n,)), ((m, d), (m,))
    """
    (x_train, y_train), (x_test, y_test) = load_dataset("mnist")
    train_idxs = np.logical_or(y_train == 2, y_train == 7)
    test_idxs = np.logical_or(y_test == 2, y_test == 7)

    y_train_2_7 = y_train[train_idxs]
    y_train_2_7 = np.where(y_train_2_7 == 7, 1, -1)

    y_test_2_7 = y_test[test_idxs]
    y_test_2_7 = np.where(y_test_2_7 == 7, 1, -1)

    return (x_train[train_idxs], y_train_2_7), (x_test[test_idxs], y_test_2_7)


class BinaryLogReg:
    @problem.tag("hw2-A", start_line=3)
    def __init__(self, _lambda: float = 1e-3):
        """Initializes the Binary Log Regression model.
        NOTE: Please DO NOT change `self.weight` and `self.bias` values, since it may break testing and lead to lost points!

        Args:
            _lambda (float, optional): Ridge Regularization coefficient. Defaults to 1e-3.
        """
        self._lambda: float = _lambda
        # Fill in with matrix with the correct shape
        self.weight: np.ndarray = None  # type: ignore
        self.bias: np.ndarray = None

    @problem.tag("hw2-A")
    def mu(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        exponential = np.exp(-y*self.bias - y*np.dot(X, self.weight))
        return 1/(1+exponential)

    @problem.tag("hw2-A")
    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.mean(np.log(1 + np.exp(-y * (self.bias + X.dot(self.weight))))) + \
               self._lambda * self.weight.dot(self.weight)

    @problem.tag("hw2-A")
    def gradient_J_weight(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.mean(((self.mu(X, y) - 1)*y)[:,None]*X, axis = 0) +2*self._lambda*self.weight
        
    @problem.tag("hw2-A")
    def gradient_J_bias(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.mean((self.mu(X, y)-1)*y, axis=0)

    @problem.tag("hw2-A")
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.sign(self.bias + X.dot(self.weight))


    @problem.tag("hw2-A")
    def misclassification_error(self, X: np.ndarray, y: np.ndarray) -> float:
        X_train_pred = np.dot(X, self.weight)+self.bias
        X_train_pred[X_train_pred <= 0] = -1
        X_train_pred[X_train_pred > 0] = 1
        return 1 - (X_train_pred == y).sum()/float(X.shape[0])

    @problem.tag("hw2-A")
    def step(self, X: np.ndarray, y: np.ndarray, learning_rate: float = 1e-4):
        """Single step in training loop.
        It does not return anything but should update self.weight and self.bias with correct values.

        Args:
            X (np.ndarray): observations represented as `(n, d)` matrix.
                n is number of observations, d is number of features.
                d = 784 in case of MNIST.
            y (np.ndarray): targets represented as `(n, )` vector.
                n is number of observations.
            learning_rate (float, optional): Learning rate of SGD/GD algorithm.
                Defaults to 1e-4.
        """
        self.weight = self.weight - learning_rate*self.gradient_J_weight(X, y)
        self.bias = self.bias - learning_rate*self.gradient_J_bias(X, y)
            

    @problem.tag("hw2-A", start_line=7)
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        sgd: bool = False,
        learning_rate: float = 1e-2,
        epochs: int = 30,
        batch_size: int = 100,
    ) -> Dict[str, List[float]]:
        """Train function that given dataset X_train and y_train adjusts weights and biases of this model.
        It also should calculate misclassification error and J loss at the END of each epoch.

        For each epoch please call step function `num_batches` times as defined on top of the starter code.

        NOTE: This function due to complexity and number of possible implementations will not be publicly unit tested.
        However, we might still test it using gradescope, and you will be graded based on the plots that are generated using this function.

        Args:
            X_train (np.ndarray): observations in training set represented as `(n, d)` matrix.
                n is number of observations, d is number of features.
                d = 784 in case of MNIST.
            y_train (np.ndarray): targets in training set represented as `(n, )` vector.
                n is number of observations.
            X_test (np.ndarray): observations in testing set represented as `(m, d)` matrix.
                m is number of observations, d is number of features.
                d = 784 in case of MNIST.
            y_test (np.ndarray): targets in testing set represented as `(m, )` vector.
                m is number of observations.
            learning_rate (float, optional): Learning rate of SGD/GD algorithm. Defaults to 1e-2.
            epochs (int, optional): Number of epochs (loops through the whole data) to train SGD/GD algorithm for.
                Defaults to 30.
            batch_size (int, optional): Number of observation/target pairs to use for a single update.
                Defaults to 100.

        Returns:
            Dict[str, List[float]]: Dictionary containing 4 keys, each pointing to a list/numpy array of length `epochs`:
            {
                "training_losses": [<Loss at the end of each epoch on training set>],
                "training_errors": [<Misclassification error at the end of each epoch on training set>],
                "testing_losses": [<Same as above but for testing set>],
                "testing_errors": [<Same as above but for testing set>],
            }
            Skeleton for this result is provided in the starter code.

        Note:
            - When shuffling batches/randomly choosing batches makes sure you are using RNG variable defined on the top of the file.
        """
        num_batches = int(np.ceil(len(X_train) // batch_size))
        result: Dict[str, List[float]] = {
            "train_losses": [],  # You should append to these lists
            "train_errors": [],
            "test_losses": [],
            "test_errors": [],
        }

        _, d = X_train.shape
        self.weight = np.zeros(d)
        self.bias = 0
           
        for _ in range(epochs):
            if sgd:
                inds_train = RNG.permutation(len(X_train))[:batch_size]
                X_train_step = X_train[inds_train]
                y_train_pred_step = y_train[inds_train]
                self.step(X_train_step, y_train_pred_step, learning_rate)
            else:
                self.step(X_train, y_train, learning_rate)
        
            train_loss = self.loss(X_train, y_train)
            train_error = self.misclassification_error(X_train, y_train)

            test_loss = self.loss(X_test, y_test)
            test_error = self.misclassification_error(X_test, y_test)

            result["train_losses"].append(train_loss)
            result["train_errors"].append(train_error)
            result["test_losses"].append(test_loss)
            result["test_errors"].append(test_error)
        
        return result
         


if __name__ == "__main__":
    model = BinaryLogReg()
    (x_train, y_train), (x_test, y_test) = load_2_7_mnist()
    history = model.train(x_train, y_train, x_test, y_test, epochs=200)

    # Plot losses (b)
    plt.figure(figsize=(10,6))
    plt.plot(history["train_losses"], label="Train")
    plt.plot(history["test_losses"], label="Test")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("A4b1_plot.png")

    # Plot error (b)
    plt.figure(figsize=(10,6))
    plt.plot(history["train_errors"], label="Train")
    plt.plot(history["test_errors"], label="Test")
    plt.xlabel("Epochs")
    plt.ylabel("Misclassification Error")
    plt.legend()
    plt.savefig("A4b2_plot.png")


    history1 = model.train(x_train, y_train, x_test, y_test, epochs=200, sgd = True, batch_size=1)

    # Plot losses (c)
    plt.figure(figsize=(10,6))
    plt.plot(history1["train_losses"], label="Train")
    plt.plot(history1["test_losses"], label="Test")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("A4c1_plot.png")

    # Plot error (c)
    plt.figure(figsize=(10,6))
    plt.plot(history1["train_errors"], label="Train")
    plt.plot(history1["test_errors"], label="Test")
    plt.xlabel("Epochs")
    plt.ylabel("Misclassification Error")
    plt.legend()
    plt.savefig("A4c2_plot.png")

    history2 = model.train(x_train, y_train, x_test, y_test, epochs=200, sgd = True, batch_size=100)

    # Plot losses (d)
    plt.figure(figsize=(10,6))
    plt.plot(history2["train_losses"], label="Train")
    plt.plot(history2["test_losses"], label="Test")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("A4d1_plot.png")

    # Plot error (d)
    plt.figure(figsize=(10,6))
    plt.plot(history2["train_errors"], label="Train")
    plt.plot(history2["test_errors"], label="Test")
    plt.xlabel("Epochs")
    plt.ylabel("Misclassification Error")
    plt.legend()
    plt.savefig("A4d2_plot.png")
