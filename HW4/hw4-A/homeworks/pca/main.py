from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils import load_dataset, problem


@problem.tag("hw4-A")
def reconstruct_demean(uk: np.ndarray, demean_data: np.ndarray) -> np.ndarray:
    """Given a demeaned data, create a recontruction using eigenvectors provided by `uk`.

    Args:
        uk (np.ndarray): First k eigenvectors. Shape (d, k).
        demean_vec (np.ndarray): Demeaned data (centered at 0). Shape (n, d)

    Returns:
        np.ndarray: Array of shape (n, d).
            Each row should correspond to row in demean_data,
            but first compressed and then reconstructed using uk eigenvectors.
    """
    # for k in range(1, 101):
    #     demean_reconstruct = np.dot(demean_data, np.dot(uk[:,:k], uk[:,:k].T))

    # return demean_reconstruct
    return demean_data.dot(uk).dot(uk.T)


@problem.tag("hw4-A")
def reconstruction_error(uk: np.ndarray, demean_data: np.ndarray) -> float:
    """Given a demeaned data and some eigenvectors calculate the squared L-2 error that recontruction will incur.

    Args:
        uk (np.ndarray): First k eigenvectors. Shape (d, k).
        demean_data (np.ndarray): Demeaned data (centered at 0). Shape (n, d)

    Returns:
        float: Squared L-2 error on reconstructed data.
    """
    # for k in range(1, 101):
    #     demean_reconstruct = reconstruct_demean(uk, demean_data)
    # # error = np.mean((demean_data - demean_reconstruct)**2)
    # # return error
    #     return np.linalg.norm(demean_data - demean_reconstruct, ord=2)
    reconstruct = reconstruct_demean(uk, demean_data)
    error = np.mean(np.linalg.norm(demean_data-reconstruct, axis=1)**2)
    return error

@problem.tag("hw4-A")
def calculate_eigen(demean_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given demeaned data calculate eigenvalues and eigenvectors of it.

    Args:
        demean_data (np.ndarray): Demeaned data (centered at 0). Shape (n, d)

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of two numpy arrays representing:
            1. Eigenvalues array with shape (d,)
            2. Matrix with eigenvectors as columns with shape (d, d)
    """
    n, d = demean_data.shape
    sigma = np.dot(demean_data.T, demean_data)/n
    eigen_values, eigen_vectors = np.linalg.eig(sigma)
    # idx = eigen_values.argsort()[::-1]
    # eigen_values_sorted, eigen_vectors_sorted = eigen_values[idx], eigen_vectors[:, idx]
    return eigen_values, eigen_vectors

@problem.tag("hw4-A", start_line=2)
def main():
    """
    Main function of PCA problem. It should load data, calculate eigenvalues/-vectors,
    and then answer all questions from problem statement.

    Part A:
        - Report 1st, 2nd, 10th, 30th and 50th largest eigenvalues
        - Report sum of eigenvalues

    Part C:
        - Plot reconstruction error as a function of k (# of eigenvectors used)
            Use k from 1 to 101.
            Plot should have two lines, one for train, one for test.
        - Plot ratio of sum of eigenvalues remaining after k^th eigenvalue with respect to whole sum of eigenvalues.
            Use k from 1 to 101.

    Part D:
        - Visualize 10 first eigenvectors as 28x28 grayscale images.

    Part E:
        - For each of digits 2, 6, 7 plot original image, and images reconstruced from PCA with
            k values of 5, 15, 40, 100.
    """
    #part A
    (x_tr, y_tr), (x_test, _) = load_dataset("mnist")
    n, d = x_tr.shape
    I = np.ones((n, 1))
    mu = np.dot(x_tr.T, I)/n
    demean_data_tr = x_tr - np.dot(I, mu.T)
    eigen_values, eigen_vectors = calculate_eigen(demean_data_tr)
    idx = eigen_values.argsort()[::-1]
    eigen_values_sorted, eigen_vectors_sorted = eigen_values[idx], eigen_vectors[:, idx]
    print(eigen_values_sorted[0], eigen_values_sorted[1], eigen_values_sorted[9], eigen_values_sorted[29], eigen_values_sorted[49])
    print('Summation of eigenvalues:', sum(eigen_values_sorted))

    #Part C
    # current_eigen_values_sum = 0
    # ks = []
    # ratios = []
    # train_losses = []
    # test_losses = []

    # def mse(U , V):
    #     error = np.mean((U - V)**2)
    #     return error

    # mu = np.mean(x_tr)
    # for k in range(1, 101):
    #     eigenvectors = eigen_vectors_sorted[:,0:k]
    #     X_train_reconstruct = np.dot(np.dot(x_tr - mu, eigenvectors), eigenvectors.T)
    #     X_test_reconstruct = np.dot(np.dot(x_test - mu, eigenvectors), eigenvectors.T)
    #     train_losses.append(mse(x_tr, X_train_reconstruct))
    #     test_losses.append(mse(x_test, X_test_reconstruct))
    #     ks.append(k)
    #     current_eigen_values_sum += eigen_values_sorted[k-1]
    #     ratios.append(1 - (current_eigen_values_sum/sum(eigen_values_sorted)))
    #     print(k, train_losses[-1], test_losses[-1])
    # plt.figure(figsize=(10, 6))
    # plt.plot(ks, train_losses, label='Train Reconstruction Error')
    # plt.plot(ks, test_losses, label='Test Reconstruction Error')
    # plt.xlabel('k')
    # plt.ylabel('Reconstruction Error')
    # plt.legend()
    # plt.savefig('./A5c_err.png')

    # plt.figure(figsize=(10, 6))
    # plt.plot(ks, ratios)
    # plt.xlabel('k')
    # plt.ylabel('Eigenvalue Ratio')
    # plt.savefig('./A5c_ratio.png')

    #Part D
    # plt.figure(figsize=(10, 2))
    # fig, axes = plt.subplots(2, 5)
    # for i, ax in enumerate(axes.flatten()):
    #     plottable_image = np.reshape(eigen_vectors_sorted[:,i], (28,28))
    #     ax.imshow(plottable_image.real)
    #     ax.set_title('k = {}'.format(i + 1))
    #     ax.axis('off')
    # plt.savefig('./A5d.png')

   #Part E
    # plt.figure(figsize=(10, 6))
    # fig, axes = plt.subplots(3, 5)
    # # Y_train[5] => 2, Y_train[13] => 6, Y_train[15]
    # plot_labels = [(2, 5), (6, 13), (7, 15)]
    # plot_ks = [-1, 5, 15, 40, 100]
    # for nrow, (label, index) in enumerate(plot_labels):
    #     for ncol, k in enumerate(plot_ks):
    #         # if k == -1:
    #         #     plottable_image = np.reshape(x_tr[index], (28,28))
    #         #     axes[nrow, ncol].imshow(plottable_image.real)
    #         #     axes[nrow, ncol].set_title('Original Image')
    #         #     axes[nrow, ncol].axis('off')
    #         # else:
    #             reconstruct = (np.dot(x_tr - mu.T, np.dot(eigen_vectors_sorted[:, :k], eigen_vectors_sorted[:, :k].T)) + mu.reshape((784,)))[index]
    #             plottable_image = np.reshape((reconstruct), (28,28))
    #             axes[nrow, ncol].imshow(plottable_image.real)
    #             axes[nrow, ncol].set_title('k = {}'.format(k))
    #             axes[nrow, ncol].axis('off')
    # plt.savefig('./A5e.png')

    #For A6(d)
    plt.figure(figsize=(10, 6))
    fig, axes = plt.subplots(3, 4)
    # Y_train[5] => 2, Y_train[13] => 6, Y_train[15]
    plot_labels = [(2, 5), (6, 13), (7, 15)]
    plot_ks = [-1, 32, 64, 128]
    for nrow, (label, index) in enumerate(plot_labels):
        for ncol, k in enumerate(plot_ks):
            # if k == -1:
            #     plottable_image = np.reshape(x_tr[index], (28,28))
            #     axes[nrow, ncol].imshow(plottable_image.real)
            #     axes[nrow, ncol].set_title('Original Image')
            #     axes[nrow, ncol].axis('off')
            # else:
                reconstruct = (np.dot(x_tr - mu.T, np.dot(eigen_vectors_sorted[:, :k], eigen_vectors_sorted[:, :k].T)) + mu.reshape((784,)))[index]
                plottable_image = np.reshape((reconstruct), (28,28))
                axes[nrow, ncol].imshow(plottable_image.real)
                axes[nrow, ncol].set_title('k = {}'.format(k))
                axes[nrow, ncol].axis('off')
                plt.savefig('./A6d.png')

if __name__ == "__main__":
    main()
