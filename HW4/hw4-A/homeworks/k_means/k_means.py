import numpy as np
from copy import deepcopy

from utils import problem


@problem.tag("hw4-A")
def calculate_centers(
    data: np.ndarray, classifications: np.ndarray, num_centers: int
) -> np.ndarray:
    """
    Sub-routine of Lloyd's algorithm that calculates the centers given datapoints and their respective classifications/assignments.
    num_centers is additionally provided for speed-up purposes.

    Args:
        data (np.ndarray): Array of shape (n, d). Training data set.
        classifications (np.ndarray): Array of shape (n,) full of integers in range {0, 1, ...,  num_centers - 1}.
            Data point at index i is assigned to classifications[i].
        num_centers (int): Number of centers for reference.
            Might be usefull for pre-allocating numpy array (Faster that appending to list).

    Returns:
        np.ndarray: Array of shape (num_centers, d) containing new centers.
    """
    new_centers = []
    for i in range(num_centers):
        key = classifications == i
        new_centers.append(np.mean(data[key, :], axis=0))
    new_centers = np.array(new_centers)
    return new_centers


@problem.tag("hw4-A")
def cluster_data(data: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Sub-routine of Lloyd's algorithm that clusters datapoints to centers given datapoints and centers.

    Args:
        data (np.ndarray): Array of shape (n, d). Training data set.
        centers (np.ndarray): Array of shape (k, d). Each row is a center to which a datapoint can be clustered.

    Returns:
        np.ndarray: Array of integers of shape (n,), with each entry being in range {0, 1, 2, ..., k - 1}.
            Entry j at index i should mean that j^th center is the closest to data[i] datapoint.
    """
    nClusters = len(centers)
    distances = [np.linalg.norm(data - center, axis=1) for center in centers]
    closestClusters = np.argmin(np.array(distances), axis=0)
    return closestClusters





@problem.tag("hw4-A")
def calculate_error(data: np.ndarray, centers: np.ndarray) -> float:
    """Calculates error/objective function on a provided dataset, with trained centers.

    Args:
        data (np.ndarray): Array of shape (n, d). Dataset to evaluate centers on.
        centers (np.ndarray): Array of shape (k, d). Each row is a center to which a datapoint can be clustered.
            These should be trained on training dataset.

    Returns:
        float: Single value representing mean objective function of centers on a provided dataset.
    """
    classifications_pred = cluster_data(data, centers)
    k = centers.shape[0]
    n = data.shape[0]

    error_per_cluster = np.zeros(k)
    for i in range(k):
        trunk = data[np.where(classifications_pred==i)[0]]
        error_per_cluster[i] = np.sum(np.sqrt(np.sum(np.square(trunk-centers[i]), axis=1)))
        total_error = np.sum(error_per_cluster)/n
        print("Processing ", str(i))
    return total_error

@problem.tag("hw4-A")
def lloyd_algorithm(
    data: np.ndarray, num_centers: int, epsilon: float = 10e-3
) -> np.ndarray:
    """Main part of Lloyd's Algorithm.

    Args:
        data (np.ndarray): Array of shape (n, d). Training data set.
        num_centers (int): Number of centers to train/cluster around.
        epsilon (float, optional): Epsilon for stopping condition.
            Training should stop when max(abs(centers - previous_centers)) is smaller or equal to epsilon.
            Defaults to 10e-3.

    Returns:
        np.ndarray: Array of shape (num_centers, d) containing trained centers.

    Note:
        - For initializing centers please use the first `num_centers` data points.
    """

    centers = data[:num_centers]
    cluster = cluster_data(data, centers)
    prev_centers = centers
    centers = calculate_centers(data, cluster, num_centers)
    cluster = cluster_data(data, centers)

    while np.max(np.absolute(centers - prev_centers)) > epsilon :
        prev_centers = centers
        centers = calculate_centers(data, cluster, num_centers)
        cluster = cluster_data(data, centers)

    return centers