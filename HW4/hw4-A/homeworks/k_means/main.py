if __name__ == "__main__":
    from k_means import calculate_error, lloyd_algorithm# type: ignore
else:
    from .k_means import lloyd_algorithm, calculate_error

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


@problem.tag("hw4-A", start_line=1)
def main():
    """Main function of k-means problem

    You should:
        a. Run Lloyd's Algorithm for k=10, and report 10 centers returned.
        b. For ks: 2, 4, 8, 16, 32, 64 run Lloyd's Algorithm,
            and report objective function value on both training set and test set.
            (All one plot, 2 lines)

    NOTE: This code takes a while to run. For debugging purposes you might want to change:
        x_train to x_train[:10000]. CHANGE IT BACK before submission.
    """
    (x_train, _), (x_test, _) = load_dataset("mnist")
    
    #Part B
    # centers = lloyd_algorithm(x_train, 10)
    # errors = calculate_error(x_train, centers)
    # plt.figure()
    # plt.plot(errors[:,0],errors[:,1],label='train error')
    # plt.xlabel('iteration number')
    # plt.ylabel('objective function')
    # plt.legend()
    # plt.savefig('./A4b_graph.png')

    # fig, ax = plt.subplots(2,5)
    # for i, ax in enumerate(ax.flatten()):
    #     plottable_image = np.reshape(centers[i,:], (28, 28))
    #     ax.imshow(plottable_image.real, cmap='gray_r')
    # plt.savefig('./A4b_image.png')

    #Part C
    K_list = np.array([2, 4, 8, 16, 32, 64])
    result = list()
    for k in K_list:
        centers = lloyd_algorithm(x_train, k)
        train_error = calculate_error(x_train, centers)
        test_error = calculate_error(x_test, centers)
        result.append((k,train_error,test_error))
        print("Done processing k = " + str(k))
        

    #plot
    plt.figure()
    plt.plot(result[:,0],result[:,1],label='train error')
    plt.plot(result[:,0],result[:,2],label='test error')
    plt.xlabel('k')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('./A4c.png')


if __name__ == "__main__":
    main()
