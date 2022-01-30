if __name__ == "__main__":
    from coordinate_descent_algo import train  # type: ignore
else:
    from .coordinate_descent_algo import train

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils import load_dataset, problem
sns.set()

def mse(x, y, w):
    a = y - np.dot(x, w) 
    return (a.T @ a)/ len(y)

@problem.tag("hw2-A", start_line=3)
def main():
    # df_train and df_test are pandas dataframes.
    # Make sure you split them into observations and targets
    m=20
    df_train, df_test = load_dataset("crime")

    y_train = df_train['ViolentCrimesPerPop'] 
    X_train = df_train.drop('ViolentCrimesPerPop', axis = 1) 
    y_test = df_test['ViolentCrimesPerPop'] 
    X_test = df_test.drop('ViolentCrimesPerPop', axis = 1)  

    reg_lambda = max(2*np.sum(X_train.T*(y_train-np.mean(y_train)) , axis=0))
    print('Max lambda: ', reg_lambda)

    lambdas = []
    nonzeros = []
    w_regularization_path = []
    mse_train = []
    mse_test = []
    
    for _ in range(m):
        w, b = train(X_train.values, y_train.values, reg_lambda)
        lambdas.append(reg_lambda)
        nonzeros.append(np.sum(abs(w) > 0))
        w_regularization_path.append(np.copy(w))
        mse_train.append(mse(X_train.values, y_train.values, w))
        mse_test.append(mse(X_test.values, y_test.values, w))
        reg_lambda /= 2
    
    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, nonzeros, '--x')
    plt.xscale('log')
    plt.xlabel('$\lambda$')
    plt.ylabel('Number of Non-zeros')
    plt.savefig('A3c.png')
    plt.tight_layout()

    plt.figure(figsize=(10, 6))
    w_regularization_path = np.array(w_regularization_path)
    coeffs_names = ['agePct12t29', 'pctWSocSec', 'pctUrban', 'agePct65up', 'householdsize']
    coeffs_indices = [X_train.columns.get_loc(i) for i in coeffs_names]

    for coeff_path, label in zip(w_regularization_path[:, coeffs_indices].T, coeffs_names):
	    plt.plot(lambdas, coeff_path, '--x', label=label,)
    plt.xscale('log')
    plt.xlabel('$\lambda$')
    plt.ylabel('Regularization Paths')
    plt.legend()
    plt.savefig('A3d.png')
    plt.tight_layout()

    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, mse_train, '--x', label = 'train_mse')
    plt.plot(lambdas, mse_test, '--x', label = 'test_mse')	
    plt.xscale('log')
    plt.xlabel('$\lambda$')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.savefig('A3e.png')
    plt.tight_layout()
 
    w, b = train(X_train.values, y_train.values, _lambda=30)    

    print('Largest Lasso coefficient: ', X_train.columns[np.argmax(w)], max(w))
    print('Smallest Lasso coefficient: ', X_train.columns[np.argmin(w)], min(w))

if __name__ == "__main__":
    main()
