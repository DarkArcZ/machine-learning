from typing import Tuple
import numpy as np
from utils import problem


class PolynomialRegression:
    @problem.tag("hw1-A", start_line=4)
    def __init__(self, degree: int = 1, reg_lambda: float = 1e-8):
        self.degree: int = degree
        self.reg_lambda: float = reg_lambda
        self.mean = None
        self.std = None


    @staticmethod
    @problem.tag("hw1-A")
    def polyfeatures(X: np.ndarray, degree: int) -> np.ndarray:
        X_ = np.zeros((X.shape[0], degree))
        for i in range(0, X.shape[0]):
            for j in range(1, degree+1):
                X_[i, j-1] = X[i]**j
        return X_


    @problem.tag("hw1-A")
    def fit(self, X: np.ndarray, y: np.ndarray):
        n = X.shape[0]
        X_ = self.polyfeatures(X, self.degree)        
        self.mean = np.mean(X_, axis = 0)
        self.std = np.std(X_, axis = 0) 
        X_ = (X_ - self.mean)/self.std 
        
        # closed-formed poly regression
        X_closed = np.c_[np.ones([n,1]), X_]

        n, d = X_closed.shape 
        d = d-1
        # construct reg matrix 
        reg_matrix = self.reg_lambda * np.eye(d+1)
        reg_matrix[0, 0] = 0

        # analytical solution (X'X + regMatrix)^-1 X' y 
        self.weight = np.linalg.pinv(X_closed.T.dot(X_closed) + reg_matrix).dot(X_closed.T).dot(y)
                

    @problem.tag("hw1-A")
    def predict(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        X_ = self.polyfeatures(X, self.degree)
        X_ = (X_-self.mean)/self.std
        X_closed = np.c_[np.ones([n,1]), X_]
        return X_closed.dot(self.weight)


@problem.tag("hw1-A")
def mean_squared_error(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((b-a)**2)
    return mse


@problem.tag("hw1-A", start_line=5)
def learningCurve(
    Xtrain: np.ndarray,
    Ytrain: np.ndarray,
    Xtest: np.ndarray,
    Ytest: np.ndarray,
    reg_lambda: float,
    degree: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(Xtrain)

    errorTrain = np.zeros(n)
    errorTest = np.zeros(n)
    # Fill in errorTrain and errorTest arrays
    model = PolynomialRegression(degree=degree, reg_lambda=reg_lambda)

    for i in range(1,n):
        model.fit(Xtrain[0:(i+1)], Ytrain[0:(i+1)])
        errorTest[i] = ((model.predict(Xtest[0:(i+1)])-Ytest[0:(i+1)])**2).mean()
        errorTrain[i] = ((model.predict(Xtrain[0:(i+1)])-Ytrain[0:(i+1)])**2).mean()
   
    return errorTrain, errorTest 