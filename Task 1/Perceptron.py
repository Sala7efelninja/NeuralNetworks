import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class Perceptron:
    def __init__(self, Cols, Classes=["Iris-versicolor", "Iris-virginica", "Iris-setosa"]):
        self.Cols = Cols
        self.Classes = Classes
        X, Y = self.read_Date()
        # Plot_Data(Data)
        X = np.asarray(X)
        Y = np.asarray(Y)
        X_train, X_test, Y_train, Y_test = self.select_Data(X, Y, self.Cols, self.Classes)
        W = np.random.rand(1, len(self.Cols) + 1)

    def Plot_Data(Data):
        plt.figure('fig1')
        plt.scatter(Data['X1'], Data['X2'])
        plt.scatter(Data['X1'], Data['X3'])
        plt.scatter(Data['X1'], Data['X4'])
        plt.scatter(Data['X2'], Data['X3'])
        plt.scatter(Data['X2'], Data['X4'])
        plt.scatter(Data['X3'], Data['X4'])
        plt.xlabel("X")
        plt.xlabel("Y")
        plt.show()
