import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
class Perceptron:
    def __init__(self,Cols,Classes=["Iris-versicolor", "Iris-virginica", "Iris-setosa"]):
        self.Cols=Cols
        self.Classes=Classes
        X, Y = self.read_Date()
        # Plot_Data(Data)
        X = np.asarray(X)
        Y = np.asarray(Y)
        X_train, X_test, Y_train, Y_test = self.select_Data(X, Y, self.Cols, self.Classes)
        W = np.random.rand(1,len(self.Cols)+1)

    def read_Date(Path="IrisData.txt"):
        File = open(Path)
        D = pd.read_csv(Path)
        return D.iloc[:, [0, 1, 2, 3]], D['Class']

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

    def select_Data(X, Y, Cols, Classes):
        X = X[:, Cols]
        x1 = x2 = x3 = t1 = t2 = t3 = y1 = y2 = y3 = yt1 = yt2 = yt3 = []

        if "Iris-setosa" in Classes:
            x1, t1, y1, yt1 = train_test_split(X[0:50], Y[0:50], test_size=0.40, shuffle=True)
        if "Iris-versicolor" in Classes:
            x2, t2, y2, yt2 = train_test_split(X[50:100], Y[50:100], test_size=0.40, shuffle=True)
        if "Iris-virginica" in Classes:
            x3, t3, y3, yt3 = train_test_split(X[100:150], Y[100:150], test_size=0.40, shuffle=True)
        X_train = np.concatenate([x1, x2, x3])
        X_test = np.concatenate([t1, t2, t3])
        Y_train = np.concatenate([y1, y2, y3])
        Y_test = np.concatenate([yt1, yt2, yt3])
        return X_train, X_test, Y_train, Y_test



    print(X_test)