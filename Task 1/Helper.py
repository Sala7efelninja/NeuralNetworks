from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


class Helper:
    def __init__(self):
        self.dataset = []
        self.Path = "IrisData.txt"

        self.X_train = []
        self.X_test = []
        self.Y_train = []
        self.Y_test = []

    def read_file(self):
        D = pd.read_csv(self.Path)
        self.dataset = np.asarray(D)

    def set_Data(self, c1, c2, f1, f2):
        c1s = c1 * 50
        c1e = c1s + 50
        c2s = c2 * 50
        c2e = c2s + 50
        x1 = np.array([self.dataset[c1s:c1e, f1], self.dataset[c1s:c1e, f2]],dtype=float).transpose()
        x2 = np.array([self.dataset[c2s:c2e, f1], self.dataset[c2s:c2e, f2]],dtype=float).transpose()

        y1 = np.ones(50)
        y2 = np.ones(50) * -1
        x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.4, shuffle=True)
        x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.4, shuffle=True)
        self.X_train = np.concatenate((x1_train, x2_train))
        self.X_test = np.concatenate((x1_test, x2_test))
        self.Y_train = np.concatenate((y1_train, y2_train))
        self.Y_test = np.concatenate((x1_test, x2_test))

