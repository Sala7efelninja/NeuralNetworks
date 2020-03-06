import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def Plot_data():
    data = pd.read_csv("IrisData.txt")
    for feature_x in range(4):
        for feature_y in range(feature_x, 4):
            if feature_y != feature_x:
                plt.scatter(data.iloc[0:50, feature_x], data.iloc[0:50, feature_y])
                plt.scatter(data.iloc[50:101, feature_x], data.iloc[50:101, feature_y])
                plt.scatter(data.iloc[101:, feature_x], data.iloc[101:, feature_y])
                plt.xlabel('X' + str(feature_x + 1))
                plt.ylabel('X' + str(feature_y + 1))
                plt.show()


def line(Y, x, w):
    p = []
    if len(w) == 2:
        w = [0, w[0], w[1]]
        x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
    X = x[:, 1]
    y = (w[1] * X + w[0]) / -w[2]
    plt.plot(X, y, color="red", linewidth=3)
    plt.scatter(x[:, 1][Y == 1], x[:, 2][Y == 1])
    plt.scatter(x[:, 1][Y == -1], x[:, 2][Y == -1])
    plt.show()
# Plot_data()
