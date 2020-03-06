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

Plot_data()