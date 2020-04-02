from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import pandas as pd

def read_file():
    df= pd.read_csv("IrisData.txt")
    dataset = np.asarray(df)
    return dataset,df

def set_Data(num_of_features,num_of_classes,bias):

    dataset,df=read_file()

    x_train=[]
    x_test=[]
    y_train=[]
    y_test=[]

    for c in range(num_of_classes):
        cs= c * 50
        ce = cs +50
        x=np.array(dataset[cs:ce,:-1],dtype=float)
        y=np.ones(50,dtype="int8").reshape(50,1)*c

        x1,x2,y1,y2=train_test_split(x, y, test_size=0.4, shuffle=True)

        x_train.append(x1)
        y_train.append(y1)
        x_test.append(x2)
        y_test.append(y2)

    x_train=np.concatenate(x_train)
    y_train=np.concatenate(y_train)
    x_test=np.concatenate(x_test)
    y_test=np.concatenate(y_test)


    x_train, y_train = shuffle(x_train, y_train, random_state=0)
    x_test, y_test = shuffle(x_test, y_test, random_state=0)
    if bias:
     x_train= add_bias(x_train)
     x_test = add_bias(x_test)
    return [x_train, x_test, y_train, y_test]

def add_bias(x):
    bias=np.ones((x.shape[0],1))
    x = np.concatenate((bias, x), axis=1)
    return x