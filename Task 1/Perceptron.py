import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
import helper as hlp

def Activation(z):
    if z == 0:
        return 0
    if z>0:
        return 1
    if z<0:
        return -1


def fit(x_train,y_train,epochs,eta):
    w0=np.random.rand(2,1)
    for ephoch in range(epochs):
        for i in range(len(x_train)):
            z=np.dot(x_train[i],w0)
            #compute Activation
            a=Activation(z)
            if a !=y_train[i]:
                #calc loss
                loss=a-y_train[i]
                #@q=np.dot(loss,x_train[i]).reshape(2,1)
                q=loss*x_train[i]
                c=(eta*q).reshape(2,1)
                w0=w0+c

    return w0

def predict(x_test,w):
    y_pred=[]
    for i in range(len(x_test)):
        z = np.dot(x_test[i], w)
        # compute Activation
        a = Activation(z)
        y_pred.append(a)

    return y_pred

