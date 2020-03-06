import numpy as np
from sklearn import metrics
#from sklearn.model_selection import train_test_split
import helper as hlp

def Activation(z):
    if z == 0:
        return 0
    if z>0:
        return 1
    if z<0:
        return -1


def fit(x_train,y_train,epochs,eta,bias):
    print(x_train[0].shape[0])
    w0 = np.random.rand(x_train[0].shape[0], 1)
    for ephoch in range(epochs):
        for i in range(len(x_train)):
            z=np.dot(x_train[i],w0)
            #compute Activation
            a=Activation(z)
            if a !=y_train[i]:
                #calc loss
                loss=y_train[i]-a
                q=loss*x_train[i]
                c=(eta*q).reshape(x_train[0].shape[0],1)
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

def evaluate_model(y_true,y_pred):
    cm = metrics.confusion_matrix(y_true, y_pred)
    accuracy=np.mean([y_pred==y_true])
    return accuracy,cm
