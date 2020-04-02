import numpy as np

class backPropragation:
    no_of_features = 4
    no_of_classes = 3
    def __init__(self,no_of_layers,no_of_neurons,epochs,learning_rate,bias,xy,activation):
        self.no_of_features= self.no_of_features
        self.no_of_classes= self.no_of_classes
        self.no_of_layers=no_of_layers
        self.no_of_neurons=no_of_neurons
        self.epochs=epochs
        self.lr=learning_rate
        self.bias=bias
        self.activation=activation
        self.x_train=xy[0]
        self.x_test=xy[1]
        self.y_train=xy[2]
        self.y_test=xy[3]
        self.bias=bias
        self.dim_of_weights=[self.no_of_features]+self.no_of_neurons+[self.no_of_classes+self.bias]

        self.init_weights()
        print(self.weights)
        print(x)
        print(self.activation(self, "Works OoO"))

    def init_weights(self):
        self.weights=[]
        for i in range(len(self.dim_of_weights)-1):
            dim=np.random.rand(self.dim_of_weights[i+1],self.dim_of_weights[i]+self.bias,1)

            # print(dim.shape)
            # print(dim[0, :, :])
            # print("...................")
            # print(dim[0, :, 0].reshape(dim[0, :, 0].shape[0],1))
            # print("...................")

            self.weights.append(dim)

    def Sigmoid(self,x):
        return x

