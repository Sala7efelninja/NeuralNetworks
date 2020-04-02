import numpy as np
import helper as hp

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

        self.dim_of_weights=[self.no_of_features]+self.no_of_neurons+[self.no_of_classes]

        self.init_network()

        # print(self.weights)
        self.fit()


    def init_network(self):
        self.weights = []
        self.f=[[] for i in range(len(self.dim_of_weights) - 1)]
        for i in range(len(self.dim_of_weights) - 1):
            dim = np.random.rand(self.dim_of_weights[i + 1], self.dim_of_weights[i] + self.bias)
            # f   = np.random.ones(self.x_train.shape[0],self.dim_of_weights[i+1])
            # print(f.shape)
            # print(f)
            # print("...................")
            # print(dim[0, :, 0].reshape(dim[0, :, 0].shape[0],1))
            # print("...................")
            # self.f.append(f)
            self.weights.append(dim)

    def fit(self):
        # for i in range(self.epochs):
        self.forward()

    def forward(self):
        l=self.x_train
        for i in range(self.no_of_layers+1):
            if self.bias:
                l=hp.add_bias(l)
            l = np.dot(l, self.weights[i].T)
            self.f[i]=self.activation(l)

    def Sigmoid(x):
        return 1/(1+np.exp(-1*x))

    def tanh(x):
        ex = np.exp(x)
        nex = np.exp(-x)
        return (ex-nex)/(ex+nex)

