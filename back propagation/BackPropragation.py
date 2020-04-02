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
        if self.bias:
            self.x_train = hp.add_bias( self.x_train)
            self.x_test = hp.add_bias(self.x_test)
        self.dim_of_weights=[self.no_of_features]+self.no_of_neurons+[self.no_of_classes]

        self.init_network()

        # print(self.weights)
        self.fit()


    def init_network(self):
        self.weights = []
        self.f=[[] for i in range(len(self.dim_of_weights) - 1)]
        for i in range(len(self.dim_of_weights) - 1):
            dim = np.random.rand(self.dim_of_weights[i + 1], self.dim_of_weights[i] + self.bias)
            self.weights.append(dim)
            print(dim.shape)

    def fit(self):
        for i in range(self.epochs):
            self.forward(self.x_train)
            self.reach_output()
            self.backward()
            self.update()
    def forward(self,l):
        for i in range(self.no_of_layers+1):
            l = np.dot(l, self.weights[i].T)
            self.f[i]=self.activation(l)
    def backward(self):
        self.delta=[]
        for i in reversed( range(self.no_of_layers+1)):

            if i == self.no_of_layers:
                d=(self.y_train-self.f[i])*self.f[i]*(1-self.f[i])
                self.delta.append(d)
            else:
                print("i= ", i, "weight[i]= ", self.weights[i + 1].shape[0])
                errors=[]
                for j in range(self.weights[i+1].shape[0]):

                    d=np.sum(self.delta[-1]*self.weights[i+1].T[j],axis=1).reshape(90,1)
                    #print(self.f[i][:,j].T.shape).reshape(90,1)
                    d=d*(self.f[i][:,j].reshape(90,1)*(1-self.f[i][:,j]).reshape(90,1))
                    errors.append(d)
                self.delta.append(np.concatenate(errors,axis=1))
    def update(self):
        if self.bias:
            self.x_train=hp.add_bias(self.x_train)
        self.weights[0] = self.weights[0] + self.lr *np.dot( self.delta[-1].T, self.x_train)
        for i in range(1,self.no_of_layers+1):
            if self.bias:
                self.f[i-1] = hp.add_bias(self.f[i-1])
            self.weights[i]=self.weights[i]+self.lr*np.dot(self.delta[-(i+1)].T,self.f[i-1])
    def predict(self):
        self.forward(self.x_test)
        self.reach_output()
        return self.f[-1]
    def Sigmoid(x):
        return 1/(1+np.exp(-1*x))

    def tanh(x):
        ex = np.exp(x)
        nex = np.exp(-x)
        return (ex-nex)/(ex+nex)

    def reach_output(self):
        max_list=np.argmax(self.f[-1],axis=1)
        self.f[-1]=np.asarray(self.f[-1]*0,dtype="int8")
        for i in range(len(max_list)):
            self.f[-1][i, max_list[i]] = 1
