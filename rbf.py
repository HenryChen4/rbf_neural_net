import numpy as np

class FixedRBFLayer:
    def __init__(self,  
                 units, 
                 dist_type, 
                 seed):
        self.units = units

    def initialize(self, X_train):
        self.X_train = X_train

        # choose centers and sigmas randomly
        n = X_train.shape[0]
        if units > n:
            raise Exception(f"Expected hidden unit length < {n}. Actual hidden unit length: {self.units}")
        else:
            rng = np.random.default_rng(seed)
            self.centers = rng.choice(a=X_train, size=units, replace=False)
            if dist_type == "max":
                all_dist = []
                for x in X_train:
                    for c in self.centers:
                        all_dist.append(abs(x - c))
                max_dist = np.max(np.array(all_dist))
                self.sigma = max_dist / np.sqrt(2 * self.units)
            elif dist_type == "avg":
                all_dist = []
                for x in X_train:
                    for c in self.centers:
                        all_dist.append(abs(x - c))
                avg_dist = np.average(np.array(all_dist))
                self.sigma = 2 * avg_dist
            else:
                raise Exception(f"{dist_type} is not a valid distance type. Choose max or avg.")

    def feed_forward(self, x_in):
        h = np.array([])
        for c_j in self.centers:
            d_j = 0
            for x_i in x_in:
                d_j += ((x_i - c_j) ** 2)
            d_j = np.sqrt(d_j)            
            h_j = np.exp(-(d_j**2)/(2 * (self.sigma ** 2)))
            h.append(h_j)
        return h_j

class OutputLayer:
    def __init__(self, 
                 units):
        self.units = units
        self.neurons = np.array([])
        self.W_l = np.array([])
        self.B_l = np.array([])
    
    def initialize(self, prev_layer_units, seed):
        rng = np.random.default_rng(seed)
        W_shape = (self.units, prev_layer_units)
        B_shape = (self.units, 1)
        self.W_l = rng.random(W_shape[0], W_shape[1])
        self.B_l = np.zeros(B_shape)
    
    def feed_forward(self, a_in):
        Z = np.matmul(self.W_l, a_in) + self.B_l
        self.neurons = Z
    
    def output_backprop(self, del_J_del_w, del_J_del_b):
        self.W_l -= del_J_del_w
        self.B_l -= del_J_del_b
    
class RBFModel:
    def __init__(self, layer_arr):
        self.layers = np.array(layer_arr)

        if self.layers.shape[0] != 2:
            raise Exception("Wrong number of layers, check layer array")
        if type(self.layers[0]) != FixedRBFLayer:
            raise Exception("Wrong layer type at position 0, please use FixedRBFLayer")
        if type(self.layers[1]) != OutputLayer:
            raise Exception("Wrong layer type at position 1, please use OutputLayer")
    
    def initialize(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

        # initialize RBF layer
        self.layers[0].initialize(X_train)

        # initialize output layer
        rbfUnits = self.layers[0].units
        self.layers[1].initialize(prev_layer_units=rbfUnits, seed=np.random.default_rng().integers(0))
    
    def forward_propagate(self):
        # forward prop RBF layer
        

        # forward prop output layer