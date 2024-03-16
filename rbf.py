import numpy as np

class FixedRBFLayer:
    def __init__(self, 
                 X_train, 
                 units, 
                 dist_type, 
                 seed):
        self.X_train = X_train
        self.units = units

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

    # runtime: O(len(x_in)^2) unless x has ridiculous dims look for ways to vectorize
    def feed_forward(self):
        h = np.array([])
        for x_in in self.X_train:
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
        
        rbfLayerUnits = self.layers[0].units
        self.layers[-1].initialize(rbfLayerUnits, np.random.randint(low=0))

    def forward_propagate(self):
        hiddenLayerResults = FixedRBFLayer.feed_forward()