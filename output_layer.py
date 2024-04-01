import numpy as np

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