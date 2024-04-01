import numpy as np

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
    
    def forward_prop_single(self, x_in):
        # forward prop RBF layer
        self.h = self.layers[0].feed_fowards(x_in)
        
        # forward prop output layer
        y = self.layers[1].feed_forward(self.h)

        # return prediction
        return y
    
    def back_prop_single(self, y_pred, y_actual):
        del_j_del_w = (y_actual - y_pred) * np.transpose(self.h)
        del_j_del_b = (y_actual - y_pred)
        self.layers[1].output_backprop(del_j_del_w, del_j_del_b)