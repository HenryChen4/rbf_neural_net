import numpy as np

from layer import Layer
from rbf_layer import FixedRBFLayer
import matplotlib.pyplot as plt

class RBFModel:
    def __init__(self, layer_arr):
        self.layers = np.array(layer_arr)
    
    def forward_propagate(self, x_in):
        # feed forward rbf_layer
        a_in = self.layers[0].feed_forward(x_in)
        # feed forward output_layer
        y_hat = self.layers[1].feed_forward(a_in)

        return y_hat

    def fit(self, X_train, Y_train, alpha, epochs, seed=10):
        m = X_train.shape[0]
        n = X_train.shape[1]
        self.loss = 0

        # initialize rbf layer
        self.layers[0].initialize(X_train)
        # initialize output layer
        self.layers[-1].initialize(self.layers[0].units, seed)

        total_loss = []

        for i in range(epochs):
            self.loss = 0

            # batch forward propagation
            all_predictions = []
            for x_in in X_train:
                y_hat = self.forward_propagate(x_in)
                all_predictions.append(y_hat)
            all_predictions = np.array(all_predictions)

            # backpropagation of output layer
            del_j_g_vect = (2/n) * np.array(all_predictions - Y_train)
            del_j_g_mean = np.mean(del_j_g_vect)
            del_j_w_vect = self.layers[-1].W_l * del_j_g_mean
            self.layers[-1].W_l -= alpha * del_j_w_vect
            self.layers[-1].B_l -= alpha * del_j_g_mean

            # compute total lose
            for j in range(m):
                self.loss += (Y_train[j] - all_predictions[j]) ** 2
            self.loss /= m
            total_loss.append(self.loss)

            print(f"Epoch {i} completed!")

        return total_loss
    
    def predict(self, X_train):
        all_predictions = []
        for x_in in X_train:
            y_hat = self.forward_propagate(x_in)
            all_predictions.append(y_hat)
        all_predictions = np.array(all_predictions)
        return all_predictions

model = RBFModel([FixedRBFLayer(50, "max", 3010), Layer(1)])
X_train = np.arange(100)
X_train = X_train.reshape((X_train.shape[0], 1))

Y_train = np.arange(100) + np.random.rand(100) * 10
Y_train = Y_train.reshape((Y_train.shape[0], 1))

epochs = 100
alpha = 0.01
total_loss = model.fit(X_train, Y_train, alpha, epochs) 
predictions = model.predict(X_train)
predictions = predictions.reshape((predictions.shape[0], 1))

total_loss = np.array(total_loss)
total_loss = total_loss.reshape((total_loss.shape[0], ))

plt.subplot(1, 2, 1)
plt.title("Regression")
plt.scatter(X_train, Y_train, s=5)
plt.plot(X_train, predictions)

plt.subplot(1, 2, 2)
plt.title("loss")
plt.plot(np.arange(epochs), total_loss)

plt.show()