Radial basis function neural network

Exploring function approximation and more eccentric techniques.

Anticipated usage:
1. create the model:
model = RBFModel([FixedRBFLayer(units=5, dist_type="max", seed=12345), (OutputLayer(units=1)])
2. initialize the model
model.initialize(X_train, Y_train)
3. train, test, and get results of model
results = model.fit(X_train, Y_train, alpha, epochs, seed)
