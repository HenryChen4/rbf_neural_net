Radial basis function neural network for function approximation

Sample output:
<img width="888" alt="Screenshot 2024-04-03 at 6 19 01 PM" src="https://github.com/HenryChen4/rbf_neural_net/assets/71111859/77a33c68-2788-4be3-ba38-b3df1c22845f">

Usage:
model = RBFModel([FixedRBFLayer(units, "sigma selection type", seed for sigma and center selection), Layer(units)])
total_loss = model.fit(x_train, y_train, alpha, epochs)
predictions = model.predict(x_test)
