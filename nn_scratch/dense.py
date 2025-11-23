"""
dense_example.py
----------------
Journey from procedural neurons to vectorized dense layers:

1. Procedural multi-neuron layer
2. DenseLayer class (OOP)
3. Fully vectorized DenseLayer

Author: Mohammed Silva
"""

import numpy as np
from neuron import Neuron
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# ==========================
# Activation functions
# ==========================
activations = {
    "linear":  (lambda z: z, lambda z, y: 1.0),
    "sigmoid": (lambda z: 1/(1+np.exp(-z)), lambda z, y: y*(1-y)),
    "tanh":    (np.tanh, lambda z, y: 1 - y**2),
    "relu":    (lambda z: np.maximum(0, z), lambda z, y: np.where(z>0, 1, 0))
}

# ==========================
# 1. Create dataset
# ==========================
X, y = make_classification(
    n_samples=100,
    n_features=3,
    n_informative=3,
    n_redundant=0,
    n_classes=2,
    random_state=42
)

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = y.astype(float)  # convert to float for MSE

print("Sample data:")
print("X[0]:", X[0])
print("y[0]:", y[0])
print()

# ==========================
# Global parameters
# ==========================
n_inputs = X.shape[1]
n_neurons = 3
lr = 0.1
epochs = 50

# ==========================
# 2. Procedural multi-neuron layer
# ==========================
print("=== Procedural multi-neuron layer ===")

dense_layer = [Neuron(n_inputs, activation="sigmoid") for _ in range(n_neurons)]

for epoch in range(1, epochs + 1):
    epoch_loss = 0
    for i in range(X.shape[0]):
        x_sample = X[i]
        y_sample = y[i]

        outputs = np.array([neuron.forward(x_sample) for neuron in dense_layer])
        y_pred = outputs.mean()
        loss = (y_pred - y_sample) ** 2
        epoch_loss += loss

        for neuron in dense_layer:
            neuron.backward(y_sample, lr)

    if epoch % 10 == 0 or epoch == 1:
        print(f"[Procedural] Epoch {epoch}: Mean Loss = {epoch_loss / X.shape[0]:.4f}")

preds_proc = np.array([np.mean([neuron.forward(x) for neuron in dense_layer]) for x in X])
pred_labels_proc = (preds_proc > 0.5).astype(int)
accuracy_proc = (pred_labels_proc == y).mean()
print(f"[Procedural] Final Accuracy: {accuracy_proc*100:.2f}%\n")


# ==========================
# 3. DenseLayer class (OOP)
# ==========================
class DenseLayer:
    """
    Reusable dense layer containing multiple neurons.
    Supports forward, backward, and training operations.
    """
    def __init__(self, n_inputs, n_neurons, activation="sigmoid"):
        self.neurons = [Neuron(n_inputs, activation=activation) for _ in range(n_neurons)]
        self.n_neurons = n_neurons

    def forward(self, x):
        outputs = np.array([neuron.forward(x) for neuron in self.neurons])
        return outputs.mean()

    def backward(self, y_true, lr):
        for neuron in self.neurons:
            neuron.backward(y_true, lr)

    def train(self, X, y, epochs=50, lr=0.1, verbose=True):
        loss_history = []
        for epoch in range(1, epochs + 1):
            epoch_loss = 0
            for i in range(X.shape[0]):
                x_sample = X[i]
                y_sample = y[i]

                y_pred = self.forward(x_sample)
                loss = (y_pred - y_sample) ** 2
                epoch_loss += loss

                self.backward(y_sample, lr)

            mean_loss = epoch_loss / X.shape[0]
            loss_history.append(mean_loss)
            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(f"[DenseLayer Class] Epoch {epoch}: Mean Loss = {mean_loss:.4f}")
        return loss_history

    def predict(self, X):
        preds = np.array([self.forward(x) for x in X])
        return (preds > 0.5).astype(int)


# ==========================
# 4. Use DenseLayer class
# ==========================
print("=== DenseLayer class (OOP) ===")
dense_layer_class = DenseLayer(n_inputs=n_inputs, n_neurons=n_neurons, activation="sigmoid")
dense_layer_class.train(X, y, epochs=epochs, lr=lr)

pred_labels_class = dense_layer_class.predict(X)
accuracy_class = (pred_labels_class == y).mean()
print(f"[DenseLayer Class] Final Accuracy: {accuracy_class*100:.2f}%\n")


# ==========================
# 5. Fully vectorized DenseLayer
# ==========================
class DenseLayerVectorized:
    """
    Fully vectorized dense layer for multiple neurons.
    Supports vectorized forward/backward and training.
    """
    def __init__(self, n_inputs, n_neurons, activation="sigmoid"):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.activation_name = activation
        self.w = np.random.randn(n_neurons, n_inputs) * 0.01
        self.b = np.zeros(n_neurons)

    def forward(self, x):
        self.x = x
        self.z = self.w @ x + self.b
        act_func, _ = activations[self.activation_name]
        self.y = act_func(self.z)
        return self.y  # Return full vector

    def backward(self, dL_dy, lr):
        """
        dL_dy: gradient from next layer (vector for hidden, scalar for output)
        """
        # If dL_dy is scalar (from output layer), broadcast to all neurons
        if np.isscalar(dL_dy):
            dL_dy = np.full(self.n_neurons, dL_dy / self.n_neurons)
        
        _, act_deriv = activations[self.activation_name]
        dy_dz = act_deriv(self.z, self.y)
        dL_dz = dL_dy * dy_dz

        dL_dw = dL_dz[:, None] * self.x[None, :]
        dL_db = dL_dz

        self.w -= lr * dL_dw
        self.b -= lr * dL_db
        
        # Gradient to pass to previous layer
        dL_dx = self.w.T @ dL_dz
        
        return dL_dx

    def train(self, X, y, epochs=50, lr=0.1, verbose=True):
        """
        Train single layer (for standalone testing)
        """
        loss_history = []
        for epoch in range(1, epochs + 1):
            epoch_loss = 0
            for i in range(X.shape[0]):
                y_pred_vec = self.forward(X[i])
                y_pred = y_pred_vec.mean()  # Convert to scalar for loss
                loss = (y_pred - y[i]) ** 2
                epoch_loss += loss
                
                # Backward pass for single layer
                dL_dy = 2 * (y_pred - y[i])
                self.backward(dL_dy, lr)
            
            mean_loss = epoch_loss / X.shape[0]
            loss_history.append(mean_loss)
            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(f"[DenseLayerVectorized] Epoch {epoch}: Mean Loss = {mean_loss:.4f}")
        return loss_history

    def predict(self, X):
        """
        Predict class labels for input samples X
        """
        preds = np.array([self.forward(x).mean() for x in X])
        return (preds > 0.5).astype(int)

# ==========================
# 6. Use Fully Vectorized DenseLayer
# ==========================
print("=== Fully Vectorized DenseLayer ===")
dense_layer_vec = DenseLayerVectorized(n_inputs=n_inputs, n_neurons=n_neurons, activation="sigmoid")
dense_layer_vec.train(X, y, epochs=epochs, lr=lr)

pred_labels_vec = dense_layer_vec.predict(X)
accuracy_vec = (pred_labels_vec == y).mean()
print(f"[DenseLayerVectorized] Final Accuracy: {accuracy_vec*100:.2f}%\n")


# ==========================
# 7. Accuracy comparison
# ==========================
print("=== Accuracy Comparison ===")
print(f"Procedural   : {accuracy_proc*100:.2f}%")
print(f"DenseLayer   : {accuracy_class*100:.2f}%")
print(f"Vectorized   : {accuracy_vec*100:.2f}%")
