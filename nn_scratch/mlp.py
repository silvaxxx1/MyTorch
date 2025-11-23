# mlp.py
import numpy as np
from dense import DenseLayerVectorized  # fully vectorized dense layer

class MLP:
    """
    Multi-Layer Perceptron (MLP) using fully vectorized dense layers.
    Supports multiple layers, training, and prediction.
    """
    def __init__(self, layer_sizes, activations=None):
        self.layers = []
        if activations is None:
            activations = ["sigmoid"] * (len(layer_sizes) - 1)
        assert len(activations) == len(layer_sizes) - 1, "Activations length mismatch"

        for i in range(1, len(layer_sizes)):
            n_inputs = layer_sizes[i - 1]
            n_neurons = layer_sizes[i]
            layer = DenseLayerVectorized(n_inputs, n_neurons, activation=activations[i - 1])
            self.layers.append(layer)

    def forward(self, x):
        """
        Forward pass through all layers.
        """
        out = x
        for layer in self.layers:
            out = layer.forward(out)  # forward returns vector
        # Collapse final layer output to scalar
        if isinstance(out, np.ndarray):
            out = out.mean()
        return out

    def backward(self, y_true, y_pred, lr):
        """
        Backward pass through all layers with proper gradient propagation.
        """
        # Start with gradient from loss: dL/dy_pred
        dL_dy = 2 * (y_pred - y_true)
        
        # Get the number of output neurons for proper gradient distribution
        n_output_neurons = self.layers[-1].n_neurons
        dL_dy = dL_dy / n_output_neurons  # Distribute gradient across output neurons
        
        # Backpropagate through layers in reverse order
        for layer in reversed(self.layers):
            dL_dy = layer.backward(dL_dy, lr)

    def train(self, X, y, epochs=50, lr=0.1, verbose=True):
        """
        Train the MLP on dataset X, y
        """
        for epoch in range(1, epochs + 1):
            epoch_loss = 0
            for i in range(X.shape[0]):
                x_sample = X[i]
                y_sample = y[i]

                y_pred = self.forward(x_sample)
                loss = (y_pred - y_sample) ** 2
                epoch_loss += loss

                self.backward(y_sample, y_pred, lr)  # Pass both y_true and y_pred

            mean_loss = epoch_loss / X.shape[0]
            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(f"[MLP] Epoch {epoch}: Mean Loss = {mean_loss:.4f}")

    def predict(self, X):
        preds = np.array([self.forward(x) for x in X])
        return (preds > 0.5).astype(int)

# mlp_real_example.py
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# ==========================
# 1. Load and preprocess dataset
# ==========================
data = load_breast_cancer()
X, y = data.data, data.target.astype(float)

scaler = StandardScaler()
X = scaler.fit_transform(X)

print("Sample data:")
print("X[0]:", X[0])
print("y[0]:", y[0])
print()

# ==========================
# 2. Define MLP
# ==========================
# 30 inputs -> 16 hidden -> 8 hidden -> 1 output
mlp = MLP(layer_sizes=[X.shape[1], 16, 8, 1], activations=["sigmoid", "sigmoid", "sigmoid"])

# ==========================
# 3. Train MLP
# ==========================
mlp.train(X, y, epochs=100, lr=0.1)

# ==========================
# 4. Evaluate MLP
# ==========================
pred_labels = mlp.predict(X)
accuracy = (pred_labels == y).mean()
print(f"\nMLP Accuracy on Breast Cancer dataset: {accuracy*100:.2f}%")
