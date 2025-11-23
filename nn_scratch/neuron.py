import numpy as np

# ===================================================================
#                 STEP 0 — DEFINE INPUT AND TARGET
# ===================================================================

x = np.array([1.0, 2.0, -1.0])
y_true = 0.7

print("=== INPUT AND TARGET ===")
print("x:", x)
print("y_true:", y_true)
print()


# ===================================================================
#           STEP 1 — MANUAL TRAINING LOOP (basic)
# ===================================================================

w = np.random.randn(3) * 0.01
b = 0.0
activation = "linear"   # "linear", "relu", "sigmoid", "tanh"
lr = 0.1
epochs = 50

print("=== MANUAL TRAINING LOOP ===")
for epoch in range(1, epochs + 1):
    z = np.dot(w, x) + b
    
    if activation == "linear":
        y_pred = z
        dy_dz = 1.0
    elif activation == "relu":
        y_pred = np.maximum(0, z)
        dy_dz = np.where(z > 0, 1.0, 0.0)
    elif activation == "sigmoid":
        y_pred = 1 / (1 + np.exp(-z))
        dy_dz = y_pred * (1 - y_pred)
    elif activation == "tanh":
        y_pred = np.tanh(z)
        dy_dz = 1 - np.tanh(z)**2
    
    loss = (y_pred - y_true)**2
    dL_dypred = 2 * (y_pred - y_true)
    dL_dz = dL_dypred * dy_dz
    dL_dw = dL_dz * x
    dL_db = dL_dz
    w -= lr * dL_dw
    b -= lr * dL_db

    if epoch % 10 == 0 or epoch == 1:
        print(f"=== Epoch {epoch} ===")
        print("z:", z)
        print("y_pred:", y_pred)
        print("Loss:", loss)
        print("dL/dw:", dL_dw)
        print("dL/db:", dL_db)
        print("Updated weights:", w)
        print("Updated bias:", b)
        print()

print("=== MANUAL TRAINING COMPLETE ===")
print("Final prediction:", y_pred)
print("Final loss:", loss)
print("Final weights:", w)
print("Final bias:", b)
print("\n\n")


# ===================================================================
#           ACTIVATION DICTIONARY (GLOBAL)
# ===================================================================

activations = {
    "linear": (lambda z: z, lambda z, y: 1.0),
    "sigmoid": (lambda z: 1/(1+np.exp(-z)), lambda z, y: y*(1-y)),
    "tanh": (np.tanh, lambda z, y: 1 - y**2),
    "relu": (lambda z: np.maximum(0, z), lambda z, y: np.where(z>0, 1, 0))
}


# ===================================================================
#           STEP 2 — NEURON CLASS TRAINING
# ===================================================================

class Neuron:
    def __init__(self, n_inputs, activation="linear"):
        self.w = np.random.randn(n_inputs) * 0.01
        self.b = 0.0
        self.activation = activation

    def forward(self, x):
        self.x = x
        self.z = np.dot(self.w, x) + self.b
        act_func, _ = activations[self.activation]  # reference global dict
        self.y_pred = act_func(self.z)
        return self.y_pred

    def backward(self, y_true, lr):
        dL_dypred = 2 * (self.y_pred - y_true)
        _, act_deriv = activations[self.activation]  # reference global dict
        dy_dz = act_deriv(self.z, self.y_pred)
        dL_dz = dL_dypred * dy_dz
        dL_dw = dL_dz * self.x
        dL_db = dL_dz
        self.w -= lr * dL_dw
        self.b -= lr * dL_db
        loss = (self.y_pred - y_true)**2
        return dL_dw, dL_db, loss


# Train neuron
neuron = Neuron(n_inputs=3, activation="linear")
lr = 0.1
epochs = 50

print("=== CLASS TRAINING LOOP ===")
for epoch in range(1, epochs + 1):
    y_pred = neuron.forward(x)
    dL_dw, dL_db, loss = neuron.backward(y_true, lr)

    if epoch % 10 == 0 or epoch == 1:
        print(f"=== Epoch {epoch} ===")
        print("z:", neuron.z)
        print("y_pred:", y_pred)
        print("Loss:", loss)
        print("dL/dw:", dL_dw)
        print("dL/db:", dL_db)
        print("Updated weights:", neuron.w)
        print("Updated bias:", neuron.b)
        print()

print("=== CLASS TRAINING COMPLETE ===")
print("Final prediction:", y_pred)
print("Final loss:", loss)
print("Final weights:", neuron.w)
print("Final bias:", neuron.b)



# ===================================================================
# COMMENTED: NEW DESIGN-PRINCIPLE APPROACH
# ===================================================================

"""
# Define activation functions separately
def linear(z): return z
def dlinear(z, y): return 1.0

def sigmoid(z): return 1 / (1 + np.exp(-z))
def dsigmoid(z, y): return y * (1 - y)  # y = sigmoid(z)

def tanh(z): return np.tanh(z)
def dtanh(z, y): return 1 - y**2        # y = tanh(z)

def relu(z): return np.maximum(0, z)
def drelu(z, y): return np.where(z>0, 1, 0)

# Neuron class using explicit activation functions
class Neuron:
    def __init__(self, n_inputs, activation_func=linear, activation_deriv=dlinear):
        self.w = np.random.randn(n_inputs) * 0.01
        self.b = 0.0
        self.activation_func = activation_func
        self.activation_deriv = activation_deriv

    def forward(self, x):
        self.x = x
        self.z = np.dot(self.w, x) + self.b
        self.y_pred = self.activation_func(self.z)
        return self.y_pred

    def backward(self, y_true, lr):
        dL_dypred = 2 * (self.y_pred - y_true)
        dy_dz = self.activation_deriv(self.z, self.y_pred)
        dL_dz = dL_dypred * dy_dz
        dL_dw = dL_dz * self.x
        dL_db = dL_dz
        self.w -= lr * dL_dw
        self.b -= lr * dL_db
        loss = (self.y_pred - y_true)**2
        return dL_dw, dL_db, loss

# Training is identical, just pass activation functions instead of string
# neuron = Neuron(n_inputs=3, activation_func=sigmoid, activation_deriv=dsigmoid)
"""