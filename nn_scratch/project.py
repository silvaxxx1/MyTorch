# mlp_cats_dogs.py
import numpy as np
from PIL import Image
import os
from mlp import MLP

# ==========================s
# 1. Load and preprocess images
# ==========================
def load_images(folder, label, size=(64,64)):
    X_list, y_list = [], []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            path = os.path.join(folder, filename)
            img = Image.open(path).resize(size)
            img_array = np.array(img) / 255.0  # normalize
            X_list.append(img_array.flatten())
            y_list.append(label)
    return X_list, y_list

# Load cats (0) and dogs (1)
X_cats, y_cats = load_images("data/cats", label=0)
X_dogs, y_dogs = load_images("data/dogs", label=1)

# Combine and convert to numpy arrays
X = np.array(X_cats + X_dogs)
y = np.array(y_cats + y_dogs).astype(float)

# ==========================
# 2. Shuffle dataset
# ==========================
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X, y = X[indices], y[indices]

# ==========================
# 3. Define MLP
# ==========================
input_size = X.shape[1]
mlp = MLP(layer_sizes=[input_size, 128, 64, 1], activations=["sigmoid", "sigmoid", "sigmoid"])

# ==========================
# 4. Train
# ==========================
mlp.train(X, y, epochs=50, lr=0.1)

# ==========================
# 5. Evaluate
# ==========================
pred_labels = mlp.predict(X)
accuracy = (pred_labels == y).mean()
print(f"MLP Accuracy on Cats vs Dogs: {accuracy*100:.2f}%")
