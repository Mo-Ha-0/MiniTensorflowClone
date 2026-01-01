# MiniNN - Educational Neural Network Library

A lightweight, educational neural network library built from scratch using NumPy. MiniNN implements core deep learning concepts with clear, readable code perfect for learning and experimentation.

## Features

### Layers
- **Dense**: Fully connected layers with He/Xavier initialization
- **Activations**: ReLU, Sigmoid, Tanh, Linear
- **Regularization**: Dropout, Batch Normalization
- **Loss Functions**: Mean Squared Error, Softmax Cross-Entropy

### Optimizers
- **SGD**: Stochastic Gradient Descent
- **Momentum**: SGD with momentum
- **AdaGrad**: Adaptive learning rates
- **Adam**: Adaptive moment estimation

### Training Features
- Built-in training loop with validation
- Mini-batch gradient descent
- Training/validation metrics tracking
- Hyperparameter tuning via grid search

## Installation

```bash
# Clone the repository
git clone https://github.com/Mo-Ha-0/BeyondersTensorflow.git
cd BeyondersTENSORFLOW

# Install dependencies
pip install numpy scikit-learn
```

## Quick Start

```python
import numpy as np
from BeyondersTENSORFLOW import (
    Dense, ReLU, SoftmaxCrossEntropy, 
    NeuralNetwork, Trainer, Adam
)

# Prepare your data
X_train, y_train = ...  # Your training data
X_test, y_test = ...    # Your test data

# Build a neural network
layers = [
    Dense(input_size, 64),
    ReLU(),
    Dense(64, 32),
    ReLU(),
    Dense(32, num_classes)
]

# Create network and optimizer
network = NeuralNetwork(layers, SoftmaxCrossEntropy())
optimizer = Adam(learning_rate=0.001)

# Train the network
trainer = Trainer(network, optimizer)
trainer.fit(
    X_train, y_train, 
    X_test, y_test,
    epochs=100, 
    batch_size=32,
    verbose=True
)

# Evaluate
accuracy = network.accuracy(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
```

## Examples

### Simple 2D Classification

```python
# See examples/simple_example.py
layers = [
    Dense(2, 10),
    ReLU(),
    Dense(10, 3)
]
network = NeuralNetwork(layers, SoftmaxCrossEntropy())
trainer = Trainer(network, SGD(learning_rate=0.1))
trainer.fit(X_train, y_train, epochs=500, batch_size=16)
```

### Iris Dataset Classification

```python
# See examples/iris_example.py
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load and normalize data
iris = load_iris()
X, y = iris.data, iris.target
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Build network with batch normalization
layers = [
    Dense(4, 16),
    Sigmoid(),
    BatchNormalization(16),
    Dense(16, 8),
    ReLU(),
    Dense(8, 3)
]

network = NeuralNetwork(layers, SoftmaxCrossEntropy())
optimizer = Adam(learning_rate=0.01)
```

### MNIST Digit Classification

```python
# See examples/mnist_example.py
from sklearn.datasets import fetch_openml

# Load MNIST
mnist = fetch_openml("mnist_784", version=1)
X, y = mnist.data / 255.0, mnist.target.astype(int)

# Build deeper network
layers = [
    Dense(784, 256),
    ReLU(),
    BatchNormalization(256),
    Dense(256, 128),
    ReLU(),
    BatchNormalization(128),
    Dense(128, 10)
]

network = NeuralNetwork(layers, SoftmaxCrossEntropy())
optimizer = Adam(learning_rate=0.001)
```

## API Reference

### Layers

#### Dense Layer
```python
Dense(input_size: int, output_size: int)
```
Fully connected layer with He initialization.

#### Activation Layers
```python
ReLU()        # Rectified Linear Unit
Sigmoid()     # Sigmoid activation
Tanh()        # Hyperbolic tangent
Linear()      # Identity function
```

#### Regularization Layers
```python
Dropout(drop_rate: float = 0.5)
BatchNormalization(num_features: int, momentum: float = 0.9)
```

### Loss Functions

```python
MeanSquaredError()           # For regression
SoftmaxCrossEntropy()       # For classification
```

### Optimizers

```python
SGD(learning_rate: float = 0.01)
Momentum(learning_rate: float = 0.01, momentum: float = 0.9)
AdaGrad(learning_rate: float = 0.01, eps: float = 1e-8)
Adam(learning_rate: float = 0.001, beta1: float = 0.9, 
     beta2: float = 0.999, eps: float = 1e-8)
```

### Neural Network

```python
network = NeuralNetwork(layers: List[Layer], loss_layer: Layer)

# Forward pass
predictions = network.predict(x, training=False)

# Compute loss
loss = network.loss(x, y, training=True)

# Compute accuracy
accuracy = network.accuracy(x, y)

# Get gradients
params_grads = network.gradient()
```

### Trainer

```python
trainer = Trainer(network: NeuralNetwork, optimizer: Optimizer)

trainer.fit(
    x_train, y_train,
    x_val=None, y_val=None,
    epochs: int = 100,
    batch_size: int = 32,
    verbose: bool = True,
    print_every: int = 10
)
```

### Hyperparameter Tuning

```python
tuner = HyperparameterTuner()

param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'hidden_size': [32, 64, 128]
}

def network_builder(params):
    layers = [
        Dense(input_size, params['hidden_size']),
        ReLU(),
        Dense(params['hidden_size'], output_size)
    ]
    network = NeuralNetwork(layers, SoftmaxCrossEntropy())
    optimizer = Adam(learning_rate=params['learning_rate'])
    return network, optimizer

best_params = tuner.grid_search(
    X_train, y_train, X_val, y_val,
    param_grid, network_builder,
    epochs=50, batch_size=32
)
```

## Project Structure

```
BeyondersTENSORFLOW/
├── __init__.py              # Main package initialization
├── base.py                  # Base Layer class
├── network.py               # Neural Network class
├── trainer.py               # Training loop implementation
├── tuning.py                # Hyperparameter tuning
├── layers/
│   ├── __init__.py
│   ├── dense.py            # Dense/Fully connected layer
│   ├── activations.py      # Activation functions
│   ├── losses.py           # Loss functions
│   └── regularization.py   # Dropout, BatchNorm
├── optimizers/
│   ├── __init__.py
│   └── optimizers.py       # SGD, Adam, etc.
└── examples/
    ├── simple_example.py   # Basic 2D classification
    ├── iris_example.py     # Iris dataset
    └── mnist_example.py    # MNIST digits
```

## Key Implementation Details

### Optimizer Fix
The optimizers (Momentum, AdaGrad, Adam) use Python's `id()` function to track parameters by memory address, ensuring correct tracking across multiple layers with same parameter names.

### Batch Normalization
Maintains running statistics for inference and uses proper gradient computation during training.

### Numerical Stability
- Softmax uses max subtraction for stability
- Sigmoid clips input values to prevent overflow
- Small epsilon values prevent division by zero

## Requirements

- Python 3.13+ # i mean why not, install one of the latest versions :)
- NumPy
- scikit-learn (for examples and data loading)

## Contributing

This is an educational project. Feel free to:
- Report bugs
- Suggest improvements
- Add new features
- Improve documentation

## License

This project is for educational purposes.

## Author

Mohammad - Version 1.0.0

## Acknowledgments

Built as an educational tool to understand neural network fundamentals and backpropagation from first principles.