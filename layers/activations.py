import numpy as np
from ..base import Layer


class Linear(Layer):
    """Linear activation: f(x) = x"""
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        return x
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout


class ReLU(Layer):
    """ReLU activation: f(x) = max(0, x)"""
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self.cache['x'] = x
        return np.maximum(0, x)
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        x = self.cache['x']
        dx = dout.copy()
        dx[x <= 0] = 0
        return dx


class Sigmoid(Layer):
    """Sigmoid activation: f(x) = 1 / (1 + exp(-x))"""
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        x_clipped = np.clip(x, -500, 500)
        out = 1 / (1 + np.exp(-x_clipped))
        self.cache['out'] = out
        return out
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        out = self.cache['out']
        dx = dout * out * (1 - out)
        return dx


class Tanh(Layer):
    """Tanh activation: f(x) = tanh(x)"""
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        out = np.tanh(x)
        self.cache['out'] = out
        return out
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        out = self.cache['out']
        dx = dout * (1 - out ** 2)
        return dx