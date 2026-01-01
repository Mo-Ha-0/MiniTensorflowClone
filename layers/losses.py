import numpy as np
from ..base import Layer


class MeanSquaredError(Layer):
    
    def forward(self, x: np.ndarray, y: np.ndarray, training: bool = True) -> float:

        self.cache['x'] = x
        self.cache['y'] = y
        loss = np.mean(np.sum((x - y) ** 2, axis=1))
        return loss
    
    def backward(self, dout: float = 1.0) -> np.ndarray:
        x = self.cache['x']
        y = self.cache['y']
        N = x.shape[0]
        dx = 2 * (x - y) / N
        return dx


class SoftmaxCrossEntropy(Layer):
    
    def forward(self, x: np.ndarray, y: np.ndarray, training: bool = True) -> float:

        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        
        self.cache['probs'] = probs
        
        
        if y.ndim == 1:
            N = x.shape[0]
            loss = -np.sum(np.log(probs[np.arange(N), y] + 1e-8)) / N
            self.cache['y'] = y
            self.cache['one_hot'] = False
        else:
            loss = -np.mean(np.sum(y * np.log(probs + 1e-8), axis=1))
            self.cache['y'] = y
            self.cache['one_hot'] = True
        
        return loss
    
    def backward(self, dout: float = 1.0) -> np.ndarray:
        probs = self.cache['probs']
        y = self.cache['y']
        N = probs.shape[0]
        
        dx = probs.copy()
        if not self.cache['one_hot']:
            dx[np.arange(N), y] -= 1
        else:
            dx -= y
        
        dx = dx / N
        return dx