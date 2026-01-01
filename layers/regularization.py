import numpy as np
from ..base import Layer


class Dropout(Layer):
    
    def __init__(self, drop_rate: float = 0.5):
        super().__init__()
        self.drop_rate = drop_rate
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            mask = np.random.rand(*x.shape) > self.drop_rate
            self.cache['mask'] = mask
            return x * mask / (1 - self.drop_rate)
        else:
            return x
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        mask = self.cache['mask']
        return dout * mask / (1 - self.drop_rate)


class BatchNormalization(Layer):
    
    def __init__(self, num_features: int, momentum: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        
        self.params['gamma'] = np.ones(num_features)
        self.params['beta'] = np.zeros(num_features)
        self.grads['gamma'] = np.zeros(num_features)
        self.grads['beta'] = np.zeros(num_features)
        
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            
            x_centered = x - mean
            std = np.sqrt(var + self.eps)
            x_norm = x_centered / std
            
            self.cache['x_centered'] = x_centered
            self.cache['std'] = std
            self.cache['x_norm'] = x_norm
        else:
            
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
        
        out = self.params['gamma'] * x_norm + self.params['beta']
        return out
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        x_centered = self.cache['x_centered']
        std = self.cache['std']
        x_norm = self.cache['x_norm']
        N = dout.shape[0]
        
        self.grads['gamma'] = np.sum(dout * x_norm, axis=0)
        self.grads['beta'] = np.sum(dout, axis=0)
        
        dx_norm = dout * self.params['gamma']
        dvar = np.sum(dx_norm * x_centered * -0.5 * std**(-3), axis=0)
        dmean = np.sum(dx_norm * -1/std, axis=0) + dvar * np.mean(-2 * x_centered, axis=0)
        dx = dx_norm / std + dvar * 2 * x_centered / N + dmean / N
        
        return dx