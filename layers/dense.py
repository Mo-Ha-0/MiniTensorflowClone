import numpy as np
from ..base import Layer


class Dense(Layer):
    """
    Fully connected layer: y = xW + b
    
    Forward: y = xW + b
    Backward: 
        - dL/dW = x^T * dout
        - dL/db = sum(dout)
        - dL/dx = dout * W^T
    """
    
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        
        self.params['W'] = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.params['b'] = np.zeros(output_size)
        
        self.grads['W'] = np.zeros_like(self.params['W'])
        self.grads['b'] = np.zeros_like(self.params['b'])
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self.cache['x'] = x
        out = np.dot(x, self.params['W']) + self.params['b']
        return out
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        x = self.cache['x']
        
        self.grads['W'] = np.dot(x.T, dout)
        self.grads['b'] = np.sum(dout, axis=0)
        dx = np.dot(dout, self.params['W'].T)
        
        return dx