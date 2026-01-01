import numpy as np
from abc import ABC, abstractmethod


class Layer(ABC):

    
    def __init__(self):
        self.params = {}
        self.grads = {}
        self.cache = {} 
    
    @abstractmethod
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        pass
    
    @abstractmethod
    def backward(self, dout: np.ndarray) -> np.ndarray:
        pass