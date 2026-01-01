import numpy as np
from abc import ABC, abstractmethod
from typing import Dict


class Optimizer(ABC):
    
    @abstractmethod
    def update(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        pass


class SGD(Optimizer):    
    def __init__(self, learning_rate: float = 0.01):
        self.lr = learning_rate
    
    def update(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Momentum(Optimizer):
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.v = {}
    
    def update(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        for key in params.keys():
            param_id = id(params[key])
            

            if param_id not in self.v:
                self.v[param_id] = np.zeros_like(params[key])
            
            self.v[param_id] = self.momentum * self.v[param_id] - self.lr * grads[key]
            params[key] += self.v[param_id]


class AdaGrad(Optimizer):
    
    def __init__(self, learning_rate: float = 0.01, eps: float = 1e-8):
        self.lr = learning_rate
        self.eps = eps
        self.h = {}
    
    def update(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        for key in params.keys():
            param_id = id(params[key])
            
            if param_id not in self.h:
                self.h[param_id] = np.zeros_like(params[key])
            
            self.h[param_id] += grads[key] ** 2
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[param_id]) + self.eps)


class Adam(Optimizer):

    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, eps: float = 1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {} 
        self.v = {}
        self.t = 0
    
    def update(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        self.t += 1
        
        for key in params.keys():
            
            param_id = id(params[key])
            
            if param_id not in self.m:
                self.m[param_id] = np.zeros_like(params[key])
                self.v[param_id] = np.zeros_like(params[key])
            
            self.m[param_id] = self.beta1 * self.m[param_id] + (1 - self.beta1) * grads[key]
            
            self.v[param_id] = self.beta2 * self.v[param_id] + (1 - self.beta2) * (grads[key] ** 2)
            
            m_hat = self.m[param_id] / (1 - self.beta1 ** self.t)
            
            v_hat = self.v[param_id] / (1 - self.beta2 ** self.t)
            
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)