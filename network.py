import numpy as np
from typing import List, Tuple, Dict
from .base import Layer


class NeuralNetwork:
    
    def __init__(self, layers: List[Layer], loss_layer: Layer):
        self.layers = layers
        self.loss_layer = loss_layer
    
    def predict(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x, training=training)
        return x
    
    def loss(self, x: np.ndarray, y: np.ndarray, training: bool = True) -> float:
        pred = self.predict(x, training=training)
        loss = self.loss_layer.forward(pred, y, training=training)
        return loss
    
    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        pred = self.predict(x, training=False)
        pred_classes = np.argmax(pred, axis=1)
        
        if y.ndim == 1:
            true_classes = y
        else:
            true_classes = np.argmax(y, axis=1)
        
        accuracy = np.mean(pred_classes == true_classes)
        return accuracy
    
    def gradient(self) -> List[Tuple[Dict, Dict]]:
        dout = self.loss_layer.backward()
        
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        
        params_grads = []
        for layer in self.layers:
            if layer.params:
                params_grads.append((layer.params, layer.grads))
        
        return params_grads
    
    def init_weights(self, method: str = 'he'):
        from .layers.dense import Dense
        
        for layer in self.layers:
            if isinstance(layer, Dense):
                input_size, output_size = layer.params['W'].shape
                if method == 'he':
                    layer.params['W'] = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
                elif method == 'xavier':
                    layer.params['W'] = np.random.randn(input_size, output_size) * np.sqrt(1.0 / input_size)
                layer.params['b'] = np.zeros(output_size)