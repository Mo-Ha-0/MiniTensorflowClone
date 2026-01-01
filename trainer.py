import numpy as np
from typing import Optional, Tuple
from .network import NeuralNetwork
from .optimizers.optimizers import Optimizer


class Trainer:
    
    def __init__(self, network: NeuralNetwork, optimizer: Optimizer):
        self.network = network
        self.optimizer = optimizer
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []
    
    def train_step(self, x_batch: np.ndarray, y_batch: np.ndarray) -> Tuple[float, float]:
        loss = self.network.loss(x_batch, y_batch, training=True)
        params_grads = self.network.gradient()
        
        for params, grads in params_grads:
            self.optimizer.update(params, grads)
        
        acc = self.network.accuracy(x_batch, y_batch)
        return loss, acc
    
    def fit(self, x_train: np.ndarray, y_train: np.ndarray,
            x_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            epochs: int = 100, batch_size: int = 32, verbose: bool = True, print_every: int = 10):
        n_samples = x_train.shape[0]
        n_batches = n_samples // batch_size
        
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            x_train_shuffled = x_train[indices]
            y_train_shuffled = y_train[indices]
            
            epoch_loss = 0
            epoch_acc = 0
            
            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size
                x_batch = x_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]
                
                loss, acc = self.train_step(x_batch, y_batch)
                epoch_loss += loss
                epoch_acc += acc
            
            epoch_loss /= n_batches
            epoch_acc /= n_batches
            
            self.train_loss_history.append(epoch_loss)
            self.train_acc_history.append(epoch_acc)
            
            if x_val is not None and y_val is not None:
                val_loss = self.network.loss(x_val, y_val, training=False)
                val_acc = self.network.accuracy(x_val, y_val)
                self.val_loss_history.append(val_loss)
                self.val_acc_history.append(val_acc)
                

                
                if verbose and (epoch + 1) % print_every == 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f} - "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            else:
                if verbose and (epoch + 1) % print_every == 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")