import numpy as np
from typing import Dict, Callable
from .trainer import Trainer


class HyperparameterTuner:
    
    def __init__(self):
        self.results = []
    
    def grid_search(self, x_train: np.ndarray, y_train: np.ndarray,
                   x_val: np.ndarray, y_val: np.ndarray,
                   param_grid: Dict, network_builder: Callable,
                   epochs: int = 50, batch_size: int = 32) -> Dict:
        import itertools
        
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(itertools.product(*values))
        
        print(f"Testing {len(combinations)} combinations...")
        
        for i, combo in enumerate(combinations):
            params = dict(zip(keys, combo))
            print(f"\n[{i+1}/{len(combinations)}] Testing: {params}")
            
            network, optimizer = network_builder(params)
            trainer = Trainer(network, optimizer)
            
            trainer.fit(x_train, y_train, x_val, y_val,
                       epochs=epochs, batch_size=batch_size, verbose=False)
            
            result = {
                'params': params,
                'val_acc': trainer.val_acc_history[-1],
                'val_loss': trainer.val_loss_history[-1],
                'train_acc': trainer.train_acc_history[-1],
                'train_loss': trainer.train_loss_history[-1]
            }
            self.results.append(result)
            print(f"Val Acc: {result['val_acc']:.4f}")
        
        self.results.sort(key=lambda x: x['val_acc'], reverse=True)
        
        print("\n" + "="*60)
        print("BEST PARAMETERS:")
        for i, result in enumerate(self.results[:3], 1):
            print(f"{i}. {result['params']} - Acc: {result['val_acc']:.4f}")
        
        return self.results[0]['params']