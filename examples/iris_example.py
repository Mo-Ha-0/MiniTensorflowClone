import numpy as np
import sys
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
sys.path.append('..')

from BeyondersTENSORFLOW import (
    Dense, Sigmoid, ReLU, BatchNormalization,
    SoftmaxCrossEntropy, NeuralNetwork, Trainer, Adam
)


def main():

    
    print("="*70)
    print(" " * 15 + "MiniNN - Iris Dataset Example")
    print("="*70)
    
    iris = load_iris()
    X, y = iris.data, iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"\nDataset loaded:")
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Features: {X_train.shape[1]}, Classes: {len(np.unique(y))}")
    
    print("\nNetwork: Dense(4→16) → Sigmoid → BatchNorm → Dense(16→8) → ReLU → Dense(8→3)")
    
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
    
    print("\nTraining...")
    print("-" * 70)
    trainer = Trainer(network, optimizer)
    trainer.fit(X_train, y_train, X_test, y_test, epochs=2340, batch_size=16, verbose=True, print_every=260)
    
    print("\n" + "="*70)
    print("FINAL RESULTS:")
    print("="*70)
    train_acc = network.accuracy(X_train, y_train)
    test_acc = network.accuracy(X_test, y_test)
    print(f"Training Accuracy: {train_acc:.4f} ({train_acc*100:.1f}%)")
    print(f"Test Accuracy:     {test_acc:.4f} ({test_acc*100:.1f}%)")
    print("\n✓ Complete!")


if __name__ == "__main__":
    main()