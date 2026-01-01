import numpy as np
import sys
sys.path.append("..")

from BeyondersTENSORFLOW import (
    Dense, Sigmoid, ReLU, BatchNormalization,
    SoftmaxCrossEntropy, NeuralNetwork, Trainer, Adam
)


def main():
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split

    print("="*70)
    print(" " * 15 + "MiniNN - MNIST Dataset Example (784 features)")
    print("="*70)

    print("\nLoading MNIST dataset...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)


    n_samples = 10000
    indices = np.random.choice(X.shape[0], n_samples, replace=False)
    X = X[indices]
    y = y[indices]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    print(f"\nDataset loaded:")
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Features: {X_train.shape[1]} (784 = 28x28 pixels flattened), Classes: {len(np.unique(y))}")

    print("\nNetwork: Dense(784→256) → ReLU → BatchNorm → Dense(256→128) → ReLU → BatchNorm → Dense(128→10)")

    layers = [
        Dense(784, 256),
        ReLU(),
        BatchNormalization(256),
        Dense(256, 128),
        ReLU(),
        BatchNormalization(128),
        Dense(128, 10)
    ]

    network = NeuralNetwork(layers, SoftmaxCrossEntropy())
    optimizer = Adam(learning_rate=0.001)

    print("\nTraining...")
    print("-" * 70)
    trainer = Trainer(network, optimizer)
    trainer.fit(X_train, y_train, X_test, y_test, epochs=50, batch_size=64, verbose=True, print_every=10)

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
