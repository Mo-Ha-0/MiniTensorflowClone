import numpy as np
import sys
sys.path.append('..')

from BeyondersTENSORFLOW import Dense, ReLU, SoftmaxCrossEntropy, NeuralNetwork, Trainer, SGD


def generate_data():
    np.random.seed(42)
    n_samples = 150
    X, y = [], []
    for class_idx in range(3):
        center = np.random.randn(2) * 2
        class_data = np.random.randn(n_samples // 3, 2) + center
        X.append(class_data)
        y.append(np.full(n_samples // 3, class_idx))
    # print(X)
    # print('-'*59)
    # print(y)
    X = np.vstack(X)
    y = np.concatenate(y)
    # print(X.shape)
    # print(X)
    # print('-'*59)
    # print(y.shape)
    # print(y)
    
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]
    # print(X)
    # print('-'*59)
    # print(y)

    split = int(0.8 * len(X))
    # print(split)
    return X[:split], X[split:], y[:split], y[split:]


def main():
    print("Simple MiniNN Example")
    print("=" * 50)
    
    X_train, X_test, y_train, y_test = generate_data()
    # print(X_train, y_train, sep='\n')
    # print("-" * 50)
    # print(X_test, y_test, sep='\n')
    print(f"Data: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    
    layers = [Dense(2, 10), ReLU(), Dense(10, 3), ReLU(), Dense(3, 3)]

    network = NeuralNetwork(layers, SoftmaxCrossEntropy())
    
    print("\nTraining...")
    trainer = Trainer(network, SGD(learning_rate=0.1))
    trainer.fit(X_train, y_train, X_test, y_test, epochs=500, batch_size=16, verbose = True, print_every=100)
    
    print(f"\nFinal accuracy: {network.accuracy(X_test, y_test):.4f}")


if __name__ == "__main__":
    main()