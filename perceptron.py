import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.5, threshold=0):
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.weights = []

    def classify(self, x):
        return 1 if np.dot(x + [1], self.weights) > self.threshold else -1

    def train(self, Xs, ys, epochs=100):
        self.weights = np.random.uniform(-1, 1, len(Xs[0])+1)
        for _ in range(epochs):
            for i in range(len(Xs)):
                _y = self.classify(Xs[i])
                _x = Xs[i] + [1]
                err = ys[i] - _y
                self.weights = [self.learning_rate * err * _x[j]
                                + self.weights[j] for j in range(len(_x))]


if __name__ == '__main__':
    p = Perceptron(0.1)
    X = [[0,0],
         [0,1],
         [1,0],
         [1,1]]
    y = [-1, -1, -1, 1]
    p.train(X, y)
    for x in X:
        print p.classify(x)