import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.5, threshold=0, bias=0):
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.bias = bias
        self.weights = []

    def classify(self, X, labels = [0, 1]):
        y = []
        for Xi in X:
            result = self.feed_forward(Xi)
            if result == 1:
                y.append(labels[1])
            else:
                y.append(labels[0])
        return y

    def feed_forward(self, X):
        value = self.bias
        for i in range(len(X)):
            value += X[i] * self.weights[i]
        if value > self.threshold:
            return 1
        else:
            return -1

    def train(self, X, y, epochs=10):
        vec_len = len(X[0])
        self.weights = np.random.uniform(-1, 1, vec_len)
        for _ in range(epochs):
            for i in range(len(X)):
                yp = self.feed_forward(X[i])
                err = y[i] - yp
                self.bias += self.learning_rate * err
                for j in range(vec_len):
                    self.weights[j] += self.learning_rate * err * X[i][j]


if __name__ == '__main__':
    p = Perceptron(0.1)
    X = [[0,0],
         [0,1],
         [1,0],
         [1,1]]
    y = [-1, -1, -1, 1]
    p.train(X, y)
    print p.classify(X)