import numpy as np


class Perceptron:

    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.bias = 0
        self.weights = []

    def feed_forward(self, x):
        value = self.bias
        for i in range(len(x)):
            value += x[i] * self.weights[i]
        if value > 0:
            return 1
        else:
            return -1

    def train(self, x, y, epochs=10, show_weights=False):
        vec_len = len(x[0])
        self.weights = [0] * vec_len

        for _ in range(epochs):
            for i in range(len(x)):
                yp = self.feed_forward(x[i])
                err = y[i] - yp
                self.bias += self.learning_rate * err
                for j in range(vec_len):
                    self.weights[j] += self.learning_rate * err * x[i][j]

        if show_weights:
            print self.weights, self.bias

    def classify(self, x, labels=[0,1]):
        y = []
        for i in x:
            result = self.feed_forward(i)
            if result == 1:
                y.append(labels[1])
            else:
                y.append(labels[0])
        return y


def accuracy(expected, calculated):
    return np.mean(np.array(expected) == np.array(calculated))


if __name__ == '__main__':
    p = Perceptron()
    # x = [[0,0], [0,1], [1,0], [1,1]]
    # y_and = [-1, -1, -1, 1]
    # y_or = [-1, 1, 1, 1]
    # p.train(x, y_and)
    # print p.classify(x)
    # p.train(x, y_or)
    # print p.classify(x)

    x, y, y_label = [], [], []

    f = open('ionosphere.data')
    for line in f.readlines():
        vals = line.rstrip().split(',')
        x.append(map(float, vals[:-1]))
        y_label.append(vals[-1])
        if vals[-1] == 'b':
            y.append(-1)
        else:
            y.append(1)
    f.close()

    p.train(x, y, 10)
    y_calc = p.classify(x, ['b','g'])
    print 'Accuracy:', accuracy(y_label, y_calc)