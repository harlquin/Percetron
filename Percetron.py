import numpy as np


class Perceptron(object):

    def __init__(self, no_of_inputs, threshold=100, learning_rate=0.01):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
            activation = 1
        else:
            activation = 0
        return activation

    def train(self, training_inputs, labels):
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)


training_inputs = []
training_inputs.append(np.array([90, 90, 90]))
training_inputs.append(np.array([90, 90, 50]))
training_inputs.append(np.array([50, 70, 70]))
training_inputs.append(np.array([60, 60, 70]))
training_inputs.append(np.array([70, 70, 60]))
training_inputs.append(np.array([80, 50, 60]))

labels = np.array([0, 1, 1, 0, 0, 1])

perceptron = Perceptron(3)
perceptron.train(training_inputs, labels)

inputs = np.array([0.7, 0.6])
print(perceptron.predict(inputs))
if perceptron.predict(inputs) == 0:
    print("Diterima")
else:
    print("Tidak Diterima")
# => 1