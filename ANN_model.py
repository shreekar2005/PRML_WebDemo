import numpy as np
def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z)*(1-sigmoid(z))

def relu(z):
    return np.maximum(0, z)

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)  # numerical stability
    z = np.clip(z, -50, 50)  # limit extreme values
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def relu_derivative(z):
    return (z>0)*1

def linear(z):
    return z

def linear_derivative(z):
    return 1

def mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_true-y_pred))

def cross_entropy_loss(y_true, y_pred):
    eps = 1e-15  # avoid log(0)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def activation_decider(activation, z):
  if activation == 'sigmoid':
    return sigmoid(z)
  elif activation == 'relu':
    return relu(z)
  elif activation == 'linear':
    return linear(z)
  elif activation == 'softmax':
    return softmax(z)

def activation_derivative_decider(activation, z):
  if activation == 'sigmoid':
    return sigmoid_derivative(z)
  elif activation == 'relu':
    return relu_derivative(z)
  elif activation == 'linear':
    return linear_derivative(z)
  
class NeuralNetwork:

    def __init__(self, layers, activations):
        self.layers = layers
        self.activations = activations
        self.weights = []
        self.biases = []

        for i in range(len(layers) - 1):
            # He initialization for ReLU, Glorot otherwise
            if activations[i] == 'relu':
                stddev = np.sqrt(2 / layers[i])  # He initialization
            else:
                stddev = np.sqrt(1 / (layers[i] + layers[i + 1]))  # Glorot/Xavier
            self.weights.append(np.random.randn(layers[i], layers[i + 1]) * stddev)
            self.biases.append(np.zeros((1, layers[i + 1])))

    def forward(self, x):
        z_values = []
        activation_values = [x]

        for i in range(len(self.weights)):
            z = np.dot(x, self.weights[i]) + self.biases[i]
            z_values.append(z)
            x = activation_decider(self.activations[i], z)
            activation_values.append(x)

        return z_values, activation_values

    def backward(self, x, y, activations, z_values, learning_rate):
        m = x.shape[0]
        deltas = []

        # Output layer delta
        if self.activations[-1] == 'softmax':
            delta = activations[-1] - y  # softmax + cross-entropy
        else:
            delta = (activations[-1] - y) * activation_derivative_decider(self.activations[-1], z_values[-1])

        deltas.append(delta)

        # Backpropagate through hidden layers
        for i in reversed(range(len(self.weights) - 1)):
            delta = deltas[-1].dot(self.weights[i + 1].T) * activation_derivative_decider(self.activations[i], z_values[i])
            deltas.append(delta)

        deltas.reverse()

        # Update weights and biases
        for i in range(len(self.weights)):
            a_prev = x if i == 0 else activations[i]
            grad_w = a_prev.T.dot(deltas[i]) / m
            grad_b = np.mean(deltas[i], axis=0, keepdims=True)
            grad_w = np.clip(grad_w, -1.0, 1.0)
            grad_b = np.clip(grad_b, -1.0, 1.0)
            self.weights[i] -= learning_rate * grad_w
            self.biases[i] -= learning_rate * grad_b

    def train(self, x_train, y_train, epochs, learning_rate, x_test, y_test):
        train_acc = {}
        test_acc = {}
        losses = {}
        for epoch in range(epochs):
            z_values, activations = self.forward(x_train)
            self.backward(x_train, y_train, activations, z_values, learning_rate)
            if (epoch + 1)%100 == 0:
                loss = cross_entropy_loss(y_train, activations[-1])
                losses[epoch+1] = loss
                print(f"Epoch {epoch+1}: \nLoss = {loss:.6f}")
                predictions = self.predict(x_train)
                acc = np.mean(predictions == y_train.argmax(axis=1))
                train_acc[epoch+1] = acc
                print(f"Train Accuracy : {acc}")

                predictions = self.predict(x_test)
                acc = np.mean(predictions == y_test.argmax(axis=1))
                test_acc[epoch+1] = acc
                print(f"Test Accuracy : {acc}")

        return train_acc, test_acc, losses

    def predict(self, x):
        _, activations = self.forward(x)
        return np.argmax(activations[-1], axis=1)  # class prediction
