import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from torchvision.datasets import EMNIST
from torch.utils.data import DataLoader
import torch.nn as nn
import pickle
import os, sys
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import seaborn as sb
import matplotlib.pyplot as plt

layers = []
learning_rate = None

EPSILON = 1e-8
np.random.seed(1)

class DenseLayer:
    def __init__(self, input_size, output_size, beta1=0.9, beta2=0.999, optimizer=False):
        '''
            This factor, np.sqrt(6/ (input_size + output_size)), is a common heuristic for initializing 
            weights in neural networks. The purpose of this initialization scheme is to prevent the 
            weights from becoming too large or too small, which could lead to issues like vanishing 
            or exploding gradients during training. It aims to keep the variance of activations and 
            gradients relatively consistent across layers. 
        '''
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(6 / (input_size + output_size))
        self.bias = np.random.randn(output_size) * np.sqrt(6 / (input_size + output_size))
        self.inputs = None
        self.gradient_weights = None
        self.gradient_bias = None
        self.decay = 0.01
        self.learning_rate = None
        self.optimizer = optimizer
        
        # Adam parameters
        self.m_weights = np.zeros((input_size, output_size))
        self.v_weights = np.zeros((input_size, output_size))
        self.m_bias = np.zeros(output_size)
        self.v_bias = np.zeros(output_size)
        self.beta1 = beta1
        self.beta2 = beta2
        self.timestep = 0
    
    def forward(self, inputs):
        self.inputs = inputs
        if self.optimizer:
            self.timestep += 1
        return np.dot(inputs, self.weights) + self.bias

class Sigmoid:
    def __init__(self):
        self.output = None

    def forward(self, input):
        self.output = 1 / (1 + np.exp(-input))
        return self.output

class ReLULayer:
    def __init__(self):
        self.inputs = None
        
    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)


class DropoutLayer:
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.mask = None
    
    def forward(self, inputs, is_training=True):
        if is_training:
            self.mask = np.random.rand(*inputs.shape) < self.dropout_rate
            return inputs * self.mask / self.dropout_rate
        else:
            return inputs


class SoftmaxLayer:
    def __init__(self):
        self.output = None
        
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output


class NeuralNetwork:
    def __init__(self, layers, learning_rate):
        self.layers = layers
        self.learning_rate = learning_rate
    
    def forward_pass(self, X):
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                layer.learning_rate = self.learning_rate
            X = layer.forward(X)
            if isinstance(layer, DropoutLayer):
                layer.inputs = X
        return X
    
    def cross_entropy_loss(self, one_hot_labels, outputs):
        loss = -np.sum(one_hot_labels * np.log(np.clip(outputs, EPSILON, 1.0 - EPSILON))) / len(one_hot_labels)
        return loss

    def predict(self, X):
        return np.argmax(self.forward_pass(X), axis=1)
    
def getReportData(model, X_test, y_test):
    test_loss = model.cross_entropy_loss(y_test, model.forward_pass(X_test))
    test_accuracy = np.mean(model.predict(X_test) == np.argmax(y_test, axis=1)) * 100
    print(f"Test Data Loss: {test_loss:.4f}")
    print(f"Test Data Accuracy: {test_accuracy:.4f}%")
    
    # Calculate confusion matrix
    confusion_matrix = np.zeros((26, 26))
    y_pred = model.predict(X_test)
    y_true = np.argmax(y_test, axis=1)
    for act, pred in zip(y_true, y_pred):
        confusion_matrix[act][pred] += 1
        
    # print(confusion_matrix)
    # Calculate f1 score
    macro_f1_score = 0
    for i in range(26):
        precision = confusion_matrix[i][i] / max(1, np.sum(confusion_matrix[i]))
        recall = confusion_matrix[i][i] / max(1, np.sum(confusion_matrix[:, i]))
        macro_f1_score += 2 * precision * recall / max(1, (precision + recall))
    macro_f1_score /= 26
    print("MacroF1: ", macro_f1_score)
    print("======================\n\n\n")
    
def load(path):
    global layers, learning_rate
    with open(path, "rb") as file:
        model = pickle.load(file)
    layers = model.layers
    learning_rate = model.learning_rate
    print("Model loaded from: ", path)
    return model
    
def getTrainAndValidationData():
    # Load the EMNIST dataset
    emnist_test = EMNIST(root='./data', split='letters', train=False, transform=transforms.ToTensor(), download=True)

    X_test = emnist_test.data.numpy() / 255.0
    y_test = emnist_test.targets.numpy().astype(int)

    X_test = X_test.reshape(X_test.shape[0], -1)
    y_test = y_test.reshape(y_test.shape[0], -1)
    
    num_classes = 27
    # Remove the first column from each row
    y_test_onehot = np.eye(num_classes)[y_test][:, :, 1:]

    # Flatten one-hot encoded labels
    y_test = y_test_onehot.reshape(y_test_onehot.shape[0], -1)
    
    return X_test, y_test

X_test, y_test = getTrainAndValidationData()
model = load("model_1805088.pkl")
getReportData(model, X_test, y_test)