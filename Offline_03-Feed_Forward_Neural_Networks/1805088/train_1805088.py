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
        return np.dot(inputs, self.weights) + self.bias
    
    def backward(self, gradient_output):
        if self.optimizer:
            self.timestep += 1
        self.gradient_weights = np.dot(self.inputs.T, gradient_output)
        self.gradient_bias = np.sum(gradient_output, axis=0, keepdims=True)
        self.learning_rate = self.learning_rate * (1-self.decay)
        return np.dot(gradient_output, self.weights.T)

    def update_weights(self):
        if self.optimizer:
            self.m_weights = self.beta1 * self.m_weights + (1 - self.beta1) * self.gradient_weights
            self.v_weights = self.beta2 * self.v_weights + (1 - self.beta2) * (self.gradient_weights ** 2)
            m_hat_weights = self.m_weights / (1 - np.power(self.beta1, self.timestep))
            v_hat_weights = self.v_weights / (1 - np.power(self.beta2, self.timestep))
            self.weights = self.weights - self.learning_rate * m_hat_weights / (np.sqrt(v_hat_weights) + EPSILON)
            
            self.m_bias = self.beta1 * self.m_bias + (1 - self.beta1) * self.gradient_bias
            self.v_bias = self.beta2 * self.v_bias + (1 - self.beta2) * (self.gradient_bias ** 2)
            m_hat_bias = self.m_bias / (1 - np.power(self.beta1, self.timestep))
            v_hat_bias = self.v_bias / (1 - np.power(self.beta2, self.timestep))
            self.bias = self.bias - self.learning_rate * m_hat_bias / (np.sqrt(v_hat_bias) + EPSILON)
        else:
            self.weights = self.weights - self.learning_rate * self.gradient_weights
            self.bias = self.bias - self.learning_rate * self.gradient_bias
            
    def getLayerName(self):
        if self.optimizer:
            return "Dense_Adam_" + str(self.weights.shape[0]) + "_" + str(self.weights.shape[1]) + "-"
        else:
            return "Dense_" + str(self.weights.shape[0]) + "_" + str(self.weights.shape[1]) + "-"
            

class Sigmoid:
    def __init__(self):
        self.output = None

    def forward(self, input):
        self.output = 1 / (1 + np.exp(-input))
        return self.output
    
    def backward(self, gradient_output):
        return self.output * (1 - self.output) * gradient_output
    
    def getLayerName(self):
        return "Sigmoid-"
    

class ReLULayer:
    def __init__(self):
        self.inputs = None
        
    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)
    
    def backward(self, gradient_output):
        # return gradient_output * (self.inputs > 0)
        return gradient_output * (self.forward(self.inputs) > 0)
    
    def getLayerName(self):
        return "ReLU-"


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
    
    def backward(self, gradient_output):
        return gradient_output * self.mask
    
    def getLayerName(self):
        return "DRopout_" + str(self.dropout_rate) + "-"


class SoftmaxLayer:
    def __init__(self):
        self.output = None
        
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output
    
    def backward(self, y):
        return self.output - y
    
    def getLayerName(self):
        return "SOftmax"


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
    
    def backward_pass(self, gradient):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient
    
    def update_weights(self):
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                layer.update_weights()
    
    
    def cross_entropy_loss(self, one_hot_labels, outputs):
        loss = -np.sum(one_hot_labels * np.log(np.clip(outputs, EPSILON, 1.0 - EPSILON))) / len(one_hot_labels)
        return loss
    
    def train(self, X_train, X_val, y_train, y_val, epochs, batch_size, modelName):
        best_epoch = 0
        best_validation_accuracy = 0
        epochs_list = range(1, epochs+1)
        train_loss_list = []
        validation_loss_list = []
        train_accuracy_list = []
        validation_accuracy_list = []
        f1_score_list = []
        for epoch in range(epochs):
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                self.forward_pass(X_batch)
                self.backward_pass(y_batch)
                self.update_weights()
                
            # Print loss at the end of each epoch
            train_loss = self.cross_entropy_loss(y_train, self.forward_pass(X_train))
            train_accuracy = np.mean(self.predict(X_train) == np.argmax(y_train, axis=1)) * 100
            validation_loss = self.cross_entropy_loss(y_val, self.forward_pass(X_val))
            validation_accuracy = np.mean(self.predict(X_val) == np.argmax(y_val, axis=1)) * 100
            f1_score_val = f1_score(np.argmax(y_val, axis=1), self.predict(X_val), average='macro')
            
            train_loss_list.append(train_loss)
            validation_loss_list.append(validation_loss)
            train_accuracy_list.append(train_accuracy)
            validation_accuracy_list.append(validation_accuracy)
            f1_score_list.append(f1_score_val)
            
            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                best_epoch = epoch
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Training Data Loss: {train_loss:.4f}")
            print(f"Training Data Accuracy: {train_accuracy:.4f}%")
            print(f"Validation Data Loss: {validation_loss:.4f}")
            print(f"Validation Data Accuracy: {validation_accuracy:.4f}%")
            print("MacroF1: ", f1_score_val)
            print("\n\n")
            sys.stdout.flush()
        
        plotGraph(epochs_list, train_loss_list, validation_loss_list, "Train and Validation Loss", 
                  "Train Loss", "Validation Loss", "Epochs", "Loss", modelName+"_loss.png")
        plotGraph(epochs_list, train_accuracy_list, validation_accuracy_list, "Train and Validation Accuracy", 
                  "Train Accuracy", "Validation Accuracy", "Epochs", "Accuracy", modelName+"_accuracy.png")
        plotGraph(epochs_list, f1_score_list, f1_score_list, "F1 Macro Score", 
                  "Validation F1", "Validation F1", "Epochs", "F1 Score", modelName+"_f1.png")
        
        return best_validation_accuracy, best_epoch

    def predict(self, X):
        return np.argmax(self.forward_pass(X), axis=1)
    
    def getNetworkModelName(self):
        networkName = ""
        for layer in self.layers:
            networkName += layer.getLayerName()
        return networkName + "_" + str(self.learning_rate)
    
    def dumpPickleFile(self):
        path = ''.join([char for char in self.getNetworkModelName() if not char.islower()]) + ".pkl"

        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                layer.inputs = None
                layer.m_weights = None
                layer.v_weights = None
                layer.m_bias = None
                layer.v_bias = None
                layer.timestep = None
            elif isinstance(layer, ReLULayer):
                layer.inputs = None
            elif isinstance(layer, Sigmoid) or isinstance(layer, SoftmaxLayer):
                layer.output = None   

        with open(path, "wb") as file:
            pickle.dump(self, file)
            print("Pickle file saved to: ", path)


def one_hot_labels(labels, output_size):
    labels = labels - 1 # EMNIST are numbered from 1 to 26. So one hot from 0 to 25        
    one_hot_labels = np.zeros((len(labels), output_size))
    for i, label in enumerate(labels):
        one_hot_labels[i, label] = 1  # Assuming labels range from 0 to 25
    return one_hot_labels

def getTrainAndValidationData():
    # Load the EMNIST dataset
    emnist_train = EMNIST(root='./data', split='letters', train=True, transform=transforms.ToTensor(), download=True)
    emnist_test = EMNIST(root='./data', split='letters', train=False, transform=transforms.ToTensor(), download=True)

    X_train = emnist_train.data.numpy() / 255.0
    X_test = emnist_test.data.numpy() / 255.0
    
    y_train = emnist_train.targets.numpy().astype(int)
    y_test = emnist_test.targets.numpy().astype(int)

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    y_train = y_train.reshape(y_train.shape[0], -1)
    y_test = y_test.reshape(y_test.shape[0], -1)
    
    num_classes = 27
    # Remove the first column from each row
    y_train_onehot = np.eye(num_classes)[y_train][:, :, 1:]
    y_test_onehot = np.eye(num_classes)[y_test][:, :, 1:]

    # Flatten one-hot encoded labels
    y_train = y_train_onehot.reshape(y_train_onehot.shape[0], -1)
    y_test = y_test_onehot.reshape(y_test_onehot.shape[0], -1)
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
    return X_train, X_val, y_train, y_val, X_test, y_test

def getAccF1ConfMat(model, y_true, X, dataset_name):
    print(dataset_name+": ")
    # test_loss = model.cross_entropy_loss(y_test, model.forward_pass(X_test))
    # test_accuracy = np.mean(model.predict(X_test) == np.argmax(y_test, axis=1)) * 100
    # print(f"Test Data Loss: {test_loss:.4f}")
    # print(f"Test Data Accuracy: {test_accuracy:.4f}%")
    
    # # Calculate confusion matrix
    # confusion_matrix = np.zeros((output_size, output_size))
    # y_pred = model.predict(X_test)
    # y_true = np.argmax(y_test, axis=1)
    # for act, pred in zip(y_true, y_pred):
    #     confusion_matrix[act][pred] += 1
        
    # # print(confusion_matrix)
    # # Calculate f1 score
    # macro_f1_score = 0
    # for i in range(26):
    #     precision = confusion_matrix[i][i] / max(1, np.sum(confusion_matrix[i]))
    #     recall = confusion_matrix[i][i] / max(1, np.sum(confusion_matrix[:, i]))
    #     macro_f1_score += 2 * precision * recall / max(1, (precision + recall))
    # macro_f1_score /= 26
    # print("MacroF1: ", macro_f1_score)
    # print("======================\n\n\n")
    
    print(f"CrossEntropy ({dataset_name}): {model.cross_entropy_loss(y_true, model.forward_pass(X)):.4f}")
    y_true = np.argmax(y_true, axis=1)
    y_pred = model.predict(X)
    print(f"Accuracy ({dataset_name}): {accuracy_score(y_true, y_pred)*100:.4f}%")
    print("MacroF1 ("+dataset_name+"): ", f1_score(y_true, y_pred, average='macro'))
    return confusion_matrix(y_true, y_pred)

def plotConfusionMatrix(conf_matrix, modelName):
    plt.clf()
    plt.figure(figsize=(14, 14))
    sb.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    
    plt.title('Confusion Matrix-'+ str(modelName))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(modelName+".png")
    
def plotGraph(x, y1, y2, graphName, y1_label, y2_label, x_label, y_label, figName):
    plt.clf()
    plt.plot(x, y1, 'g', label=y1_label)
    plt.plot(x, y2, 'b', label=y2_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='center right')
    plt.title(graphName)
    plt.xlim(max(0, min(x) - min(x)/10), max(x) + max(x)/20)
    plt.ylim(max(0, min(min(y1), min(y2)) - min(min(y1), min(y2)) / 10), max(max(y1), max(y2)) + max(max(y1), max(y2)) / 10)
    plt.savefig(figName)
    
def getConfusionMatrix(modelName, conf_mat_train, conf_mat_valid, conf_mat_test):
    plotConfusionMatrix(conf_mat_train, modelName + "TrainDataset")
    plotConfusionMatrix(conf_mat_valid, modelName + "ValidationDataset")
    plotConfusionMatrix(conf_mat_test, modelName + "TestDataset")

def getReportData(model, epochs, batch_size, modelName):
    X_train, X_val, y_train, y_val, X_test, y_test = getTrainAndValidationData()
    best_validation_accuray, best_epoch = model.train(X_train, X_val, y_train, y_val, epochs, batch_size, modelName)

    conf_mat_train = getAccF1ConfMat(model, y_train, X_train, "Train Dataset")
    conf_mat_valid = getAccF1ConfMat(model, y_val, X_val, "Validation Dataset")
    conf_mat_test = getAccF1ConfMat(model, y_test, X_test, "Test dataset")
    
    print("Best accuracy for ", modelName, " :" , best_validation_accuray)
    return conf_mat_train, conf_mat_valid, conf_mat_test

def handleFNNModel(model, epochs, batch_size):
    modelName = model.getNetworkModelName()
    modelName = ''.join(filter(lambda char: not char.islower(), modelName))
    # if such file exists then skip
    if os.path.isfile(modelName  + ".txt"):
        print(modelName, " already exists")
        return
    sys.stdout = open(modelName + ".txt", "w")
    conf_mat_train, conf_mat_valid, conf_mat_test = getReportData(model, epochs, batch_size, modelName)
    getConfusionMatrix(modelName, conf_mat_train, conf_mat_valid, conf_mat_test)
    try:
        sys.stdout.close()
    finally:
        # Reset sys.stdout to sys.__stdout__
        sys.stdout = sys.__stdout__
    print(modelName, " created successfully")
    model.dumpPickleFile()

def getReport():
    # Define the neural network architecture
    input_size = 28 * 28  # Size of EMNIST images
    hidden_size = 128
    output_size = 26  # Number of classes (letters)

    # Hyperparameters
    learning_rates = [0.0005, 0.005, 0.01, 0.02]
    epochs = 60
    batch_size = 64
    
    # Create the neural network architecture
    layers = [
        # [DenseLayer(input_size, hidden_size), Sigmoid(), DenseLayer(hidden_size, output_size), SoftmaxLayer()],
        [DenseLayer(input_size, hidden_size), ReLULayer(), DenseLayer(hidden_size, output_size), SoftmaxLayer()],
        [DenseLayer(input_size, hidden_size), Sigmoid(), DropoutLayer(0.75), DenseLayer(hidden_size, output_size), SoftmaxLayer()],
        # [DenseLayer(input_size, hidden_size), ReLULayer(), DropoutLayer(0.75), DenseLayer(hidden_size, output_size), SoftmaxLayer()],
        [DenseLayer(input_size, hidden_size, optimizer=True), Sigmoid(), DenseLayer(hidden_size, output_size, optimizer=True), SoftmaxLayer()],
        # [DenseLayer(input_size, hidden_size, optimizer=True), ReLULayer(), DenseLayer(hidden_size, output_size, optimizer=True), SoftmaxLayer()],
    ]
    
    for rate in learning_rates:
        for i in range(len(layers)):
            model = NeuralNetwork(layers[i], rate)
            handleFNNModel(model, epochs, batch_size)

def test():
    model = NeuralNetwork([DenseLayer(28*28, 128), ReLULayer(), DenseLayer(128, 26), SoftmaxLayer()], 0.0005)
    handleFNNModel(model, 5, 64)

if __name__ == "__main__":
    # getReport()    
    test()