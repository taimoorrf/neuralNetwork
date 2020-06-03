# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 23:00:51 2019

@author: YourAverageSciencePal
"""
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
from PIL import Image
import os
import os.path as path
import random
import pickle
# import sklearn
'''
Depending on your choice of library you have to install that library using pip
'''


'''
Read chapter on neural network from the book. Most of the derivatives,formulas 
are already there.
Before starting this assignment. Familarize yourselves with np.dot(),
What is meant by a*b in 2 numpy arrays.
What is difference between np.matmul and a*b and np.dot.
Numpy already has vectorized functions for addition and subtraction and even for division
For transpose just do a.T where a is a numpy array 
Also search how to call a static method in a class.
If there is some error. You will get error in shapes dimensions not matched
because a*b !=b*a in matrices
'''

class NeuralNetwork():
    @staticmethod
    #note the self argument is missing i.e. why you have to search how to use static methods/functions
    def cross_entropy_loss(y_pred, y_true):
        '''implement cross_entropy loss error function here
        Hint: Numpy has a sum function already
        Numpy has also a log function
        Remember loss is a number so if y_pred and y_true are arrays you have to sum them in the end
        after calculating -[y_true*log(y_pred)]'''
        y_pred_final = np.log(y_pred)
        cross_entropy_loss = -(y_true * y_pred_final).sum()
        return cross_entropy_loss
    @staticmethod
    def accuracy(y_pred, y_true):
        '''function to calculate accuracy of the two lists/arrays
        Accuracy = (number of same elements at same position in both arrays)/total length of any array
        Ex-> y_pred = np.array([1,2,3]) y_true=np.array([1,2,4]) Accuracy = 2/3*100 (2 Matches and 1 Mismatch)'''
        acc = 0
        for i in range(len(y_pred)):
            if np.argmax(y_pred[i]) == np.argmax(y_true[i]):
                acc += 1
        acc = acc/len(y_pred)
        acc = acc * 100
        return acc

    @staticmethod
    def softmax(x):
        '''Implement the softmax function using numpy here
        Hint: Numpy sum has a parameter axis to sum across row or column. You have to use that
        Use keepdims=True for broadcasting
        You guys should have a pretty good idea what the size of returned value is.
        '''
        exp = np.exp(x)
        softmax = exp/(exp.sum(axis=1, keepdims=True))
        return softmax
    
    @staticmethod
    def sigmoid(x):
        '''Implement the sigmoid function using numpy here
        Sigmoid function is 1/(1+e^(-x))
        Numpy even has a exp function search for it.Eh?
        '''
        sigmoid = 1/(1 + np.exp(-x))
        return sigmoid
   
    
    def __init__(self):
        '''Creates a Feed-Forward Neural Network.
        "nodes_per_layer" is a list containing number of nodes in each layer (including input layer)
        "num_layers" is the number of layers in your network 
        "input_shape" is the shape of the image you are feeding to the network
        "output_shape" is the number of probabilities you are expecting from your network'''

        self.num_layers = 3 # includes input layer
        self.nodes_per_layer = [784, 30, 10]
        self.input_shape = 784
        self.output_shape = 10
        self.__init_weights(self.nodes_per_layer)
        self.history = None
        self.times = []
    def __init_weights(self, nodes_per_layer):
        '''Initializes all weights and biases between -1 and 1 using numpy'''
        self.weights_ = []
        self.biases_ = []
        for i,_ in enumerate(nodes_per_layer):
            if i == 0:
                # skip input layer, it does not have weights/bias
                continue
            weight_matrix = np.random.normal(size=(self.nodes_per_layer[i-1], self.nodes_per_layer[i]))
            self.weights_.append(weight_matrix)
            bias_vector = np.zeros(shape=(self.nodes_per_layer[i],))
            self.biases_.append(bias_vector)
    
    def fit(self, Xs, Ys, epochs, lr=1e-3):
        '''Trains the model on the given dataset for "epoch" number of itterations with step size="lr". 
        Returns list containing loss for each epoch.'''
        history = []
        for i in range(epochs):
            print('Epoch:', i + 1)
            for j in range(Xs.shape[0]):
                inputs = Xs[j,:]                                        #Taking column 
                input_vals = inputs.reshape((1, self.input_shape))      #Converting to row form
                target = Ys[j, :]
                target_vals = target.reshape((1,self.output_shape))      #Converting to row form
                ##Calling forward pass on the input data
                activations = self.forward_pass(input_vals)
                ##Calling backward pass on the input data
                deltas = self.backward_pass(target_vals, activations)

                input_and_activations = [input_vals] + activations[: -1]
                self.weight_update(deltas, input_and_activations, lr)
            
            loss, acc = self.evaluate(Xs, Ys)
            # print('Epoch: ', i + 1)
            print('Accuracy: ', acc, '%')
            print('Error: ', 100 - acc, '%')
            history.append(loss)
        return history

    
    def forward_pass(self, input_data):
        '''Executes the feed forward algorithm.
        "input_data" is the input to the network in row-major form
        Returns "activations", which is a list of all layer outputs (excluding input layer of course)
        What is activation?
        In neural network you have inputs(x) and weights(w).
        What is first layer? It is your input right?
        A linear neuron is this: y = w.T*x+b =>T is the transpose operator 
        A sigmoid neuron activation is y = sigmoid(w1.T*x+b1) for 1st hidden layer 
        Now for the last hidden layer the activation y = sigmoid(w2.T*y+b2).
        '''
        #Multiplying the weights by the input data then adding the biases for both the hidden and the output layer
        hidd = np.dot(input_data, self.weights_[0]) 
        hidden_layer_activations = self.sigmoid(hidd + self.biases_[0])     #We will use sigmoid for hidden layer activation because sigmoid binary classifier
        out = np.dot(hidden_layer_activations, self.weights_[1]) 
        output_layer_activations = self.sigmoid(out + self.biases_[1])      
        activations = [hidden_layer_activations, output_layer_activations]
        return activations

    
    def backward_pass(self, targets, layer_activations):
        '''Executes the backpropogation algorithm.
        "targets" is the ground truth/labels
        "layer_activations" are the return value of the forward pass step
        Returns "deltas", which is a list containing weight update values for all layers (excluding the input layer of course)
        You need to work on the paper to develop a generalized formulae before implementing this.
        Chain rule and derivatives are the pre-requisite for this part.
        '''

        deltas = []
        output_activations = layer_activations[1]
        output_activation_deriv = output_activations * (1 - output_activations)
        error_deriv = output_activation_deriv - targets
        output_deltas = output_activation_deriv * error_deriv

        hidden_layer_activations = layer_activations[0]
        hidden_activation_deriv = hidden_layer_activations * (1 - hidden_layer_activations)
        hidden_layer_deltas = hidden_activation_deriv * np.matmul(output_deltas, self.weights_[1].T)
        deltas = [hidden_layer_deltas, output_deltas]
        return deltas
            
    def weight_update(self, deltas, layer_inputs, lr):
        '''Executes the gradient descent algorithm.
        "deltas" is return value of the backward pass step
        "layer_inputs" is a list containing the inputs for all layers (including the input layer)
        "lr" is the learning rate
        You just have to implement the simple weight update equation. 
        
        '''
        self.weights_[0] = self.weights_[0] - lr * np.dot(layer_inputs[0].T, deltas[0])
        self.biases_[0] = self.biases_[0] - lr * deltas[0].sum(axis = 0)
        self.weights_[1] = self.weights_[1] - lr * np.dot(layer_inputs[1].T, deltas[1])
        self.biases_[1] = self.biases_[1] - lr * deltas[1].sum(axis = 0)
       
        
    def predict(self, Xs):
        '''Returns the model predictions (output of the last layer) for the given "Xs".'''
        predictions = []
        for i in range(Xs.shape[0]):
            inputs = Xs[i,:]
            inputs_final = inputs.reshape((1, self.input_shape))        #Reshaping taken sample row
            inputs_forward_pass = self.forward_pass(inputs_final)       #Applying forward pass on the inputs
            last_layer = inputs_forward_pass[-1]                        #Taking the last (Output) layer after applying forward pass
            last_layer_final = last_layer.reshape((self.output_shape,)) #Reshaping the last layer to row
            predictions.append(last_layer_final)                        #Appending the last later to predictions
        predictions_final = np.array(predictions)
        return predictions_final
    
    def evaluate(self, Xs, Ys):
        '''Returns appropriate metrics for the task, calculated on the dataset passed to this method.'''
        pred = self.predict(Xs)
        acc = self.accuracy(pred, Ys) 
        loss = self.cross_entropy_loss(pred, Ys)
        return loss, acc

    
    def normalize_image(self, image):
        return (image - np.mean(image))/np.std(image)
    
    def give_images(self,listDirImages):
        '''Returns the images and labels from the listDirImages list after reading
        Hint: Use os.listdir(),os.getcwd() functions to get list of all directories
        in the provided folder. Similarly os.getcwd() returns you the current working
        directory. 
        For image reading use any library of your choice. Commonly used are opencv,pillow but
        you have to install them using pip
        "images" is list of numpy array of images 
        labels is a list of labels you read 
        '''

        images = []
        labels = []

        main_directory = path.join(os.getcwd(), listDirImages)
        for sub_directory in os.listdir(main_directory):
            if "." not in sub_directory:
                for file in os.listdir(path.join(main_directory, sub_directory)):
                    images.append(self.normalize_image((np.asarray(Image.open(path.join(main_directory, sub_directory, file), 'r')).flatten()) - 1))
                    labels.append(int(sub_directory))
                        
        shuffler = list(zip( np.asarray(images), self.generate_labels(labels) ))
        random.shuffle(shuffler)
        images, labels = zip(*shuffler)
        return np.array(images), np.array(labels)

    def generate_labels(self,labels):
        '''Returns your labels into one hot encoding array
        labels is a list of labels [0,1,2,3,4,1,3,3,4,1........]
        Ex-> If label is 1 then one hot encoding should be [0,1,0,0,0,0,0,0,0,0]
        Ex-> If label is 9 then one hot encoding should be [0,0,0,0,0,0,0,0,0,1]
        Hint: Use sklearn one hot-encoder to convert your labels into one hot encoding array
        "onehotlabels" is a numpy array of labels. In the end just do np.array(onehotlabels).
        '''

        one_hot = np.zeros((len(labels),10))
        for i,label in enumerate(np.array(labels)):
            one_hot[i][label] = 1
        return (one_hot)
    
    def save_weights(self,fileName):
        '''save the weights of your neural network into a file
        Hint: Search python functions for file saving as a .txt'''
        # np.savetxt(fileName, self.weights_)

        data = {'weights': self.weights_, 'biases': self.biases_}
        file = open(fileName, 'wb')
        pickle.dump(data, file)
        file.close()
    
    def reassign_weights(self,fileName):
        '''assign the saved weights from the fileName to the network
        Hint: Search python functions for file reading
        '''
        file = open(fileName, 'rb')
        data = pickle.load(file)
        file.close()
        self.weights_ = data['weights']
        self.biases_ = data['biases']

    def savePlot(self, lrs):
        '''function to plot the execution time versus learning rate plot
        You can edit the parameters pass to the savePlot function'''
        plt.plot(lrs, self.times)
        plt.xlabel("Learning rate")
        plt.ylabel("Execution time")

        plt.savefig('lr vs exectime.png')

        


def main():
    neural_network = NeuralNetwork()
    run_type = sys.argv[1]
    if run_type == 'train':
        # lr = float(sys.argv[4])
        print('Training')
        img, labels = neural_network.give_images('train')
        
        # print('Learning with rate: ', sys.argv[4])
        start_time = time.time()
        history = neural_network.fit(np.array(img), labels, epochs=2, lr=0.001)
        end_time = time.time()
        total_time = (end_time - start_time)
        neural_network.times.append(total_time)
        print('Time taken: ', total_time)
        
        # print('Learning with rate: 0.01')
        # start_time = time.time()
        # history = neural_network.fit(np.array(img), labels, epochs=2, lr=0.01)
        # end_time = time.time()
        # total_time = (end_time - start_time)
        # neural_network.times.append(total_time)
        # print('Time take: ', total_time)

        # print('Learning with rate: 0.1')
        # start_time = time.time()
        # history = neural_network.fit(np.array(img), labels, epochs=2, lr=0.1)
        # end_time = time.time()
        # total_time = (end_time - start_time)
        # neural_network.times.append(total_time)
        # print('Time take: ', total_time)


        neural_network.save_weights('netWeights.txt')
        # neural_network.savePlot([0.001, 0.01, 0.1])
    
    if run_type == 'test':
        print('Testing')
        neural_network = NeuralNetwork()
        
        neural_network.reassign_weights(sys.argv[4])
        start_time = time.time()
        img, labels = neural_network.give_images('test')

        loss, accuracy = neural_network.evaluate(img, labels)    
        print('Accuracy: ',accuracy, '%')
        print('Error: ', 100 - accuracy, '%')

        total = (time.time() - start_time)
        
        print('Total testing time:', total, 'secs')    



main()

