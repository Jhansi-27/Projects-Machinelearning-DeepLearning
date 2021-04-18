#!/usr/bin/env python
# coding: utf-8

# In[80]:


import math
import numpy as np
from numpy import savetxt # used for saving files
from sklearn.model_selection import train_test_split # for splitting data into training and validation sets


# In[81]:


class DataPreprocessor:
    
    """This class cleans the data, normalizes it and splits data into train and validation sets"""
    
    def __init__(self,data):
        self.data = data
        
    def data_process(self):
        """This function removes duplicate rows and remove negative values from y-distance to target column
        """
        print("> Shape(no of rows and columns) of the Data: ",self.data.shape)
        self.unique_data = np.unique(self.data,axis=0)  
        null_values = np.isnan(np.sum(self.data))
        print("> Null values in data: ",null_values)
        if null_values > 0:
            print("treat null_values")
        self.unique_data =  self.unique_data[self.unique_data[:,1] > 0] 
        print("> Shape of the processed data: ", self.unique_data.shape)
        
        
    def  data_normalization(self):
        """This function normalizes data between 0 and 1
        """
        maxvalue_column = np.amax(self.unique_data,axis=0)
        minvalue_column = np.amin(self.unique_data,axis=0)
        """saving the maximum and minimum values of columns for using in NeuralNetHolder class"""
        np.savetxt("max_element.csv",maxvalue_column)
        np.savetxt("min_element.csv", minvalue_column)
        
        self.normalized_data = (self.unique_data - minvalue_column) / (maxvalue_column - minvalue_column)
        np.savetxt("Normalized_game_data.csv",self.normalized_data,delimiter=",")
        print("> Data Normalization Done ")
        
    def train_val_split(self):
        """
        This function splits data randomly, 80% data in to train set and 20% data into validation set
        """
        x_train,x_val,y_train,y_val = train_test_split(self.normalized_data[:,:2],self.normalized_data[:,2:],test_size=0.2,random_state=42)
        
        print("> length of train_set: {0}, length of validation_set: {1}".format(x_train.shape[0],x_val.shape[0]))
        print(" ")
        return  x_train,x_val,y_train,y_val


# In[97]:


class MLP:
    """
    This class defines the neural netwrok architecture, trains the network on every row of input and weights &
    runs forward propogation and backward propogation, calculates total error for every epoch
    """
    
    def __init__(self,input_num,output_num):
        """
        This function initializes weights, biases for hidden and output layers by taking no of features
        and number of outputs and it is also used to initialize hyperparameter values
        """
        self.input_neurons = input_num
        self.output_neurons = output_num
        
        # Hyper parameters
        self.hidden_neurons = 8
        self.lr = 0.7
        self.lamda = 0.7
        self.momentum_rate = 0.4
        print("Hyperparameter values used for training: Hidden_neurons: {}, learning_rate: {}, Lambda: {}, momentum_rate: {}\n".format(self.hidden_neurons,self.lr,self.lamda,self.momentum_rate))
        
        np.random.seed(25)
        """Initializing weights for hidden layer, dimension of weight matrix = (num of input  neurons, num of hidden neurons)"""
        """Initializing bias for hidden layer, dimension of bias matrix = (1,num of input  neurons)"""
      
        self.weights_hidden = np.random.rand(self.input_neurons,self.hidden_neurons)
        self.bias_hidden = np.random.rand(1,self.hidden_neurons)
        
        """Initializing weightsfor output layer, dimension of weight matrix = (num of hidden neurons, num of output neurons)"""
        """Initializing bias output layer, dimension of bias matrix = (1,num of hidden neurons)"""
      
        self.weights_out = np.random.rand(self.hidden_neurons,self.output_neurons)
        self.bias_out = np.random.rand(1,self.output_neurons)
        
        """similarly to add momentum term initializing zero matrices """
        self.moment_weights_hidden = np.zeros((self.input_neurons,self.hidden_neurons))
        self.moment_bias_hidden = np.zeros((1,self.hidden_neurons))
        self.moment_weights_out = np.zeros((self.hidden_neurons,self.output_neurons))
        self.moment_bias_out = np.zeros((1,self.output_neurons))

    def sigmoid(self,z):
        """
        This is the sigmoid function, it squashes values passed between 0 and 1 and returns them
        """
        return 1/ (1+np.exp(-z*self.lamda))
    
    def derivative_sigmoid(self,x):
        """
        This function returns the derivative of sigmoid function
        """
        return x * (1-x) 
    
    
    def dot_product(self,inputs, weights):
        """
        This function calculates the dot product of the inputs and weights
        """
        self.result =  np.zeros((inputs.shape[0],weights.shape[1]))
         
        for i in range(inputs.shape[0]):
            for j in range(weights.shape[1]):
                for k in range(weights.shape[0]):
                    self.result[i][j] += inputs[i][k] * weights[k][j]
        return self.result      
    
    def feed_forward(self,inputs):
        """
        This is the feed forward function: It takes the inputs and 
        1. Calculates the sum of weights and inputs 
        2. Applies sigmoid function on sum
        3. Returns the predicted output
        """
        self.inputs = inputs
        
        inputs_hidden = self.dot_product(self.inputs,self.weights_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(inputs_hidden)
        inputs_to_outlayer = self.dot_product(self.hidden_output,self.weights_out) + self.bias_out
        self.output = self.sigmoid(inputs_to_outlayer)
        return self.output         
        # returns the output of the network 
    
     
    def back_prop(self,error):
        """
        This is the back propogation function: It takes error as an argument 
        error = actual value - predicted value(ouput of network)
        1. calculates the gradient at output and hidden layers
        2. calculates the delta of weight and bias at hidden and output layers
        3. updates the weights and biases at each layer
        """
        self.error = error
        
        # output layer
        """ calculating the gradient and delta weight and delta bias at output layer"""
        self.gradient_output = self.lamda * self.error * self.derivative_sigmoid(self.output)
        self.delta_weight_output = self.lr * self.dot_product(self.hidden_output.reshape(self.hidden_neurons,-1),self.gradient_output) + self.momentum_rate * self.moment_weights_out 
        self.delta_bias_out =  self.lr* self.gradient_output + self.momentum_rate * self.moment_bias_out 
       
        #hidden layer
        """ calculating the gradient and delta weight and delta bias at hidden layer"""
        self.gradient_hidden = self.lamda* self.derivative_sigmoid(self.hidden_output)*self.dot_product(self.gradient_output,self.weights_out.T) 
        self.delta_weights_hidden = self.lr * self.dot_product(self.inputs.reshape(self.input_neurons,-1),self.gradient_hidden)+ self.momentum_rate * self.moment_weights_hidden 
        self.delta_bias_hidden = self.lr * self.gradient_hidden + self.momentum_rate * self.moment_bias_hidden
      
        """adding delta weights and biases to update weights"""
        self.weights_out =  self.weights_out +  self.delta_weight_output
        self.bias_out =   self.bias_out + self.delta_bias_out
        self.weights_hidden =  self.weights_hidden +  self.delta_weights_hidden
        self.bias_hidden =  self.bias_hidden + self.delta_bias_hidden
        
        """Changing moementum term"""
        self.moment_weights_hidden  =  self.delta_weights_hidden
        self.moment_bias_hidden =  self.delta_bias_hidden
        self.moment_weights_out = self.delta_weight_output
        self.moment_bias_out =  self.delta_bias_out
        
    
    def train(self, epochs, x_train, y_train, x_val, y_val):
        """
        This is the training function: It takes training sets and validation sets
        1. calls feed forward function and stores predicted output in a variable
        2. calculates the error
        3. calls backpropogation function to calculate gradients, deltas and adjust weights 
        4. calculates and returns total error at the end of each epoch for both training and validation data sets
        """

        self.epochs = epochs # hyperparameter
        Training_errors = list()
        Validation_errors = list()
        
        """trains the network for the number of epochs given"""
        for epoch in range(1,self.epochs+1):
            error_total_train = 0
            error_total_val = 0
            RMSE_val_previous = 0
           
            """training network on training data"""
            for i in range(len(x_train)):
                error = [[0,0]]
                predicted_output = self.feed_forward(x_train[[i]])  # predicted output of network
                error = y_train[[i]] - predicted_output # error = actual - predicted
                self.back_prop(error) # backpropogates error
                row_error = error ** 2 # squares the error
                row_sum = np.mean(row_error) # we have two errors corresponding to 2 columns so take average
                error_total_train += row_sum # add all the errors to calculate total error
               
            RMSE_Train = np.sqrt(error_total_train/len(x_train))
            # divide the total error by length of data and apply np.sqrt to calculate root mean square error
            Training_errors.append(round(RMSE_Train,7))
            # append RMSE error to a list after every epoch for  plotting
            
            #print(Training_errors)
            """testing using validation data set"""
            for i in range(len(x_val)):
               
                predicted = self.feed_forward(x_val[[i]]) # predicted output
                error2 = y_val[[i]] - predicted # error
                 
                row_error2 = error2 ** 2 # error sqaured
                row_sum2 = np.mean(row_error2)
                error_total_val +=  row_sum2 # total error
                
            RMSE_Val = np.sqrt(error_total_val/len(x_val)) # root mean square validarion error
            Validation_errors.append(round(RMSE_Val,7))
            # append RMSE error to a list after every epoch for  plotting
            
            print("> Epoch : "+str(epoch) +" - "+"Training Error: "+str(RMSE_Train)+"  "+"   Validation Error: "+str(RMSE_Val))
            
            """Stop the training if the validation error is increasing in 3 consecutive epochs and save weights """
            
            if (Validation_errors[epoch-1] > Validation_errors[epoch-2]):
                counter = counter + 1
        
            else:
                counter = 0
            if counter == 3:
                self.save_Bias_Weights()
        return Training_errors,Validation_errors    
 
           

    def save_Bias_Weights(self):
        """upon calling save the updated weights and biases """
        np.savetxt("Weights_hidden.csv",self.weights_hidden,delimiter=',') 
        np.savetxt("bias_hidden.csv",self.bias_hidden,delimiter=',') 
        np.savetxt("weights_out.csv",self.weights_out,delimiter=',') 
        np.savetxt("bias_out.csv",self.bias_out,delimiter=',') 
        


# In[98]:


"""Reading the data collected after playing the game in to a variable"""
from numpy import genfromtxt

my_data = genfromtxt('ce889_dataCollection.csv', delimiter=',')
  
"""Initializing object of DataPreprocessor to send data collected 
for cleaning, normalizing and splitting into training and validation sets"""
    
pre = DataPreprocessor(my_data)
pre.data_process()
pre.data_normalization()
x_train,x_val,y_train,y_val = pre.train_val_split()
  
    
input_num = x_train.shape[1]
output_num = y_train.shape[1]
    
"""
Initializing object of MLP class for creating network architecture and training the network
"""
network = MLP(input_num,output_num)
epochs = 200
train_errors, val_errors = network.train(epochs,x_train, y_train, x_val, y_val)
# collecting the training and validation errors for plotting


# In[99]:
# plotting training error vs number of epochs and plotting validation errors vs epochs

import matplotlib.pyplot as plt
x1 = range(1,epochs+1)
y1 = train_errors
plt.plot(x1, y1, label = "Training error")

x2 = range(1,epochs+1)
y2 = val_errors
plt.plot(x2, y2, label = "Validation error")

plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Epochs vs Error')

plt.legend()

plt.savefig("Error_plot7n.png")


# In[ ]:
# plotting data to see how the data is distributed

plt.figure(figsize=(10,8))

plt.subplot(221)
hist1,bin_edges = np.histogram(my_data[:,0],density=True)
plt.plot(hist1)

plt.subplot(222)
hist2,bin_edges = np.histogram(my_data[:,1],density=True)
plt.plot(hist2)

plt.subplot(223)
hist3,bin_edges = np.histogram(my_data[:,2],density=True)
plt.plot(hist3)

plt.subplot(224) 
hist4,bin_edges = np.histogram(my_data[:,3],density=True)
plt.plot(hist4)




