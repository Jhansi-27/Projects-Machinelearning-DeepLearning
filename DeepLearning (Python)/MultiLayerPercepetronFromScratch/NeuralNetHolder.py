from numpy import genfromtxt
import numpy as np


class NeuralNetHolder:

    def __init__(self):
        super().__init__()
        self.weights_hidden = np.genfromtxt("Hidden_weights.csv", delimiter=",")
        self.bias_hidden = np.genfromtxt("bias_hidden.csv", delimiter=",")
        self.weights_out = np.genfromtxt("weights_out.csv", delimiter=",")
        self.bias_out = np.genfromtxt("bias_out.csv", delimiter=",")
        print("Hidden_weights:", self.weights_hidden)
        print("Bias_hidden:", self.bias_hidden)
        print("Out_weights:", self.weights_out)
        print("Out_Bias:", self.bias_out)

    def dot_product(self, inputs, weights):

        self.result = np.zeros((inputs.shape[0], weights.shape[1]))

        for i in range(inputs.shape[0]):
            for j in range(weights.shape[1]):
                for k in range(weights.shape[0]):
                    self.result[i][j] += inputs[i][k] * weights[k][j]
        return self.result

    def feed_forward(self, inputs):

        self.inputs = inputs

        inputs_hidden = self.dot_product(self.inputs, self.weights_hidden) + self.bias_hidden
        self.hidden_output = 1 / (1 + np.exp(-0.7 * inputs_hidden))
        inputs_to_outlayer = self.dot_product(self.hidden_output, self.weights_out) + self.bias_out
        self.output = 1 / (1 + np.exp(-0.7 * inputs_to_outlayer))
        #print(self.output)
        return self.output
        # returns the output of the network

    def predict(self, input_row):
        maximum_element_columns = np.genfromtxt("max_element.csv",delimiter=",")
        minimum_element_columns = np.genfromtxt("min_element.csv",delimiter=",")
        print("max_column_values",maximum_element_columns[:2])
        print("min_column_values",minimum_element_columns[:2])


        input_row = input_row.split(",")
        inputs = [float(i) for i in input_row]
        inputs = np.array(inputs)
        input_r = inputs.reshape((-1, 2))
        print("inputs: ",input_r)

        # normalizing the inputs
        normalized_x = (input_r[0][0] - minimum_element_columns[0]) / (maximum_element_columns[0] - minimum_element_columns[0])
        normalized_y = (input_r[0][1] - minimum_element_columns[1]) / (maximum_element_columns[1] - minimum_element_columns[1])
        normalized_inputs = np.array([[normalized_x,normalized_y]])
        print("normalized_inputs",normalized_inputs)
        
        predicted_output = self.feed_forward(normalized_inputs)
        print("network _prediction",predicted_output)
        # denormalizing the predictions
        denorm_x = predicted_output[0][0] * (maximum_element_columns[2] - minimum_element_columns[2]) + minimum_element_columns[2]
        denorm_y = predicted_output[0][1] * (maximum_element_columns[3] - minimum_element_columns[3]) + minimum_element_columns[3]
        
        predicted = [denorm_x,denorm_y]
        print("predicted[0]",predicted)
        return predicted  # [output1, output2]

