import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, softmax


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        hidden_layer_size, int - number of neurons in the hidden layer
        n_input, int - dimension of the model input
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.input_layer = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu = ReLULayer()
        self.output_layer = FullyConnectedLayer(hidden_layer_size, n_output)
        self.W_in = None
        self.W_out = None
        self.B_in = None
        self.B_out = None
        # TODO Create necessary layers
<<<<<<< HEAD
        #raise Exception("Not implemented!")
=======
        raise Exception("Not implemented!")
>>>>>>> ed32e861bd04bb9058726f0298e51b2f9ecb40b1

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
<<<<<<< HEAD
        self.output_layer.params()['W'].grad = 0
        self.output_layer.params()['B'].grad = 0
        self.input_layer.params()['W'].grad = 0
        self.input_layer.params()['B'].grad = 0
        to_relu = self.input_layer.forward(X)
        to_outlayer = self.relu.forward(to_relu)
        pred = self.output_layer.forward(to_outlayer)
        loss, dprediction = softmax_with_cross_entropy(pred,y)
        grad_out_layer = self.output_layer.backward(dprediction)
        self.W_out = self.output_layer.params()['W']
        self.B_out = self.output_layer.params()['B']
        self.input_layer.backward(self.relu.backward(grad_out_layer))
        self.W_in = self.input_layer.params()['W']
        self.B_in = self.input_layer.params()['B']
        loss_l2_in, grad_l2_in = l2_regularization(self.W_in.value, self.reg)
        loss_l2_out, grad_l2_out = l2_regularization(self.W_out.value, self.reg)
        loss += loss_l2_in + loss_l2_out
        self.W_out.grad += grad_l2_out
        self.W_in.grad += grad_l2_in
=======
        
>>>>>>> ed32e861bd04bb9058726f0298e51b2f9ecb40b1
        # After that, implement l2 regularization on all params
        # Hint: use self.params()
        #raise Exception("Not implemented!")

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)
        to_relu = self.input_layer.forward(X)
        to_outlayer = self.relu.forward(to_relu)
        weights = self.output_layer.forward(to_outlayer)
        probs = softmax(weights)
        pred = np.argmax(probs, axis=-1)
        #raise Exception("Not implemented!")
        return pred

    def params(self):
        result = {'W_out': self.W_out, 'W_in': self.W_in, 
                   'B_out': self.B_out, 'B_in': self.B_in}

        # TODO Implement aggregating all of the params

        #raise Exception("Not implemented!")
        return result
