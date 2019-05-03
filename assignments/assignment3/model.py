import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization, softmax
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels, reg):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        self.reg = reg

        self.conv1 = ConvolutionalLayer(in_channels = input_shape[-1], out_channels = conv1_channels ,filter_size =  3, padding = 1)
        self.relu1 = ReLULayer()
        self.maxpool1 = MaxPoolingLayer(pool_size = 4, stride = 4)
        self.conv2 = ConvolutionalLayer(in_channels = conv1_channels, out_channels = conv2_channels ,filter_size =  3, padding = 1)
        self.relu2 = ReLULayer()
        self.maxpool2 = MaxPoolingLayer(pool_size = 4, stride = 4)
        self.flattener = Flattener()

        ## n_input = 4*conv2_channels - hard coding here, because of constant picture size 32 32 3
        
        self.fullyconlayer = FullyConnectedLayer(n_input = 4*conv2_channels, n_output = n_output_classes)

        self.W_fc_layer = None
        self.B_fc_layer = None
        self.W_con1_layer = None
        self.B_con1_layer = None
        self.W_con2_layer = None
        self.B_con2_layer = None
        # TODO Create necessary layers
        #raise Exception("Not implemented!")

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment

        self.conv1.params()['W'].grad = 0
        self.conv1.params()['B'].grad = 0
        self.conv2.params()['W'].grad = 0
        self.conv2.params()['B'].grad = 0
        self.fullyconlayer.params()['W'].grad = 0
        self.fullyconlayer.params()['B'].grad = 0

        to_relu = self.conv1.forward(X)
        to_maxpool1 = self.relu1.forward(to_relu)
        to_conv2 = self.maxpool1.forward(to_maxpool1)
        to_relu2 = self.conv2.forward(to_conv2)
        to_maxpool2 = self.relu2.forward(to_relu2)
        to_flat = self.maxpool2.forward(to_maxpool2)
        to_fc_layer = self.flattener.forward(to_flat)
        preds = self.fullyconlayer.forward(to_fc_layer)
        loss, dprediction = softmax_with_cross_entropy(preds, y)

        grad_from_fc_layer = self.fullyconlayer.backward(dprediction)
        self.W_fc_layer = self.fullyconlayer.params()['W']
        self.B_fc_layer = self.fullyconlayer.params()['B']

        grad_from_flatten = self.flattener.backward(grad_from_fc_layer)

        grad_from_maxpool2 = self.maxpool2.backward(grad_from_flatten)

        grad_from_relu2 = self.relu2.backward(grad_from_maxpool2)

        grad_from_conv2 = self.conv2.backward(grad_from_relu2)
        self.W_con2_layer = self.conv2.params()['W']
        self.B_con2_layer = self.conv2.params()['B']

        grad_from_maxpool1 = self.maxpool1.backward(grad_from_conv2)

        grad_from_relu1 = self.relu1.backward(grad_from_maxpool1)

        grad_from_conv1 = self.conv1.backward(grad_from_relu1)
        self.W_con1_layer = self.conv1.params()['W']
        self.B_con1_layer = self.conv1.params()['B']

        loss_fc, grad_fc = l2_regularization(self.W_fc_layer.value, self.reg)

        loss += loss_fc
        self.W_fc_layer.grad += grad_fc

        return loss
        #raise Exception("Not implemented!")

    def predict(self, X):
        # You can probably copy the code from previous assignment
        pred = np.zeros(X.shape[0], np.int)
        to_relu = self.conv1.forward(X)
        to_maxpool1 = self.relu1.forward(to_relu)
        to_conv2 = self.maxpool1.forward(to_maxpool1)
        to_relu2 = self.conv2.forward(to_conv2)
        to_maxpool2 = self.relu2.forward(to_relu2)
        to_flat = self.maxpool2.forward(to_maxpool2)
        to_fc_layer = self.flattener.forward(to_flat)
        preds = self.fullyconlayer.forward(to_fc_layer)
        
        probs = softmax(preds)
        pred = np.argmax(probs, axis=-1)
        return pred
        #raise Exception("Not implemented!")

    def params(self):
        result = {'W_fc_layer' : self.W_fc_layer, 'B_fc_layer' : self.B_fc_layer, 
                  'W_con1_layer' : self.W_con1_layer,'B_con1_layer' : self.B_con1_layer,
                  'W_con2_layer' : self.W_con2_layer, 'B_con2_layer' : self.B_con2_layer}

        # TODO: Aggregate all the params from all the layers
        # which have parameters
        #raise Exception("Not implemented!")

        return result
