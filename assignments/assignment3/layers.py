import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    loss = 0.5*reg_strength*np.sum(W**2)
    grad = reg_strength*W
    return loss, grad


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    if len(probs.shape) == 1:
      return -np.log(probs[target_index])
    else:
      #print(target_index[:])
      probs_left = np.choose(target_index, probs.T)
      return np.mean(-np.log(probs_left))


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    predictions_2 = predictions.copy()
    if len(predictions_2.shape) == 1:
      predictions_2 -= np.max(predictions_2)
      probs = np.exp(predictions_2)/np.sum(np.exp(predictions_2))
    else:
      predictions_2 -= np.max(predictions_2, axis=1).reshape(predictions_2.shape[0], 1)
      exps = np.exp(predictions_2)
      downs = np.sum(exps, axis = 1)
      probs = exps/downs[:,None]
    return probs

def softmax_with_cross_entropy(preds, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (N, batch_size) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    # TODO: Copy from the previous assignment
    probs = softmax(preds)
    loss = cross_entropy_loss(probs, target_index)
    if len(probs.shape) == 1:
      subtr = np.zeros(probs.shape)
      subtr[target_index] = 1
      dprediction = probs - subtr
    else:
      subtr = np.zeros(probs.shape)
      subtr[range(len(target_index)), target_index] = 1
      dprediction = (probs - subtr)/(preds.shape[0])
      #print(dprediction)
    return loss, dprediction


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self.orig_X = X.copy()
        lay_X = X.copy()
        lay_X[lay_X < 0] = 0
        return lay_X
        #raise Exception("Not implemented!")
    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        #raise Exception("Not implemented!")
        X_back = self.orig_X.copy()
        X_back[X_back > 0] = 1
        X_back [X_back <= 0] = 0
        d_result = X_back*d_out
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        self.X = Param(X.copy())
        output = np.dot(self.X.value, self.W.value) + self.B.value
        return output
        #raise Exception("Not implemented!")

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment

        #raise Exception("Not implemented!")
        self.W.grad = np.dot(self.X.value.T, d_out)
        self.B.grad = np.array([np.sum(d_out, axis=0)])
        d_input = np.dot(d_out, self.W.value.T)
        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))
        self.X = None

        self.padding = padding


    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X = Param(X.copy())

        #height, width calculation

        out_height = (height - self.filter_size + 2*self.padding) + 1
        out_width = (width - self.filter_size + 2*self.padding) + 1

        #result preproc

        forward_result = np.zeros([batch_size, out_height, out_width, self.out_channels])

        #padding evaluation

        self.X.value = np.pad(self.X.value, ((0,0),(self.padding, self.padding), (self.padding, self.padding), (0,0)), 'constant')

        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        for y in range(out_height):
            for x in range(out_width):

                slc = self.X.value[:, y:y+self.filter_size, x:x+self.filter_size, :]
                slc = slc.reshape(batch_size, self.filter_size*self.filter_size*self.in_channels)

                W_flat = self.W.value.reshape(self.filter_size*self.filter_size*self.in_channels, self.out_channels)

                result = np.dot(slc, W_flat) + self.B.value
                forward_result[:, y, x, :] = result


        return forward_result


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.value.shape
        _, out_height, out_width, out_channels = d_out.shape


        d_input = np.zeros_like(self.X.value)

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)

                W_flat = self.W.value.reshape(self.filter_size*self.filter_size*self.in_channels, self.out_channels)
                X_flat = self.X.value[:, y:y+self.filter_size, x:x+self.filter_size, :].reshape(batch_size, self.filter_size*self.filter_size*self.in_channels)

                d_input[:, y:y+self.filter_size, x:x+self.filter_size, :] = d_input[:, y:y+self.filter_size, x:x+self.filter_size, :] +  np.dot(d_out[:, y, x, :], W_flat.T).reshape(batch_size, self.filter_size, self.filter_size, self.in_channels)

                self.W.grad = self.W.grad + np.dot(X_flat.T, d_out[:, y, x, :]).reshape(self.filter_size, self.filter_size, self.in_channels, out_channels)
                self.B.grad = np.sum(d_out, axis=tuple(range(len(d_out.shape)))[:-1]).reshape(out_channels)
        if(self.padding):
                    d_input = d_input[:, self.padding:-self.padding, self.padding:-self.padding, :]

        return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        self.X = X

        out_height = (height - self.pool_size)/self.stride + 1 
        out_width  = (width - self.pool_size)/self.stride + 1

        if (not float(out_height).is_integer() and not float(out_width).is_integer()):
            raise Exception(f"Stride and pool size aren't consistent for {height}, {width}")

        out = np.zeros([int(batch_size), int(out_height), int(out_width), int(channels)])
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        y_1 = 0
        for y in range(int(out_height)):
            x_1 = 0
            for x in range(int(out_width)):
                out[:, y, x, :] += np.amax(self.X[:, y_1:y_1+self.pool_size, x_1:x_1+self.pool_size, :], axis=(1,2))
                x_1 += self.stride
            y_1 += self.stride
        return out

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, channels = d_out.shape
        in_l = np.zeros_like(self.X)


        for b in range(batch_size):
            for ch in range(channels):
                y_1 = 0
                for y in range(out_height):
                    x_1 = 0
                    for x in range(out_width):
                        #print(b, y, x, ch)
                        #print(b, y_1,y_1+self.pool_size, x_1,x_1+self.pool_size, ch)
                        ind = np.unravel_index(np.argmax(self.X[b, y_1:y_1+self.pool_size, x_1:x_1+self.pool_size, ch]), self.X[b, y_1:y_1+self.pool_size, x_1:x_1+self.pool_size, ch].shape)
                        in_l[b, y_1:y_1+self.pool_size, x_1:x_1+self.pool_size, ch ][ind[0], ind[1]] = d_out[b, y, x, ch]
                        x_1 += self.stride
                    y_1 += self.stride
        return in_l

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X_shape = batch_size, height, width, channels
        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        return X.reshape(batch_size, height*width*channels)
        #raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO: Implement backward pass
        return d_out.reshape(self.X_shape[0], self.X_shape[1], self.X_shape[2], self.X_shape[3])
        #raise Exception("Not implemented!")

    def params(self):
        # No params!
        return {}
