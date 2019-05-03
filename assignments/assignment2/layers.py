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
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

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
