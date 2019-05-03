import numpy as np


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


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO implement softmax with cross-entropy
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)
    if len(probs.shape) == 1:
      subtr = np.zeros(probs.shape)
      subtr[target_index] = 1
      dprediction = probs - subtr
    else:
      subtr = np.zeros(probs.shape)
      subtr[range(len(target_index)), target_index] = 1
      dprediction = (probs - subtr)/(predictions.shape[0])
      #print(dprediction)
    return loss, dprediction


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # TODO: implement l2 regularization and gradient
    #raise Exception("Not implemented!")
    loss = 0.5*reg_strength*np.sum(W**2)
    grad = reg_strength*W
    return loss, grad
    

def linear_softmax(X, W, target_index): 
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)
    loss, dprediction = softmax_with_cross_entropy(predictions, target_index)
    dW = np.dot(dprediction.T, X).T
    # TODO implement prediction and gradient over W
    #raise Exception("Not implemented!")
    
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=10, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)
            X_batch = X[batches_indices[1]]
            y_batch = y[batches_indices[1]]
            loss_pred, grad_pred = linear_softmax(X_batch, self.W, y_batch)
            loss_reg, grad_w = l2_regularization(self.W, reg)
            loss = loss_pred + loss_reg
            self.W -= learning_rate*(grad_pred + grad_w)
            loss_history.append(loss)
            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            #raise Exception("Not implemented!")

            # end
            #print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)
        predictions = np.dot(X, self.W)
        probs = softmax(predictions)
        y_pred = np.argmax(probs, axis=-1) 
        # TODO Implement class prediction
        #raise Exception("Not implemented!")

        return y_pred



                
                                                          

            

                
