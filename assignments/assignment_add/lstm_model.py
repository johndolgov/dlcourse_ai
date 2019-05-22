import numpy as np
from copy import deepcopy


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def backward_tanh(x):
    return 1 - x*x


def tanh(x):
    return np.tanh(x)


def backward_sigmoid(x):
    return x*(1 - x)


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
    probs_left = np.choose(target_index, probs.reshape(probs.shape[0], probs.shape[1]).T)
    return np.sum(-np.log(probs_left))


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
    pred = predictions.copy()
    pred_temp = np.swapaxes(pred, 0, 1)
    pred = np.swapaxes(pred_temp - np.max(pred, axis=1), 1, 0)
    exps = np.exp(pred)
    downs = np.sum(exps, axis=1)
    probs = exps/downs[:, None]
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
    probs = softmax(preds)
    loss = cross_entropy_loss(probs, target_index)
    d_out = np.copy(probs)
    for idx, row in enumerate(d_out):
        row[target_index[idx]] -= 1
    return loss, d_out


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class Seq2SeqLSTM:
    def __init__(self, hid_layer_size, num_dict, w_std):
        self.hid_layer_size = hid_layer_size
        self.num_dict = num_dict

        conc_size = hid_layer_size + num_dict

        self.W_f = Param(w_std * np.random.randn(hid_layer_size, conc_size))
        self.b_f = Param(np.zeros((hid_layer_size, 1)))

        self.W_i = Param(w_std * np.random.randn(hid_layer_size, conc_size))
        self.b_i = Param(np.zeros((hid_layer_size, 1)))

        self.W_c = Param(w_std * np.random.randn(hid_layer_size, conc_size))
        self.b_c = Param(np.zeros((hid_layer_size, 1)))

        self.W_o = Param(w_std * np.random.randn(hid_layer_size, conc_size))
        self.b_o = Param(np.zeros((hid_layer_size, 1)))

        self.W_y = Param(w_std * np.random.randn(num_dict, hid_layer_size))
        self.b_y = Param(np.zeros((num_dict, 1)))

        self.h = Param(np.zeros((hid_layer_size, 1)))
        self.C = Param(np.zeros((hid_layer_size, 1)))
        self.ft = Param(np.zeros((hid_layer_size, 1)))
        self.it = Param(np.zeros((hid_layer_size, 1)))
        self.C_hat = Param(np.zeros((hid_layer_size, 1)))
        self.out = Param(np.zeros((hid_layer_size, 1)))

        self.cache = {}

    def params(self):
        return {'W_f': self.W_f, 'W_i': self.W_i,
                'W_c': self.W_c, 'W_o': self.W_o, 'W_y': self.W_y, 'b_f': self.b_f,
                'b_i': self.b_i, 'b_c': self.b_c, 'b_o': self.b_o, 'b_y': self.b_y,
                'h': self.h, 'C': self.C, 'ft': self.ft, 'it': self.it,
                'C_hat': self.C_hat, 'out': self.out}

    def forward(self, X_train):

        outputs = []

        for idx, x in enumerate(X_train):
            x_one_hot = np.zeros(self.num_dict)
            x_one_hot[x] = 1
            x_one_hot = x_one_hot.reshape(1, -1)

            X = Param(np.row_stack((self.h.value, x_one_hot.T)))

            if -1 not in self.cache:
                self.cache[-1] = (X, self.ft, self.it, self.C_hat, self.out, self.h, self.C)

            self.ft.value = sigmoid(np.dot(self.W_f.value, X.value) + self.b_f.value)
            self.it.value = sigmoid(np.dot(self.W_i.value, X.value) + self.b_i.value)
            self.C_hat.value = tanh(np.dot(self.W_c.value, X.value) + self.b_c.value)
            self.C.value = self.ft.value*self.C.value + self.it.value*self.C_hat.value
            self.out.value = sigmoid(np.dot(self.W_o.value, X.value) + self.b_o.value)
            self.h.value = self.out.value*tanh(self.C.value)
            output = np.dot(self.W_y.value, self.h.value) + self.b_y.value
            self.cache[idx] = (X, self.ft, self.it, self.C_hat, self.out, self.h, self.C)
            outputs.append(output)

        return np.array(outputs)

    def backward(self, d_out):
        dh_next = np.zeros_like(self.h.value)
        dc_next = np.zeros_like(self.C.grad)
        for idx in range(len(self.cache.keys()) - 2, -1, -1):
            X, ft, it, C_hat, out, h, C = self.cache[idx]
            _, _, _, _, _, _, C_prev = self.cache[idx - 1]

            self.W_y.grad += np.dot(d_out[idx], h.value.T)
            self.b_y.grad += d_out[idx]

            h.grad = np.dot(self.W_y.value.T, d_out[idx]) + dh_next
            out.grad = h.grad*tanh(C.value)
            out.grad = backward_sigmoid(out.value)*out.grad

            self.W_o.grad += np.dot(out.grad, X.value.T)
            self.b_o.grad += out.grad

            C.grad = np.copy(dc_next)
            C.grad += h.grad*out.value*backward_tanh(tanh(C.value))
            C_hat.grad = C.grad * it.value
            C_hat.grad = backward_tanh(C_hat.value)*C_hat.grad
            self.W_c.grad += np.dot(C_hat.grad, X.value.T)
            self.b_c.grad += C_hat.grad

            it.grad = C.grad*C_hat.value
            it.grad = backward_sigmoid(it.value)*it.grad
            self.W_i.grad += np.dot(it.grad, X.value.T)
            self.b_i.grad += it.grad

            ft.grad = C.grad*C_prev.value
            ft.grad = backward_sigmoid(ft.value)*ft.grad
            self.W_f.grad += np.dot(ft.grad, X.value.T)
            self.b_f.grad += ft.grad

            X.grad = (np.dot(self.W_f.value.T, ft.grad) + np.dot(self.W_i.value.T, it.grad) +
                  np.dot(self.W_c.value.T, C_hat.grad) + np.dot(self.W_o.value.T, out.grad))

            dh_next = X.grad[:self.hid_layer_size, :]
            C_prev.grad = ft.value*C.grad

            dc_next = C_prev.grad

    def predict(self, X_val):
        outputs = []

        for idx, x in enumerate(X_val):
            x_one_hot = np.zeros(self.num_dict)
            x_one_hot[x] = 1
            x_one_hot = x_one_hot.reshape(1, -1)

            X = Param(np.row_stack((self.h.value, x_one_hot.T)))

            self.ft.value = sigmoid(np.dot(self.W_f.value, X.value) + self.b_f.value)
            self.it.value = sigmoid(np.dot(self.W_i.value, X.value) + self.b_i.value)
            self.C_hat.value = tanh(np.dot(self.W_c.value, X.value) + self.b_c.value)
            self.C.value = self.ft.value * self.C.value + self.it.value * self.C_hat.value
            self.out.value = sigmoid(np.dot(self.W_o.value, X.value) + self.b_o.value)
            self.h.value = self.out.value * tanh(self.C.value)
            output = np.dot(self.W_y.value, self.h.value) + self.b_y.value
            self.cache[idx] = (X, self.ft, self.it, self.C_hat, self.out, self.h, self.C)
            outputs.append(output)

        probs = softmax(outputs)
        pred = np.argmax(probs, axis=1).T
        return pred

    def clear_parameters(self):

        self.h = Param(np.zeros((self.hid_layer_size, 1)))
        self.C = Param(np.zeros((self.hid_layer_size, 1)))
        self.ft = Param(np.zeros((self.hid_layer_size, 1)))
        self.it = Param(np.zeros((self.hid_layer_size, 1)))
        self.C_hat = Param(np.zeros((self.hid_layer_size, 1)))
        self.out = Param(np.zeros((self.hid_layer_size, 1)))
        self.cache = {}

    def clear_gradients(self):
        for key in self.params().keys():
            self.params()[key].grad = np.zeros_like(self.params()[key].value)

    def clip_gradients(self):
        for key in self.params().keys():
            np.clip(self.params()[key].grad, -1, 1, out=self.params()[key].grad)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        self.clear_parameters()

        out = self.forward(X)
        loss, d_out = softmax_with_cross_entropy(out, y)
        self.backward(d_out)

        self.clip_gradients()
        return loss


class SGD:
    """
    Implements vanilla SGD update
    """
    def update(self, w, d_w, learning_rate):
        """
        Performs SGD update

        Arguments:
        w, np array - weights
        d_w, np array, same shape as w - gradient
        learning_rate, float - learning rate

        Returns:
        updated_weights, np array same shape as w
        """
        return w - d_w * learning_rate


class MomentumSGD:
    """
    Implements Momentum SGD update
    """

    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.velocity = 0

    def update(self, w, d_w, learning_rate):
        """
        Performs Momentum SGD update

        Arguments:
        w, np array - weights
        d_w, np array, same shape as w - gradient
        learning_rate, float - learning rate

        Returns:
        updated_weights, np array same shape as w
        """
        self.velocity = self.momentum * self.velocity - learning_rate * d_w
        return w + self.velocity


class Dataset:
    """
    Utility class to hold training and validation data
    """

    def __init__(self, dict_size, seq_len, train_data_size, val_data_size):

        self.dict_size = dict_size
        self.seq_len = seq_len
        self.train_data_size = train_data_size
        self.val_data_size = val_data_size

        self.train_X, self.train_y, self.val_X, self.val_y = self.generate_dataset()

    def generate_dataset(self):

        train_X = np.random.randint(self.dict_size, size=(self.train_data_size, self.seq_len))
        train_y = np.sort(train_X, axis=1)

        val_X = np.random.randint(self.dict_size, size=(self.val_data_size, self.seq_len))
        val_y = np.sort(val_X, axis=1)

        return train_X, train_y, val_X, val_y

    def data(self):
        return self.train_X, self.train_y, self.val_X, self.val_y


class Trainer:
    """
    Trainer of the neural network models
    Perform mini-batch SGD with the specified data, model,
    training parameters and optimization rule
    """

    def __init__(self, model, dataset, optim,
                 num_epochs=20,
                 batch_size=20,
                 learning_rate=1e-3,
                 learning_rate_decay=1.0):
        """
        Initializes the trainer

        Arguments:
        model - neural network model
        dataset, instance of Dataset class - data to train on
        optim - optimization method (see optim.py)
        num_epochs, int - number of epochs to train
        batch_size, int - batch size
        learning_rate, float - initial learning rate
        learning_rate_decal, float - ratio for decaying learning rate
           every epoch
        """
        self.dataset = dataset
        self.model = model
        self.optim = optim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.learning_rate_decay = learning_rate_decay
        self.optimizers = None

    def setup_optimizers(self):
        params = self.model.params()
        self.optimizers = {}
        for param_name, param in params.items():
            self.optimizers[param_name] = deepcopy(self.optim)

    def compute_accuracy(self, X, y):
        accuracy = []
        for ind, x in enumerate(X):
            pred = self.model.predict(x)
            accuracy.append(np.sum(pred == y[ind])/len(X))
        return np.mean(accuracy)

    def fit(self):
        """
        Trains a model
        """
        if self.optimizers is None:
            self.setup_optimizers()

        num_train = self.dataset.train_X.shape[0]

        loss_history = []
        accuracy_train_history = []
        accuracy_val_history = []

        for epoch in range(self.num_epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(self.batch_size, num_train, self.batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            batch_losses = []
            batch_train_history = []

            for batch_indices in batches_indices:
                self.model.clear_gradients()
                batch_X = self.dataset.train_X[batch_indices]
                batch_y = self.dataset.train_y[batch_indices]
                loss = []
                #print(self.model.W_f.grad)
                for ind, X in enumerate(batch_X):
                    loss.append(self.model.compute_loss_and_gradients(X, batch_y[ind]))
                #print(self.model.W_f.grad)

                loss = np.mean(loss)
                #batch_train_history.append(self.compute_accuracy(batch_X, batch_y))

                #print(self.model.W_f.value)
                for param_name, param in self.model.params().items():
                    optimizer = self.optimizers[param_name]
                    param.value = optimizer.update(param.value, param.grad, self.learning_rate)
                #print(self.model.W_f.value)

                batch_losses.append(loss)

            self.learning_rate *= self.learning_rate_decay

            ave_loss = np.mean(batch_losses)
            #ave_train_history = np.mean(batch_train_history)

            batch_X_val = self.dataset.val_X
            batch_y_val = self.dataset.val_y
            #accuracy_val_history.append(self.compute_accuracy(batch_X_val, batch_y_val))
            #accuracy_train_history.append(ave_train_history)

            print(f"Epoch: {epoch}, Loss: {ave_loss}, Train accuracy: {0}, Val accuracy: {0} ")

            loss_history.append(ave_loss)

        return loss_history, accuracy_train_history, accuracy_val_history







        









