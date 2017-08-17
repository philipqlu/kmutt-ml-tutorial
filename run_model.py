# coding: utf-8

'''
This file contains the solutions to our Thursday Tutorial. The functions are all
included in this module.
'''

# In[2]:
from __future__ import print_function
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

# # Artificial Neural Network for Object Recognition
# ## Team A: Taking in the Input Data
# ## Team B: Building a 2-Layer Neural Network Classifier
# ## Team C: Evaluating the Model

# In[3]:

# Data folders
train_data_dir = 'data/train/'
test_data_dir = 'data/test/'
label_file_path = 'data/labels.csv'


# In[6]:

################################################################################
# TEAM A Task: Load and preprocess the images from the directory on the disk.  #
################################################################################

def load_data(train_folder, test_folder):
    """
    Loads the images and label data from the training and test data folders.

    Inputs:
    - train_folder: string path to the training data folder
    - test_folder: string path to the testing data folder
    """
    X_train, y_train, X_test, y_test = [], [], [], []
    emotion = 0

    # Use a dictionary to map from image names to their labels from a label file

    data = []
    with open(label_file_path, 'r+') as label_file:
        next(label_file)
        for line in label_file:
            temp_data = line.split(',')
            data.append([temp_data[0],int(temp_data[1])])
    data_dicts = dict({d[0]:d[1] for d in data})

    # Loop through the images in the training folder and convert them to np arrays
    # Resize to 32x32 and convert each image to grayscale
    # Append each image's data to X_train and its label to  y_train

    for image_file in os.listdir(train_folder):
        image_path = os.path.join(train_folder, image_file)
        image_raw = Image.open(image_path).convert('L').resize((32,32), Image.ANTIALIAS)
        image_flat = np.asarray(image_raw.getdata())
        X_train.append(image_flat)
        y_train.append(data_dicts[image_file])

    ################################################################################
    # TODO:                                                                        #
    # Write code that loads the images and labels from the test folder like above  #
    # Append the image array to X_test and the label to y_test                     #
    # Hint: You only have to change four lines from the loop above!                #
    ################################################################################

    for image_file in os.listdir(test_folder):
        image_path = os.path.join(test_folder, image_file)
        image_raw = Image.open(image_path).convert('L').resize((32,32), Image.ANTIALIAS)
        image_flat = np.asarray(image_raw.getdata())
        X_test.append(image_flat)
        y_test.append(data_dicts[image_file])


    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    return X_train, y_train, X_test, y_test


# In[8]:

# PROVIDED CODE
X_train, y_train, X_test, y_test = load_data(train_data_dir, test_data_dir)

# # As a sanity check, we print out the size of the training and test data.
# The training set has 270 images, test set has 69 images.
# This represents an approximate 75/25 train/test split of the data.
print ('Training data shape: ', X_train.shape)
print ('Training labels shape: ', y_train.shape)
print ('Test data shape: ', X_test.shape)
print ('Test labels shape: ', y_test.shape)


# In[9]:

# PROVIDED CODE
# We show a few examples of training images from each class.

classes = ['cat', 'justin', 'som tam', 'temple']
num_classes = len(classes)
samples_per_class = 3
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].reshape((32,32)).astype('uint8'),cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()


################################################################################
# TODO:                                                                        #
# Compute the mean image of X and subtract that from train and test images     #
################################################################################
X_train = X_train - np.mean(X_train, axis=0)
X_test  = X_test - np.mean(X_train, axis=0)
################################################################################
#                              END OF YOUR CODE                                #
    ################################################################################

################################################################################
# TEAM B Task: Complete the 2 layer NN weight initialization and forward pass  #
################################################################################

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    a2 = X.dot(W1) + b1
    l2 = np.maximum(0,a2)
    scores = l2.dot(W2) + b2

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.                                                          #
    #############################################################################
    norm_scores = scores - np.max(scores,axis = 1).reshape(-1,1)
    exp_scores = np.exp(norm_scores)
    Probs = exp_scores/np.sum(exp_scores, axis =1).reshape(-1,1) # N*C

    correct_prob = Probs[range(N),list(y)]
    log_probs = -np.sum(np.log(correct_prob))
    data_loss = log_probs/N

    reg_loss = 0.5*reg*np.sum(W1*W1)+0.5*reg*np.sum(W2*W2)
    loss = data_loss+ reg_loss

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    dscores = Probs.copy()  # N*C
    dscores[range(N),y] += -1
    dscores /= N

    grads['W2'] = np.dot(l2.T,dscores)
    grads['b2'] = np.sum(dscores,axis = 0)
    dhidden = np.dot(dscores,W2.T)
    dhidden[l2 <= 0] = 0

    grads['W1'] = np.dot(X.T,dhidden)
    grads['b1'] = np.sum(dhidden,axis = 0)

    grads['W2'] +=  reg * W2
    grads['W1'] +=  reg * W1


    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      idx = np.random.choice(np.arange(num_train),batch_size)
      X_batch = X[idx]
      y_batch = y[idx]

      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      self.params['W1'] += -learning_rate * grads['W1']
      self.params['b1'] += -learning_rate * grads['b1']
      self.params['W2'] += -learning_rate * grads['W2']
      self.params['b2'] += -learning_rate * grads['b2']



      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################



      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        if verbose:
          print('iteration %d / %d: loss %f, val acc %f' % (it, num_iters, loss, val_acc))
        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: (TASK C) Implement this function; it should be VERY simple!       #
    ###########################################################################
    l2 = X.dot(self.params['W1']) + self.params['b1']
    a2 = np.maximum(0,l2)
    scores = a2.dot(self.params['W2']) + self.params['b2']
    y_pred = np.argmax(scores, axis=1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred

# In[ ]:


# Create a 3rd dataset, the validation set, for tuning our hyperparameters.

split_percent = (0.25)
split_index = int((1-split_percent)*len(X_train))
X_train, y_train = X_train[:split_index], y_train[:split_index]
X_val, y_val = X_train[split_index:], y_train[split_index:]
################################################################################
# TODO: (Everyone) Play around with the parameters of our network to maximize  #
#                  the validation accuracy.                                    #
################################################################################
hidden_size = 50
num_classes = 4
input_size = 32 * 32 * 1

net = TwoLayerNet(input_size, hidden_size, num_classes)

stats = net.train(X_train, y_train, X_val, y_val,
            num_iters=100, batch_size=64,
            learning_rate=1e-4, learning_rate_decay=0.95,
            reg=0.25, verbose=True)

################################################################################
# TEAM C Task: Evaluate our model using a predict fn and compute test accuracy #
################################################################################

test_acc = (net.predict(X_test) == y_test).mean()
print('Test accuracy: ', test_acc)
