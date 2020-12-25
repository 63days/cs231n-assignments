from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]

    for i in range(num_train):
      logits = X[i].dot(W) #[1, C]
      probs = np.exp(logits) / np.sum(np.exp(logits)) #[1, C]
      dlogits = probs #[1, C]
      for j in range(num_classes):
        if j == y[i]:
          loss += -np.log(probs[j])
          dlogits[j] -= 1
      dW += np.matmul(X[i].T.reshape(-1, 1), dlogits.reshape(1, -1))  

    loss /= num_train
    dW /= num_train

    loss += reg*np.sum(W*W)
    dW += 2 * W * reg
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train, num_classes = X.shape[0], W.shape[1]

    logits = np.dot(X, W)
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1).reshape(-1, 1)
    dlogits = probs #[N, C]
    correct_class_probs = probs[np.arange(num_train), y] #[N, ]
    loss = np.sum(-np.log(correct_class_probs))
    dlogits[np.arange(num_train), y] -= 1
    dW = np.dot(X.T, dlogits) #[D, N]x[N, C]=[D, C]

    loss /= num_train
    dW /= num_train

    loss += reg * np.sum(W*W)
    dW += 2 * W * reg

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
