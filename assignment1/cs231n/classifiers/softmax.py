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

    num_classes = W.shape[1]
    num_train = X.shape[0]

    for i in range(num_train):
      # scores: shape (C,) class scores of a single example
      scores = X[i].dot(W)
      correct_class_score = scores[y[i]]

      # exp_scores: shape (C,) exp class scores of a single example
      exp_scores = np.zeros(num_classes)
      # TODO can this number become far too large (ie scores[j] ~ 20)
      total_exp_score = 0
      for j in range(num_classes):
        exp_scores[j] = np.exp(scores[j])
        total_exp_score += np.exp(scores[j])

      loss += -correct_class_score + np.log(total_exp_score)

      # exp_scores_proportion: shape (C,) for grad calculation
      exp_scores_proportion = exp_scores / total_exp_score

      # grad wrt w_j
      for j in range(num_classes):
        if (j == y[i]):
          dW[:, j] += exp_scores_proportion[j] * X[i] - X[i]
        else:
          dW[:, j] += exp_scores_proportion[j] * X[i]
      
    loss /= num_train
    dW /= num_train

    loss += reg * np.sum(W * W)
    dW += reg * 2 * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    # dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]

    exp_scores = np.exp(np.matmul(X, W))
    total_exp_scores = np.sum(exp_scores, axis=1) # TODO numeric instability?

    # grad wrt w_j for grad calculation and loss calculation
    exp_scores_proportion = exp_scores / total_exp_scores[:, np.newaxis]

    loss += np.sum(-np.log(exp_scores_proportion[np.arange(num_train), y]))
    loss /= num_train
    loss += reg * np.sum(W * W) 

    # grad wrt w_j for j == y[i] has extra -X[i] term in its derivative
    exp_scores_proportion[np.arange(num_train), y] -= 1

    dW = np.matmul(X.T, exp_scores_proportion)
    dW /= num_train
    dW += reg * 2 * W

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
