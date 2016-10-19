import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.

  W = W.T

  X = X.T

  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # Compute the softmax loss and its gradient using explicit loops.           #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  # Get shapes
  num_classes = W.shape[0]
  num_train = X.shape[1]

  for i in range(num_train):
    # Compute vector of scores
    f_i = W.dot(X[:, i]) # in R^{num_classes}

    # Normalization trick to avoid numerical instability, per http://cs231n.github.io/linear-classify/#softmax
    log_c = np.max(f_i)
    f_i -= log_c

    # Compute loss (and add to it, divided later)
    # L_i = - f(x_i)_{y_i} + log \sum_j e^{f(x_i)_j}
    sum_i = 0.0
    for f_i_j in f_i:
      sum_i += np.exp(f_i_j)
    loss += -f_i[y[i]] + np.log(sum_i)

    # Compute gradient
    # dw_j = 1/num_train * \sum_i[x_i * (p(y_i = j)-Ind{y_i = j} )]
    # Here we are computing the contribution to the inner sum for a given i.
    for j in range(num_classes):
      p = np.exp(f_i[j])/sum_i
      if j == y[i]:
        p -= 1
      temp = p* X[:,i]

      dW[j, :] += temp

  # Compute average
  loss /= num_train
  dW /= num_train

  # Regularization
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W

  return loss, dW


def softmax_loss_naive1(W, X, y, reg):
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

  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  #scores -= np.max(scores, axis = 1)
  scores = np.exp(scores)
  total_scores_per_input = np.sum(scores,axis = 1).reshape(num_train,-1)

  scores /=total_scores_per_input

  correct_class_score = scores[range(num_train),y]

  loss_cost = -np.sum(np.log(correct_class_score))/num_train
  loss_reg = 0.5 * reg * np.sum(W * W)
  loss = loss_cost+loss_reg


  target = np.zeros_like(scores)
  target[range(num_train),y] = 1
  #or
  #dscores = p
  #sscores[range(num_train),y]-=1

  dW = X.T.dot(scores- target)/num_train + reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorizted version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  #scores -= np.max(scores, axis = 1)
  scores = np.exp(scores)
  total_scores_per_input = np.sum(scores,axis = 1).reshape(num_train,-1)

  scores /=total_scores_per_input

  correct_class_score = scores[range(num_train),y]

  loss_cost = -np.sum(np.log(correct_class_score))/num_train
  loss_reg = 0.5 * reg * np.sum(W * W)
  loss = loss_cost+loss_reg


  target = np.zeros_like(scores)
  target[range(num_train),y] = 1
  #or
  #dscores = p
  #sscores[range(num_train),y]-=1

  dW = X.T.dot(scores- target)/num_train + reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


if __name__ == "__main__":
  W = np.array([[1,2,3,2,1],[4,5,6,2,1],[7,8,9,5,7]],dtype=np.float64) # C*D
  W= W.T#D*C
  X = np.array([[1,0,0,1,0],[0,1,0,0,1],[0,0,1,1,0],[1,0,0,0,1]],dtype=np.float64)

  y = [0,1,2,0]

  reg = 0.1

  loss , grad =  softmax_loss_naive(W,X,y,reg)

  print loss



