import math
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import matplotlib.pyplot as plt

import utils

DATA_FILE = 'birth_life_2010.txt'

# In order to use eager execution, `tfe.enable_eager_execution()` must be
# called at the very beginning of a TensorFlow program.
tfe.enable_eager_execution()

# Read the data into a dataset.
data, n_samples = utils.read_birth_life_data(DATA_FILE)
dataset = tf.data.Dataset.from_tensor_slices((data[:,0], data[:,1]))

# Create weight and bias variables, initialized to 0.0.
w = tfe.Variable(0.0)
b = tfe.Variable(0.0)

# Define the linear predictor.
def prediction(x):
    return x * w + b

# Define loss functions of the form: L(y, y_predicted)
def squared_loss(y, y_predicted):
    return (y - y_predicted) ** 2

def huber_loss(y, y_predicted, m=1.0):
  t = y - y_predicted
  return t ** 2 if tf.abs(t) <= m else m * (2 * tf.abs(t) - m)

def train(loss_fn):
  """Train a regression model evaluated using `loss_fn`."""
  print('Training; loss function: ' + loss_fn.__name__)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

  # Define the function through which to differentiate.
  def loss_for_example(x, y):
      return loss_fn(y, prediction(x))

  # Obtain a gradients function using `tfe.implicit_value_and_gradients`.
  grad_fn = tfe.implicit_value_and_gradients(loss_for_example)

  start = time.time()
  for epoch in range(100):
    total_loss = 0.0
    for x_i, y_i in tfe.Iterator(dataset):
      # Compute the loss and gradient, and take an optimization step.
      loss, gradients = grad_fn(x_i, y_i)
      optimizer.apply_gradients(gradients)
      total_loss += loss
    if epoch % 10 == 0:
      print('Epoch {0}: {1}'.format(epoch, total_loss / n_samples))
  print('Took: %f seconds' % (time.time() - start))
  print('Eager execution exhibits significant overhead per operation. '
        'As you increase your batch size, the impact of the overhead will '
        'become less noticeable. Eager execution is under active development: '
        'expect performance to increase substantially in the near future!')

train(huber_loss)
plt.plot(data[:,0], data[:,1], 'bo')
# The `.numpy()` method of a tensor retrieves the NumPy array backing it.
# In future versions of eager, you won't need to call `.numpy()` and will
# instead be able to, in most cases, pass Tensors wherever NumPy arrays are
# expected.
plt.plot(data[:,0], data[:,0] * w.numpy() + b.numpy(), 'r',
         label="huber regression")
plt.legend()
plt.show()
