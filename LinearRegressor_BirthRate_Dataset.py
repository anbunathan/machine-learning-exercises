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

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import utils

DATA_FILE = 'birth_life_2010.txt'

# Step 1: read in data from the .txt file
data, n_samples = utils.read_birth_life_data(DATA_FILE)

# Step 2: create placeholders for X (birth rate) and Y (life expectancy)
# Remember both X and Y are scalars with type float
dataset = tf.data.Dataset.from_tensor_slices((data[:,0], data[:,1]))
iterator = dataset.make_initializable_iterator()
X, Y = iterator.get_next()

# Step 3: create weight and bias, initialized to 0.0
# Make sure to use tf.get_variable
w = tf.get_variable('weights', initializer=tf.constant(0.0))
b = tf.get_variable('bias', initializer=tf.constant(0.0))

# Step 4: build model to predict Y
# e.g. how would you derive at Y_predicted given X, w, and b
Y_predicted = w * X + b

# Step 5: use the square error as the loss function
loss = tf.square(Y - Y_predicted, name='loss')

# Step 6: using gradient descent with learning rate of 0.001 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

start = time.time()

with tf.Session() as sess:
    # Step 7: initialize the necessary variables, in this case, w and b
    sess.run(tf.global_variables_initializer())
    # Create a filewriter to write the model's graph to TensorBoard
    writer = tf.summary.FileWriter("C:\\temp\\tensorflow_logs\\", sess.graph)

    # Step 8: train the model for 100 epochs
    for i in range(100):
        sess.run(iterator.initializer)  # initialize the iterator
        total_loss = 0
        try:
            while True:
                _, l = sess.run([optimizer, loss])
                total_loss += l
        except tf.errors.OutOfRangeError:
            pass

        print('Epoch {0}: {1}'.format(i, total_loss / n_samples))

    # close the writer when you're done using it

    writer.close()

    # Step 9: output the values of w and b
    w_out, b_out = sess.run([w, b])


print('Took: %f seconds' % (time.time() - start))

# uncomment the following lines to see the plot
plt.plot(data[:,0], data[:,1], 'bo', label='Real data')
plt.plot(data[:,0], data[:,0] * w_out + b_out, 'r', label='Predicted data with squared error')
plt.legend()
plt.show()