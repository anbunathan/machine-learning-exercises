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

a = tf.constant(2)
b = tf.constant(3)
x = tf.add(a, b)
writer = tf.summary.FileWriter("C:\\temp\\tensorflow_logs\\", tf.get_default_graph())
with tf.Session() as sess:
	# writer = tf.summary.FileWriter('./graphs', sess.graph)
	print(sess.run(x))
	writer.close() # close the writer when you’re done using it